#!/usr/bin/env nextflow
nextflow.enable.dsl=2

import groovy.yaml.YamlSlurper
import java.util.regex.Pattern

params.config = params.config ?: "${projectDir}/asv_pipeline_nextflow.yml"

def configFile = file(params.config)
if( !configFile.exists() ) {
    exit 1, "Config file not found: ${configFile}"
}

def config = new YamlSlurper().parse(configFile)
def configRoot = configFile.parentFile ?: new File('.')

def resolvePath(String pathValue) {
    if( !pathValue ) {
        return null
    }
    def candidate = new File(pathValue)
    if( candidate.isAbsolute() ) {
        return candidate.canonicalPath
    }
    return new File(configRoot, pathValue).canonicalPath
}

def r1Tokens = normalizeList(config.filename_patterns?.r1_tokens, ['R1','1'])
def r2Tokens = normalizeList(config.filename_patterns?.r2_tokens, ['R2','2'])
assert r1Tokens.size() == r2Tokens.size() : "R1 token count (${r1Tokens.size()}) must match R2 token count (${r2Tokens.size()})"

def extPatterns = compilePatterns(config.filename_patterns?.ext_patterns, ['\\.fastq\\.gz$','\\.fq\\.gz$','\\.fastq$','\\.fq$'])
def stripRegex = config.filename_patterns?.sample_strip_regex ?: '(_S[0-9]+)?(_L[0-9]{3})?(_R[12])?(_[12])?(_001)?$'

def inputDir = resolvePath(config.paths?.input_dir)
def outputDir = resolvePath(config.paths?.output_dir)
assert inputDir : "paths.input_dir must be provided in the YAML config"
assert outputDir : "paths.output_dir must be provided in the YAML config"

def allowSingleEnd = (config.resources?.single_end ?: false) as boolean
def configuredThreads = config.resources?.threads
int pipelineThreads = (configuredThreads ? configuredThreads as int : Runtime.runtime.availableProcessors())

def dirMap = [
    fastp    : "${outputDir}/fastp",
    merge    : "${outputDir}/merged",
    filter   : "${outputDir}/filtered",
    concat   : "${outputDir}/concat",
    derep    : "${outputDir}/derep",
    denoise  : "${outputDir}/denoise",
    nochi    : "${outputDir}/nochimeras",
    swarm    : "${outputDir}/swarm",
    asv      : "${outputDir}/ASVs",
    logs     : "${outputDir}/logs"
]
new File(dirMap.concat).mkdirs()
new File(dirMap.logs).mkdirs()

def envConfigPath = config.environments?.main
def resolvedEnvPath = envConfigPath ? resolveOptionalPath(envConfigPath, configRoot) : null
def defaultEnvPath = new File("${projectDir}/envs/asv_pipeline.yml").canonicalPath
def condaEnvPath = resolvedEnvPath ?: defaultEnvPath
def condaEnvFile = file(condaEnvPath)
if( !condaEnvFile.exists() ) {
    exit 1, "Conda environment YAML not found: ${condaEnvPath}"
}
log.info "Using Conda/Mamba env definition: ${condaEnvPath}"

def steps = config.steps ?: [:]
boolean skipFastp   = (steps.skip_fastp   ?: false) as boolean
boolean skipMerge   = (steps.skip_merge   ?: false) as boolean
boolean skipFilter  = (steps.skip_filter  ?: false) as boolean
boolean skipConcat  = (steps.skip_concat  ?: false) as boolean
boolean skipDerep   = (steps.skip_derep   ?: false) as boolean
boolean skipDenoise = (steps.skip_denoise ?: false) as boolean
boolean skipChimera = (steps.skip_chimera ?: false) as boolean
boolean skipSwarm   = (steps.skip_swarm   ?: false) as boolean
boolean skipCount   = (steps.skip_count   ?: false) as boolean
boolean skipTabFilt = (steps.skip_tabfilt ?: false) as boolean

def sampleRecords = collectSampleRecords(inputDir, r1Tokens, r2Tokens, extPatterns, stripRegex, allowSingleEnd)
if( !sampleRecords ) {
    exit 1, "No usable FASTQ files detected in ${inputDir}"
}
log.info "Discovered ${sampleRecords.size()} input items (paired + single-end as allowed). Threads=${pipelineThreads}"

Channel
    .from(sampleRecords)
    .map { rec ->
        def meta = [ sample_id: rec.sample_id, paired: rec.paired ]
        def r1 = file(rec.r1)
        def r2 = rec.paired ? file(rec.r2) : null
        tuple(meta, r1, r2)
    }
    .set { raw_reads }

def tabFilterScript = resolveOptionalPath(config.table_filter?.script, configRoot) ?: "${projectDir}/filter_ASV_table.py"
def tableScriptFile = file(tabFilterScript)
if( !tableScriptFile.exists() ) {
    exit 1, "Table filter script not found: ${tabFilterScript}"
}

workflow {
    def reads_after_qc    = skipFastp  ? raw_reads : FASTP_QC(raw_reads)
    def reads_after_merge = skipMerge  ? reads_after_qc : MERGE_READS(reads_after_qc)
    def reads_after_filter= skipFilter ? reads_after_merge : FILTER_READS(reads_after_merge)
    def relabel_source    = skipConcat ? reads_after_filter : RELABEL_FASTA(reads_after_filter)

    def concat_input = relabel_source.map { tuple -> tuple[1] }
    def concat_fasta = concat_input.collectFile(name: 'concat.fasta', storeDir: dirMap.concat, newLine: true)

    concat_fasta.into { concat_for_derep; concat_for_counts }

    def derep_input  = skipDerep   ? concat_for_derep : DEREPLICATE(concat_for_derep)
    def denoise_input= skipDenoise ? derep_input      : DENOISE(derep_input)
    def nochi_input  = skipChimera ? denoise_input    : CHIMERA_CHECK(denoise_input)

    nochi_input.into { nochi_for_counts; nochi_for_swarm }

    if( !skipSwarm && (config.swarm?.enabled ?: true) ) {
        SWARM_CLUSTER(nochi_for_swarm)
    }

    if( !skipCount ) {
        def count_ready = concat_for_counts.combine(nochi_for_counts).map { tuple(it[0], it[1]) }
        def asv_outputs = CREATE_COUNT_MATRIX(count_ready)
        if( !skipTabFilt ) {
            FILTER_TABLE(asv_outputs)
        } else {
            log.warn "Skipping table filtering step as requested."
        }
    } else if( !skipTabFilt ) {
        log.warn "Cannot run table filtering because ASV table creation was skipped."
    }
}

process FASTP_QC {
    tag { meta.sample_id }
    cpus pipelineThreads
    conda "${condaEnvPath}"
    publishDir dirMap.fastp, mode: 'copy', pattern: '*', saveAs: { filename ->
        if( filename == 'R1.fastq.gz' ) {
            return "${meta.sample_id}_R1.fastq.gz"
        }
        if( filename == 'R2.fastq.gz' ) {
            return "${meta.sample_id}_R2.fastq.gz"
        }
        if( filename == 'fastp.json' ) {
            return "${meta.sample_id}.fastp.json"
        }
        if( filename == 'fastp.html' ) {
            return "${meta.sample_id}.fastp.html"
        }
        return filename
    }

    input:
    tuple val(meta), path(r1), path(r2, optional: true)

    output:
    tuple val(meta), path("R1.fastq.gz"), path("R2.fastq.gz", optional: true)
    path("fastp.json")
    path("fastp.html")

    script:
    def fastpCfg = config.fastp ?: [:]
    def trim = { key, fallback -> fastpCfg[key] != null ? fastpCfg[key] : fallback }
    def pairedArgs = meta.paired ? """
fastp \\
  -i "${r1}" -I "${r2}" \\
  -o R1.fastq.gz \\
  -O R2.fastq.gz \\
  -f ${trim('trim_front_r1',0)} -t ${trim('trim_tail_r1',0)} \\
  -F ${trim('trim_front_r2',0)} -T ${trim('trim_tail_r2',0)} \\
  -j fastp.json \\
  -h fastp.html \\
  -w ${task.cpus}
""" : """
fastp \\
  -i "${r1}" \\
  -o R1.fastq.gz \\
  -f ${trim('trim_front_r1',0)} -t ${trim('trim_tail_r1',0)} \\
  -j fastp.json \\
  -h fastp.html \\
  -w ${task.cpus}
"""
    pairedArgs
}

process MERGE_READS {
    tag { meta.sample_id }
    cpus pipelineThreads
    conda "${condaEnvPath}"
    publishDir dirMap.merge, mode: 'copy', saveAs: { filename ->
        filename == 'merged.fastq' ? "${meta.sample_id}.merged.fastq" : filename
    }

    input:
    tuple val(meta), path(r1), path(r2, optional: true)

    output:
    tuple val(meta), path("merged.fastq")

    script:
    def mergeCfg = config.merge ?: [:]
    def allowStagger = mergeCfg.allow_stagger ? '--fastq_allowmergestagger' : ''
    if( meta.paired && r2 ) {
        """
vsearch --fastq_mergepairs "${r1}" \\
        --reverse "${r2}" \\
        --fastqout merged.fastq \\
        --fastq_maxdiffs ${mergeCfg.max_diffs ?: 20} \\
        --fastq_minovlen ${mergeCfg.min_overlap ?: 5} \\
        --fastq_truncqual ${mergeCfg.trunc_quality ?: 5} \\
        ${allowStagger} \\
        --threads ${task.cpus}
"""
    } else {
        """
if [[ "${r1}" == *.gz ]]; then
  gunzip -c "${r1}" > merged.fastq
else
  cat "${r1}" > merged.fastq
fi
"""
    }
}

process FILTER_READS {
    tag { meta.sample_id }
    cpus pipelineThreads
    conda "${condaEnvPath}"
    publishDir dirMap.filter, mode: 'copy', saveAs: { filename ->
        filename == 'filtered.fasta' ? "${meta.sample_id}.filtered.fasta" : filename
    }

    input:
    tuple val(meta), path(merged_fastq)

    output:
    tuple val(meta), path("filtered.fasta")

    script:
    def filterCfg = config.filter ?: [:]
    """
vsearch --fastx_filter "${merged_fastq}" \\
        --fastq_maxee ${filterCfg.max_ee ?: 1.0} \\
        --fastq_minlen ${filterCfg.min_len ?: 245} \\
        --fastq_maxlen ${filterCfg.max_len ?: 1500} \\
        --fastaout filtered.fasta
"""
}

process RELABEL_FASTA {
    tag { meta.sample_id }
    conda "${condaEnvPath}"
    publishDir dirMap.filter, mode: 'copy', saveAs: { filename ->
        filename == 'relabeled.fasta' ? "${meta.sample_id}.relabeled.fasta" : filename
    }

    input:
    tuple val(meta), path(filtered_fasta)

    output:
    tuple val(meta), path("relabeled.fasta")

    """
python - <<'PY' '${meta.sample_id}' '${filtered_fasta}' 'relabeled.fasta'
import sys
from pathlib import Path
sample, src_path, dst_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
with src_path.open() as src, dst_path.open('w') as dst:
    for line in src:
        if line.startswith('>'):
            clean = line[1:].strip()
            if ':' in clean:
                clean = clean.split(':', 1)[1]
            dst.write(f'>{sample}:{clean}\\n')
        else:
            dst.write(line)
PY
"""
}

process DEREPLICATE {
    cpus pipelineThreads
    conda "${condaEnvPath}"
    publishDir dirMap.derep, mode: 'copy', pattern: '*'

    input:
    path(concat_fasta)

    output:
    path("derep.fasta")

    script:
    """
vsearch --derep_fulllength "${concat_fasta}" \\
        --output derep.fasta \\
        --sizeout \\
        --threads ${task.cpus} \\
        --log "${dirMap.logs}/derep.log"
"""
}

process DENOISE {
    cpus pipelineThreads
    conda "${condaEnvPath}"
    publishDir dirMap.denoise, mode: 'copy', pattern: '*'

    input:
    path(derep_fasta)

    output:
    path("centroids.fasta")

    script:
    def unoiseCfg = config.unoise ?: [:]
    """
vsearch --cluster_unoise "${derep_fasta}" \\
        --centroids centroids.fasta \\
        --sizein --sizeout --relabel ASV \\
        --minsize ${unoiseCfg.min_size ?: 3} \\
        --threads ${task.cpus} \\
        --log "${dirMap.logs}/denoise.log"
"""
}

process CHIMERA_CHECK {
    cpus pipelineThreads
    conda "${condaEnvPath}"
    publishDir dirMap.nochi, mode: 'copy', pattern: '*'

    input:
    path(centroids)

    output:
    path("nochimeras.fasta")

    script:
    """
vsearch --uchime3_denovo "${centroids}" \\
        --nonchimeras nochimeras.fasta \\
        --sizein \\
        --threads ${task.cpus} \\
        --log "${dirMap.logs}/nochimera.log"
"""
}

process SWARM_CLUSTER {
    cpus pipelineThreads
    conda "${condaEnvPath}"
    publishDir dirMap.swarm, mode: 'copy', pattern: '*'

    input:
    path(nochimeras)

    output:
    path("swarm_reps.fasta")
    path("swarm_reps.struct")
    path("swarm_reps.network")
    path("swarm_reps.stats.txt")
    path("swarm_reps.swarms")

    script:
    def swarmCfg = config.swarm ?: [:]
    """
swarm -d ${swarmCfg.distance ?: 1} -f -t ${task.cpus} -z \\
      -i swarm_reps.struct \\
      -j swarm_reps.network \\
      -s swarm_reps.stats.txt \\
      -w swarm_reps.fasta \\
      -o swarm_reps.swarms \\
      "${nochimeras}"
"""
}

process CREATE_COUNT_MATRIX {
    cpus pipelineThreads
    conda "${condaEnvPath}"
    publishDir dirMap.asv, mode: 'copy', pattern: '*'

    input:
    tuple path(concat_fasta), path(nochimeras)

    output:
    tuple path("ASV_counts.tsv"), path("ASVs.fasta")

    script:
    """
cp "${nochimeras}" ASVs.fasta
vsearch --usearch_global "${concat_fasta}" \\
        --db ASVs.fasta \\
        --id 0.999 \\
        --otutabout ASV_counts.tsv \\
        --threads ${task.cpus} \\
        --log "${dirMap.logs}/count.log"
"""
}

process FILTER_TABLE {
    conda "${condaEnvPath}"
    publishDir dirMap.asv, mode: 'copy', pattern: '*'

    input:
    tuple path(count_table), path(asv_fasta)

    output:
    path("ASV_filtered.tsv")
    path("ASVs_filtered.fasta")

    script:
    def tableCfg = config.table_filter ?: [:]
    """
python "${tableScriptFile}" \\
       "${count_table}" \\
       ASV_filtered.tsv \\
       ${tableCfg.min_sample_sum ?: 5000} \\
       ${tableCfg.min_asv_sum ?: 0.01} \\
       "${asv_fasta}" \\
       ASVs_filtered.fasta
"""
}

/**
 * Helpers
 */
def normalizeList(value, fallback){
    if( !value ) return fallback
    if( value instanceof List ) return value.collect { it.toString() }
    return value.toString().split(/\|/).collect { it.trim() }.findAll { it }
}

def compilePatterns(value, fallback){
    def list = value ?: fallback
    return list.collect { Pattern.compile(it.toString()) }
}

def matchesExtension(String name, List<Pattern> patterns){
    patterns.any { it.matcher(name).find() }
}

def isR1Like(String base, List<String> tokens){
    tokens.any { tok ->
        def rx = /(^|[_\.\-])${Pattern.quote(tok)}([_\.\-]|$)/
        base ==~ /.*${rx}.*/
    }
}

def sampleFromName(String baseName, String stripRegex, List<Pattern> extPatterns){
    def base = baseName
    extPatterns.each { base = base.replaceAll(it, '') }
    base = base.replaceAll(stripRegex, '')
    base = base.replaceAll(/[_\-.]+$/, '')
    return base
}

def findR2File(File r1File, List<String> r1Tokens, List<String> r2Tokens){
    def original = r1File.name
    for( int i=0; i<r1Tokens.size(); i++ ){
        def r1 = r1Tokens[i]
        def r2 = r2Tokens[i]
        def replacements = [
            [/_${Pattern.quote(r1)}_/, "_${r2}_"],
            [/\.${Pattern.quote(r1)}\./, ".${r2}."],
            [/-${Pattern.quote(r1)}-/, "-${r2}-"],
            [/-${Pattern.quote(r1)}\./, "-${r2}."],
            [/_${Pattern.quote(r1)}\./, "_${r2}."],
            [/_${Pattern.quote(r1)}$/, "_${r2}"],
            [/${Pattern.quote(r1)}_001/, "${r2}_001"]
        ]
        for( rep in replacements ){
            def candidateName = original.replaceFirst(rep[0], rep[1])
            if( candidateName != original ){
                def candidate = new File(r1File.parentFile, candidateName)
                if( candidate.exists() ) {
                    return candidate
                }
            }
        }
    }
    return null
}

def collectSampleRecords(String inputDirPath, List<String> r1Tokens, List<String> r2Tokens,
                         List<Pattern> extPatterns, String stripRegex, boolean allowSingleEnd){
    File dir = new File(inputDirPath)
    if( !dir.exists() ){
        throw new IllegalArgumentException("Input directory does not exist: ${inputDirPath}")
    }
    def files = dir.listFiles()?.findAll { it.isFile() && matchesExtension(it.name, extPatterns) }?.sort { a, b -> a.name <=> b.name } ?: []
    def records = []
    files.each { file ->
        if( isR1Like(file.name, r1Tokens) ){
            def sampleId = sampleFromName(file.name, stripRegex, extPatterns)
            def r2File = findR2File(file, r1Tokens, r2Tokens)
            if( r2File ){
                records << [ sample_id: sampleId, paired: true, r1: file.canonicalPath, r2: r2File.canonicalPath ]
            } else if( allowSingleEnd ) {
                records << [ sample_id: sampleId, paired: false, r1: file.canonicalPath ]
            } else {
                log.warn "Skipping ${file.name}: no matching R2 detected."
            }
        }
    }
    return records
}

def resolveOptionalPath(String pathValue, File baseDir){
    if( !pathValue ) return null
    def candidate = new File(pathValue)
    if( candidate.isAbsolute() ) {
        return candidate.canonicalPath
    }
    return new File(baseDir ?: new File('.'), pathValue).canonicalPath
}
