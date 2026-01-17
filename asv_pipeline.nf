#!/usr/bin/env nextflow
nextflow.enable.dsl=2

import groovy.yaml.YamlSlurper
import java.net.URL
import java.nio.file.Paths
import java.util.regex.Pattern
import java.util.zip.GZIPInputStream

def projectRootDir = new File(projectDir.toString())
def defaultConfigPath = "${projectDir}/asv_pipeline_nextflow.yml"
def paramsMap = [:]

try {
    paramsMap = workflow.params ? new LinkedHashMap(workflow.params) : [:]
} catch( Throwable ignored ) {
    paramsMap = [:]
}

def inlineKeys = [
    'paths','resources','fastp','merge','filter','unoise',
    'table_filter','filename_patterns','environments','config_root'
]
def hasInlineConfig = inlineKeys.any { paramsMap.containsKey(it) }

def config
File configFile = null
File configRoot = projectRootDir

def providedConfigPath = paramsMap.containsKey('config') ? paramsMap.config : null

if( providedConfigPath ) {
    def configPath = file(providedConfigPath)
    configFile = configPath.toFile()
    if( !configFile.exists() ) {
        exit 1, "Config file not found: ${configFile}"
    }
    config = new YamlSlurper().parse(configFile)
    configRoot = configFile.parentFile ?: projectRootDir
    log.info "Loaded config from ${configFile}"
} else if( hasInlineConfig ) {
    config = paramsMap
    if( paramsMap.containsKey('config_root') ) {
        def rootPath = file(paramsMap.config_root)
        configRoot = rootPath.toFile()
    }
    log.info "Using inline Nextflow parameters as configuration."
} else {
    def configPath = file(defaultConfigPath)
    configFile = configPath.toFile()
    if( !configFile.exists() ) {
        exit 1, "Config file not found: ${defaultConfigPath}"
    }
    config = new YamlSlurper().parse(configFile)
    configRoot = configFile.parentFile ?: projectRootDir
    log.info "Loaded default config from ${configFile}"
}

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
def workDirOverride = config.paths?.work_dir ? resolvePath(config.paths.work_dir) : null
def condaCacheOverride = config.paths?.conda_cache_dir ? resolvePath(config.paths.conda_cache_dir) : null
assert inputDir : "paths.input_dir must be provided in the YAML config"
assert outputDir : "paths.output_dir must be provided in the YAML config"
if( workDirOverride ) {
    def workDirFile = new File(workDirOverride)
    workDirFile.mkdirs()
    workflow.workDir = Paths.get(workDirFile.canonicalPath)
    log.info "Using custom Nextflow work directory: ${workflow.workDir}"
}
def resolvedCondaCacheDir = condaCacheOverride ?: new File(outputDir, ".conda_cache").canonicalPath
def condaCacheDirFile = new File(resolvedCondaCacheDir)
condaCacheDirFile.mkdirs()
System.setProperty('NXF_CONDA_CACHEDIR', condaCacheDirFile.canonicalPath)
log.info "Using custom Conda cache directory: ${condaCacheDirFile.canonicalPath}"

def allowSingleEnd = (config.resources?.single_end ?: false) as boolean
int hostThreads = Runtime.runtime.availableProcessors()
int sampleThreads = config.resources?.threads ? (config.resources.threads as int) : hostThreads
int pipelineThreads = hostThreads

if( config.fastp && !(config.fastp instanceof Map) ) {
    log.warn "Ignoring non-map fastp configuration (${config.fastp.getClass()?.simpleName})"
}
def fastpConfigMap = (config.fastp instanceof Map) ? config.fastp : [:]
def fastpTrimValues = [
    front_r1: fastpConfigMap.trim_front_r1 != null ? (fastpConfigMap.trim_front_r1 as int) : 0,
    tail_r1 : fastpConfigMap.trim_tail_r1  != null ? (fastpConfigMap.trim_tail_r1  as int) : 0,
    front_r2: fastpConfigMap.trim_front_r2 != null ? (fastpConfigMap.trim_front_r2 as int) : 0,
    tail_r2 : fastpConfigMap.trim_tail_r2  != null ? (fastpConfigMap.trim_tail_r2  as int) : 0
]

def dirMap = [
    fastp    : "${outputDir}/fastp",
    merge    : "${outputDir}/merged",
    filter   : "${outputDir}/filtered",
    concat   : "${outputDir}/concat",
    derep    : "${outputDir}/derep",
    sina     : "${outputDir}/sina",
    denoise  : "${outputDir}/denoise",
    nochi    : "${outputDir}/nochimeras",
    asv      : "${outputDir}/ASVs",
    mito     : "${outputDir}/mito",
    taxonomy : "${outputDir}/taxonomy",
    stats    : "${outputDir}/stats",
    logs     : "${outputDir}/logs"
]
new File(dirMap.concat).mkdirs()
new File(dirMap.sina).mkdirs()
new File(dirMap.asv).mkdirs()
new File(dirMap.mito).mkdirs()
new File(dirMap.taxonomy).mkdirs()
new File(dirMap.stats).mkdirs()
new File(dirMap.logs).mkdirs()

def sinaReferenceFilename = 'SILVA_138.2_SSURef_NR99_03_07_24_opt.arb'
def defaultSinaReferenceUrl = 'https://www.arb-silva.de/fileadmin/silva_databases/current/Exports/SILVA_138.2_SSURef_NR99_03_07_24_opt.arb.gz'
def defaultTaxonomyTaxUrl = 'https://data.qiime2.org/2024.10/common/silva-138-99-tax.qza'
def defaultTaxonomySeqsUrl = 'https://data.qiime2.org/2024.10/common/silva-138-99-seqs.qza'
def defaultTaxonomyTaxFilename = 'silva-138_2-ssu-nr99-tax.qza'
def defaultTaxonomySeqsFilename = 'silva-138_2-ssu-nr99-seqs-DNA.qza'

def envConfigPath = config.environments?.main
def resolvedEnvPath = envConfigPath ? resolveOptionalPath(envConfigPath, configRoot) : null
def defaultEnvPath = new File("${projectDir}/envs/asv_pipeline.yml").canonicalPath
def condaEnvPath = resolvedEnvPath ?: defaultEnvPath
def condaEnvFile = file(condaEnvPath)
if( !condaEnvFile.exists() ) {
    exit 1, "Conda environment YAML not found: ${condaEnvPath}"
}
log.info "Using Conda/Mamba env definition: ${condaEnvPath}"

def sinaEnvConfigPath = config.environments?.sina
def resolvedSinaEnvPath = sinaEnvConfigPath ? resolveOptionalPath(sinaEnvConfigPath, configRoot) : null
def defaultSinaEnvPath = new File("${projectDir}/envs/sina.yml").canonicalPath
def sinaCondaEnvPath = resolvedSinaEnvPath ?: defaultSinaEnvPath
def sinaEnvFile = file(sinaCondaEnvPath)
if( !sinaEnvFile.exists() ) {
    exit 1, "SINA conda environment YAML not found: ${sinaCondaEnvPath}"
}
log.info "Using SINA Conda/Mamba env definition: ${sinaCondaEnvPath}"

def taxonomyEnvConfigPath = config.environments?.taxonomy
def resolvedTaxonomyEnvPath = taxonomyEnvConfigPath ? resolveOptionalPath(taxonomyEnvConfigPath, configRoot) : null
def defaultTaxonomyEnvPath = new File("${projectDir}/envs/qiime2.yml").canonicalPath
def taxonomyCondaEnvPath = resolvedTaxonomyEnvPath ?: defaultTaxonomyEnvPath
def taxonomyEnvFile = file(taxonomyCondaEnvPath)
if( !taxonomyEnvFile.exists() ) {
    exit 1, "Taxonomy conda environment YAML not found: ${taxonomyCondaEnvPath}"
}
log.info "Using taxonomy Conda/Mamba env definition: ${taxonomyCondaEnvPath}"

def mitomasterEnvConfigPath = config.environments?.mitomaster
def resolvedMitomasterEnvPath = mitomasterEnvConfigPath ? resolveOptionalPath(mitomasterEnvConfigPath, configRoot) : null
def defaultMitomasterEnvPath = new File("${projectDir}/envs/mitomaster.yml").canonicalPath
def mitomasterCondaEnvPath = resolvedMitomasterEnvPath ?: defaultMitomasterEnvPath
def mitomasterEnvFile = file(mitomasterCondaEnvPath)
if( !mitomasterEnvFile.exists() ) {
    exit 1, "MITOMASTER conda environment YAML not found: ${mitomasterCondaEnvPath}"
}
log.info "Using MITOMASTER Conda/Mamba env definition: ${mitomasterCondaEnvPath}"

def mitoCheckerEnvConfigPath = config.environments?.mito_checker
def resolvedMitoCheckerEnvPath = mitoCheckerEnvConfigPath ? resolveOptionalPath(mitoCheckerEnvConfigPath, configRoot) : null
def defaultMitoCheckerEnvPath = new File("${projectDir}/envs/mito_checker.yml").canonicalPath
def mitoCheckerCondaEnvPath = resolvedMitoCheckerEnvPath ?: defaultMitoCheckerEnvPath
def mitoCheckerEnvFile = file(mitoCheckerCondaEnvPath)
if( !mitoCheckerEnvFile.exists() ) {
    exit 1, "Mito checker conda environment YAML not found: ${mitoCheckerCondaEnvPath}"
}
log.info "Using mito checker Conda/Mamba env definition: ${mitoCheckerCondaEnvPath}"

def filterCountsEnvConfigPath = config.environments?.filter_counts
def resolvedFilterCountsEnvPath = filterCountsEnvConfigPath ? resolveOptionalPath(filterCountsEnvConfigPath, configRoot) : null
def defaultFilterCountsEnvPath = new File("${projectDir}/envs/filter_counts.yml").canonicalPath
def filterCountsCondaEnvPath = resolvedFilterCountsEnvPath ?: defaultFilterCountsEnvPath
def filterCountsEnvFile = file(filterCountsCondaEnvPath)
if( !filterCountsEnvFile.exists() ) {
    exit 1, "filter_counts conda environment YAML not found: ${filterCountsCondaEnvPath}"
}
log.info "Using filter_counts Conda/Mamba env definition: ${filterCountsCondaEnvPath}"

def generalStatsEnvConfigPath = config.environments?.general_stats
def resolvedGeneralStatsEnvPath = generalStatsEnvConfigPath ? resolveOptionalPath(generalStatsEnvConfigPath, configRoot) : null
def defaultGeneralStatsEnvPath = new File("${projectDir}/envs/general_stats.yml").canonicalPath
def generalStatsCondaEnvPath = resolvedGeneralStatsEnvPath ?: defaultGeneralStatsEnvPath
def generalStatsEnvFile = file(generalStatsCondaEnvPath)
if( !generalStatsEnvFile.exists() ) {
    exit 1, "general_stats conda environment YAML not found: ${generalStatsCondaEnvPath}"
}
log.info "Using general_stats Conda/Mamba env definition: ${generalStatsCondaEnvPath}"

def sankeyEnvConfigPath = config.environments?.sankey
def resolvedSankeyEnvPath = sankeyEnvConfigPath ? resolveOptionalPath(sankeyEnvConfigPath, configRoot) : null
def defaultSankeyEnvPath = new File("${projectDir}/envs/sankey.yml").canonicalPath
def sankeyCondaEnvPath = resolvedSankeyEnvPath ?: defaultSankeyEnvPath
def sankeyEnvFile = file(sankeyCondaEnvPath)
if( !sankeyEnvFile.exists() ) {
    exit 1, "Sankey conda environment YAML not found: ${sankeyCondaEnvPath}"
}
log.info "Using Sankey Conda/Mamba env definition: ${sankeyCondaEnvPath}"

def plotMetadataEnvConfigPath = config.environments?.plot_metadata
def resolvedPlotMetadataEnvPath = plotMetadataEnvConfigPath ? resolveOptionalPath(plotMetadataEnvConfigPath, configRoot) : null
def defaultPlotMetadataEnvPath = new File("${projectDir}/envs/plot_metadata.yml").canonicalPath
def plotMetadataCondaEnvPath = resolvedPlotMetadataEnvPath ?: defaultPlotMetadataEnvPath
def plotMetadataEnvFile = file(plotMetadataCondaEnvPath)
if( !plotMetadataEnvFile.exists() ) {
    exit 1, "Plot metadata conda environment YAML not found: ${plotMetadataCondaEnvPath}"
}
log.info "Using plot metadata Conda/Mamba env definition: ${plotMetadataCondaEnvPath}"

def batchCorrectionEnvConfigPath = config.environments?.batch_correction
def resolvedBatchCorrectionEnvPath = batchCorrectionEnvConfigPath ? resolveOptionalPath(batchCorrectionEnvConfigPath, configRoot) : null
def defaultBatchCorrectionEnvPath = new File("${projectDir}/envs/asv_batch_correction.yml").canonicalPath
def batchCorrectionCondaEnvPath = resolvedBatchCorrectionEnvPath ?: defaultBatchCorrectionEnvPath
def batchCorrectionEnvFile = file(batchCorrectionCondaEnvPath)
if( !batchCorrectionEnvFile.exists() ) {
    exit 1, "Batch correction conda environment YAML not found: ${batchCorrectionCondaEnvPath}"
}
log.info "Using batch correction Conda/Mamba env definition: ${batchCorrectionCondaEnvPath}"

def outlierEnvConfigPath = config.environments?.outlier_checker
def resolvedOutlierEnvPath = outlierEnvConfigPath ? resolveOptionalPath(outlierEnvConfigPath, configRoot) : null
def defaultOutlierEnvPath = new File("${projectDir}/envs/outlier_checker.yml").canonicalPath
def outlierCondaEnvPath = resolvedOutlierEnvPath ?: defaultOutlierEnvPath
def outlierEnvFile = file(outlierCondaEnvPath)
if( !outlierEnvFile.exists() ) {
    exit 1, "Outlier checker conda environment YAML not found: ${outlierCondaEnvPath}"
}
log.info "Using outlier checker Conda/Mamba env definition: ${outlierCondaEnvPath}"

def collectorsEnvConfigPath = config.environments?.collectors_curve
def resolvedCollectorsEnvPath = collectorsEnvConfigPath ? resolveOptionalPath(collectorsEnvConfigPath, configRoot) : null
def defaultCollectorsEnvPath = new File("${projectDir}/envs/collectors_curve.yml").canonicalPath
def collectorsCondaEnvPath = resolvedCollectorsEnvPath ?: defaultCollectorsEnvPath
def collectorsEnvFile = file(collectorsCondaEnvPath)
if( !collectorsEnvFile.exists() ) {
    exit 1, "Collectors curve conda environment YAML not found: ${collectorsCondaEnvPath}"
}
log.info "Using collectors curve Conda/Mamba env definition: ${collectorsCondaEnvPath}"

def manifestPath = config.paths?.manifest ? resolveOptionalPath(config.paths.manifest, configRoot) : null
def sampleRecords
if( manifestPath ) {
    sampleRecords = loadManifestSamples(manifestPath)
} else {
    sampleRecords = collectSampleRecords(inputDir, r1Tokens, r2Tokens, extPatterns, stripRegex, allowSingleEnd)
}
if( !sampleRecords ) {
    exit 1, manifestPath ? "No usable entries detected in manifest ${manifestPath}" : "No usable FASTQ files detected in ${inputDir}"
}
log.info "Discovered ${sampleRecords.size()} input items from ${manifestPath ? "manifest ${manifestPath}" : inputDir}. Sample threads=${sampleThreads}, default threads=${pipelineThreads}"

Channel
    .from(sampleRecords)
    .map { rec ->
        def meta = [ sample_id: rec.sample_id, paired: rec.paired ]
        def r1File = file(rec.r1)
        def r2File = (rec.paired && rec.r2) ? file(rec.r2) : r1File
        tuple(meta, r1File, r2File)
    }
    .set { raw_reads }

def tabFilterScript = resolveOptionalPath(config.table_filter?.script, configRoot) ?: "${projectDir}/filter_ASV_table.py"
def tableScriptFile = file(tabFilterScript)
if( !tableScriptFile.exists() ) {
    exit 1, "Table filter script not found: ${tabFilterScript}"
}

def parseSinaScriptFile = new File("${projectDir}/parse_sina_log.py")
if( !parseSinaScriptFile.exists() ) {
    exit 1, "parse_sina_log.py not found in project directory"
}
def parseSinaScriptPath = parseSinaScriptFile.canonicalPath
def trimSinaScriptFile = new File("${projectDir}/trim_v_sina.py")
if( !trimSinaScriptFile.exists() ) {
    exit 1, "trim_v_sina.py not found in project directory"
}
def trimSinaScriptPath = trimSinaScriptFile.canonicalPath
def mitomasterScriptFile = new File("${projectDir}/mitomaster.py")
if( !mitomasterScriptFile.exists() ) {
    exit 1, "mitomaster.py not found in project directory"
}
def mitomasterScriptPath = mitomasterScriptFile.canonicalPath
def mitoCheckerScriptFile = new File("${projectDir}/mito_checker.py")
if( !mitoCheckerScriptFile.exists() ) {
    exit 1, "mito_checker.py not found in project directory"
}
def mitoCheckerScriptPath = mitoCheckerScriptFile.canonicalPath
def sankeyScriptFile = new File("${projectDir}/sankey_builder.py")
if( !sankeyScriptFile.exists() ) {
    exit 1, "sankey_builder.py not found in project directory"
}
def sankeyScriptPath = sankeyScriptFile.canonicalPath
def plotMetadataScriptFile = new File("${projectDir}/plot_metadata.py")
if( !plotMetadataScriptFile.exists() ) {
    exit 1, "plot_metadata.py not found in project directory"
}
def plotMetadataScriptPath = plotMetadataScriptFile.canonicalPath
def batchCorrectionScriptFile = new File("${projectDir}/asv_batch_correction.py")
if( !batchCorrectionScriptFile.exists() ) {
    exit 1, "asv_batch_correction.py not found in project directory"
}
def batchCorrectionScriptPath = batchCorrectionScriptFile.canonicalPath
def outlierCheckerScriptFile = new File("${projectDir}/outlier_checker.py")
if( !outlierCheckerScriptFile.exists() ) {
    exit 1, "outlier_checker.py not found in project directory"
}
def outlierCheckerScriptPath = outlierCheckerScriptFile.canonicalPath
def collectorsCurveScriptFile = new File("${projectDir}/collectors_curve.py")
if( !collectorsCurveScriptFile.exists() ) {
    exit 1, "collectors_curve.py not found in project directory"
}
def collectorsCurveScriptPath = collectorsCurveScriptFile.canonicalPath
def filterCountsScriptFile = new File("${projectDir}/filter_nontarget.py")
if( !filterCountsScriptFile.exists() ) {
    exit 1, "filter_nontarget.py not found in project directory"
}
def filterCountsScriptPath = filterCountsScriptFile.canonicalPath
def sinaConfig = config.sina ?: [:]
def sinaDownloadSubdir = (sinaConfig.download_subdir ?: 'sina_reference').toString()
def sinaDownloadDir = new File(outputDir, sinaDownloadSubdir)
sinaDownloadDir.mkdirs()
def defaultSinaReferenceFile = new File(sinaDownloadDir, sinaReferenceFilename)
def configuredSinaReference = sinaConfig.reference ? resolveOptionalPath(sinaConfig.reference, configRoot) : null
def initialSinaReferencePath = configuredSinaReference ?: defaultSinaReferenceFile.canonicalPath
File sinaReferenceFile = new File(initialSinaReferencePath)
if( !sinaReferenceFile.exists() ) {
    if( configuredSinaReference && sinaReferenceFile.canonicalPath != defaultSinaReferenceFile.canonicalPath ) {
        log.warn "Configured SINA reference not found at ${sinaReferenceFile.canonicalPath}; downloading to ${defaultSinaReferenceFile.canonicalPath}."
        sinaReferenceFile = defaultSinaReferenceFile
    }
    def referenceUrl = sinaConfig.reference_url ?: defaultSinaReferenceUrl
    log.info "Downloading SILVA reference (${sinaReferenceFilename}) from ${referenceUrl}"
    downloadReference(referenceUrl, sinaReferenceFile)
}
if( !sinaReferenceFile.exists() ) {
    exit 1, "Failed to obtain SINA reference file at ${sinaReferenceFile}"
}
def sinaReferencePath = sinaReferenceFile.canonicalPath
def sinaRegionsRaw = sinaConfig.regions
List<String> sinaRegionList
if( sinaRegionsRaw instanceof List ) {
    sinaRegionList = sinaRegionsRaw.collect { it.toString().trim() }.findAll { it }
} else if( sinaRegionsRaw ) {
    sinaRegionList = sinaRegionsRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
} else {
    sinaRegionList = ['V4']
}
if( !sinaRegionList ) {
    sinaRegionList = ['V4']
}
def sinaRegionsArg = sinaRegionList.join(',')
def sinaTrimTarget = sinaConfig.trim_to ?: sinaRegionList[0]
def sinaBatchSize = sinaConfig.batch_size ? (sinaConfig.batch_size as int) : 1000000
def sinaKeepGaps = (sinaConfig.keep_gaps ?: false) as boolean
def sinaVerbose = (sinaConfig.verbose ?: false) as boolean
def sinaIdColumn = sinaConfig.id_column ?: 'ASV_ID'
def sinaThreads = sinaConfig.threads ? (sinaConfig.threads as int) : pipelineThreads
def generalStatsConfig = config.general_stats ?: [:]
def generalStatsEnabled = generalStatsConfig.containsKey('enabled') ? (generalStatsConfig.enabled as boolean) : true
def rawInputFiles = generalStatsEnabled ? sampleRecords.collectMany { rec ->
    def list = [rec.r1]
    if( rec.paired && rec.r2 ) {
        list << rec.r2
    }
    return list
}.unique() : []
def fastpOutputFiles = generalStatsEnabled ? sampleRecords.collectMany { rec ->
    def outputs = [
        new File(dirMap.fastp, "${rec.sample_id}_R1.fastq.gz").absolutePath
    ]
    outputs << new File(dirMap.fastp, "${rec.sample_id}_R2.fastq.gz").absolutePath
    return outputs
} : []
def filteredOutputFiles = generalStatsEnabled ? sampleRecords.collect { rec ->
    new File(dirMap.filter, "${rec.sample_id}.filtered.fasta").absolutePath
} : []
def generalStatsRawArgs = joinShellArgs(rawInputFiles)
def generalStatsFastpArgs = joinShellArgs(fastpOutputFiles)
def generalStatsFilteredArgs = joinShellArgs(filteredOutputFiles)
def taxonomyScriptFile = new File("${projectDir}/qiime_vs_classifier.py")
if( !taxonomyScriptFile.exists() ) {
    exit 1, "qiime_vs_classifier.py not found in project directory"
}
def taxonomyScriptPath = taxonomyScriptFile.canonicalPath
def taxonomyConfig = config.taxonomy ?: [:]
def taxonomyDownloadSubdir = (taxonomyConfig.download_subdir ?: 'taxonomy_reference').toString()
def taxonomyDownloadDir = new File(outputDir, taxonomyDownloadSubdir)
taxonomyDownloadDir.mkdirs()
def taxonomyTaxFilename = taxonomyConfig.ref_taxonomy_filename ?: defaultTaxonomyTaxFilename
def taxonomySeqFilename = taxonomyConfig.ref_sequences_filename ?: defaultTaxonomySeqsFilename

def taxonomyRefTaxonomyResolved = taxonomyConfig.ref_taxonomy ? resolveOptionalPath(taxonomyConfig.ref_taxonomy, configRoot) : null
def taxonomyRefTaxonomyFile = taxonomyRefTaxonomyResolved ? new File(taxonomyRefTaxonomyResolved) : new File(taxonomyDownloadDir, taxonomyTaxFilename)
if( !taxonomyRefTaxonomyFile.exists() ) {
    def taxonomyUrl = taxonomyConfig.ref_taxonomy_url ?: defaultTaxonomyTaxUrl
    if( !taxonomyUrl ) {
        exit 1, "taxonomy.ref_taxonomy or taxonomy.ref_taxonomy_url must be provided to obtain SILVA taxonomy artifact"
    }
    log.info "Downloading SILVA taxonomy artifact from ${taxonomyUrl}"
    downloadReference(taxonomyUrl, taxonomyRefTaxonomyFile)
}
if( !taxonomyRefTaxonomyFile.exists() ) {
    exit 1, "Taxonomy reference taxonomy file not found: ${taxonomyRefTaxonomyFile}"
}

def taxonomyRefSequencesResolved = taxonomyConfig.ref_sequences ? resolveOptionalPath(taxonomyConfig.ref_sequences, configRoot) : null
def taxonomyRefSequencesFile = taxonomyRefSequencesResolved ? new File(taxonomyRefSequencesResolved) : new File(taxonomyDownloadDir, taxonomySeqFilename)
if( !taxonomyRefSequencesFile.exists() ) {
    def taxonomySeqsUrl = taxonomyConfig.ref_sequences_url ?: defaultTaxonomySeqsUrl
    if( !taxonomySeqsUrl ) {
        exit 1, "taxonomy.ref_sequences or taxonomy.ref_sequences_url must be provided to obtain SILVA sequences artifact"
    }
    log.info "Downloading SILVA sequences artifact from ${taxonomySeqsUrl}"
    downloadReference(taxonomySeqsUrl, taxonomyRefSequencesFile)
}
if( !taxonomyRefSequencesFile.exists() ) {
    exit 1, "Taxonomy reference sequences file not found: ${taxonomyRefSequencesFile}"
}

def taxonomyRefTaxonomy = taxonomyRefTaxonomyFile.canonicalPath
def taxonomyRefSequences = taxonomyRefSequencesFile.canonicalPath
def taxonomyOutputName = taxonomyConfig.output_tsv ?: 'ASV_SILVA_tax.full-length.vsearch.tsv'
def taxonomyStatsName = taxonomyConfig.stats_tsv ?: 'ASV_SILVA_stats.full-length.vsearch.tsv'
def taxonomyUppercaseName = taxonomyConfig.uppercase_fasta ?: 'ASVs.upper.fasta'
def taxonomyThreads = taxonomyConfig.threads ? (taxonomyConfig.threads as int) : pipelineThreads
def mitoConfig = config.mito ?: [:]
def mitoEnabled = mitoConfig.containsKey('enabled') ? (mitoConfig.enabled as boolean) : true
def defaultMitoChunkDir = new File(dirMap.asv, "chunks").canonicalPath
def mitoChunkDirPath = mitoConfig.chunk_dir ? resolveOutputRelative(mitoConfig.chunk_dir.toString(), outputDir) : defaultMitoChunkDir
def defaultMitoOutputDir = new File(dirMap.mito, "mitomap").canonicalPath
def mitoOutputDirPath = mitoConfig.output_dir ? resolveOutputRelative(mitoConfig.output_dir.toString(), outputDir) : defaultMitoOutputDir
def mitoBlastDbPath = resolveOptionalPath(mitoConfig.mito_db ?: 'ref_db/mito_ncbi', configRoot)
def mitoBiofDbPath = resolveOptionalPath(mitoConfig.biof_db ?: 'ref_db/ssu_pipeline_contaminants', configRoot)
def mitoChunkSize = mitoConfig.chunk_size ? (mitoConfig.chunk_size as int) : 10
def mitomasterWorkers = mitoConfig.mitomaster_workers ? (mitoConfig.mitomaster_workers as int) : 8
def mitomasterRetries = mitoConfig.mitomaster_retries ? (mitoConfig.mitomaster_retries as int) : 4
def mitomasterTimeout = mitoConfig.mitomaster_timeout ? (mitoConfig.mitomaster_timeout as int) : 90
def mitomasterHeaderMode = mitoConfig.mitomaster_header_mode ?: 'first'
def mitoBlastThreads = mitoConfig.blast_threads ? (mitoConfig.blast_threads as int) : pipelineThreads
def mitoPrefix = mitoConfig.prefix ?: 'nontarget'
def mitoFormats = mitoConfig.formats ?: 'svg,pdf'
def mitoMinPident = mitoConfig.min_pident != null ? (mitoConfig.min_pident as double) : 97.0
def mitoMinPercov = mitoConfig.min_percov != null ? (mitoConfig.min_percov as double) : 51.0
def mitoMitoSubstring = mitoConfig.mitochondria_substring ?: 'mitochondria'
def mitoFeatureCol = mitoConfig.feature_col ?: 'Feature ID'
def mitoTaxonCol = mitoConfig.taxon_col ?: 'Taxon'
def mitoConsensusCol = mitoConfig.consensus_col ?: 'Consensus'
def mitoSteps = mitoConfig.steps ?: 'BioFactorial,Qiime_NB_FULL,MITOMASTER,BLAST_mito'
def mitoHostFirstStep = mitoConfig.host_first_step ?: 'BioFactorial'
def mitoFigsize = mitoConfig.figsize ?: '10x6'
def mitoStyle = mitoConfig.style ?: 'whitegrid'
def mitoDpi = mitoConfig.dpi ? (mitoConfig.dpi as int) : 300
def mitoNoPlots = (mitoConfig.no_plots ?: false) as boolean
def filterCountsConfig = config.filter_counts ?: [:]
def filterCountsEnabled = filterCountsConfig.containsKey('enabled') ? (filterCountsConfig.enabled as boolean) : true
if( filterCountsEnabled && !mitoEnabled ) {
    exit 1, "filter_counts.enabled requires mito.enabled to be true"
}
def filterCountsMetadataPath = filterCountsConfig.metadata ? resolveOptionalPath(filterCountsConfig.metadata, configRoot) : null
if( filterCountsEnabled && filterCountsMetadataPath && !new File(filterCountsMetadataPath).exists() ) {
    exit 1, "filter_counts.metadata not found: ${filterCountsMetadataPath}"
}
def filterCountsOutputName = filterCountsConfig.output ?: 'ASV_target.tsv'
def filterCountsGroupCol = filterCountsConfig.group_col ?: 'Depth'
def filterCountsMinGroup = filterCountsConfig.min_group_size ? (filterCountsConfig.min_group_size as int) : 3
def filterCountsAbundance = filterCountsConfig.abundance_threshold != null ? (filterCountsConfig.abundance_threshold as double) : 0.005d
def filterCountsSampleCol = filterCountsConfig.sample_id_col ?: 'longID'
def filterCountsMinConsensus = filterCountsConfig.min_consensus != null ? (filterCountsConfig.min_consensus as double) : 0d
def filterCountsTaxonCol = filterCountsConfig.taxon_col ?: 'Taxon'
def filterCountsConsensusCol = filterCountsConfig.consensus_col ?: 'Consensus'
def filterCountsBiofactorialCol = filterCountsConfig.biofactorial_col ?: 'BioFactorial'
def filterCountsMitoColsRaw = filterCountsConfig.mito_cols
List<String> filterCountsMitoCols
if( filterCountsMitoColsRaw instanceof List ) {
    filterCountsMitoCols = filterCountsMitoColsRaw.collect { it.toString() }
} else if( filterCountsMitoColsRaw ) {
    filterCountsMitoCols = filterCountsMitoColsRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
} else {
    filterCountsMitoCols = ['MITOMASTER','BLAST_mito']
}
def filterCountsSaveIntermediates = (filterCountsConfig.save_intermediates ?: false) as boolean
def defaultFilterMitoDir = new File(dirMap.mito, "ASVs").canonicalPath
def filterCountsMitoDir = filterCountsConfig.mito_output_dir ? resolveOutputRelative(filterCountsConfig.mito_output_dir.toString(), outputDir) : defaultFilterMitoDir
if( mitoEnabled ) {
    ensureBlastDbExists(mitoBlastDbPath, 'mitochondrial')
    ensureBlastDbExists(mitoBiofDbPath, 'BioFactorial')
    new File(mitoChunkDirPath).parentFile?.mkdirs()
    new File(mitoOutputDirPath).mkdirs()
}
def sankeyConfig = config.sankey ?: [:]
def sankeyEnabled = (sankeyConfig.enabled ?: false) as boolean
def sankeyMetadataPath = sankeyConfig.metadata ? resolveOptionalPath(sankeyConfig.metadata, configRoot) : resolveOptionalPath("ref_db/asv_cruise_metadata.tsv", configRoot)
if( sankeyEnabled && (!sankeyMetadataPath || !new File(sankeyMetadataPath).exists()) ) {
    exit 1, "Sankey metadata file not found: ${sankeyMetadataPath}"
}
def sankeySubDir = sankeyConfig.sub_dir ?: '.'
def sankeySampCol = sankeyConfig.sample_col ?: 'lmp_id'
def sankeyGroupCol = sankeyConfig.group1_col ?: 'group1'
def sankeyColorCol = sankeyConfig.color_col ?: 'Color'
def sankeyKeepTypesRaw = sankeyConfig.keep_types
List<String> sankeyKeepTypes = []
if( sankeyKeepTypesRaw instanceof List ) {
    sankeyKeepTypes = sankeyKeepTypesRaw.collect { it.toString().trim() }.findAll { it }
} else if( sankeyKeepTypesRaw ) {
    sankeyKeepTypes = sankeyKeepTypesRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def sankeyOutputPrefix = sankeyConfig.output_prefix ?: "metadata/data_loss_sankey"
def sankeyTitle = sankeyConfig.title ?: "Data Loss Flow"
def sankeyMakeLabeled = (sankeyConfig.make_labeled == null) ? true : (sankeyConfig.make_labeled as boolean)
def sankeyMakeUnlabeled = (sankeyConfig.make_unlabeled == null) ? true : (sankeyConfig.make_unlabeled as boolean)
if( sankeyEnabled && filterCountsEnabled && !filterCountsSaveIntermediates ) {
    exit 1, "Sankey requires filter_counts.save_intermediates to be true to access intermediate tables"
}
if( sankeyEnabled && !filterCountsEnabled ) {
    exit 1, "Sankey requires filter_counts.enabled to be true"
}
if( sankeyEnabled && !generalStatsEnabled ) {
    exit 1, "Sankey requires general_stats.enabled to be true"
}
def metadataPlotsConfig = config.metadata_plots ?: [:]
boolean metadataPlotsEnabled = metadataPlotsConfig.containsKey('enabled') ? (metadataPlotsConfig.enabled as boolean) : true
if( metadataPlotsEnabled && !filterCountsEnabled ) {
    exit 1, "metadata_plots.enabled requires filter_counts.enabled to be true"
}
if( metadataPlotsEnabled && !generalStatsEnabled ) {
    exit 1, "metadata_plots.enabled requires general_stats.enabled to be true"
}
def metadataPlotsMetadataPath = metadataPlotsConfig.metadata ? resolveOptionalPath(metadataPlotsConfig.metadata, configRoot) : resolveOptionalPath("ref_db/asv_cruise_metadata.tsv", configRoot)
if( metadataPlotsEnabled && (!metadataPlotsMetadataPath || !new File(metadataPlotsMetadataPath).exists()) ) {
    exit 1, "metadata_plots metadata file not found: ${metadataPlotsMetadataPath}"
}
def metadataPlotsSubDir = metadataPlotsConfig.sub_dir ?: '.'
def metadataPlotsTypeCol = metadataPlotsConfig.type_col ?: 'Depth'
def metadataPlotsColorCol = metadataPlotsConfig.color_col ?: 'Color'
def metadataKeepTypesRaw = metadataPlotsConfig.keep_types
List<String> metadataKeepTypes = []
if( metadataKeepTypesRaw instanceof List ) {
    metadataKeepTypes = metadataKeepTypesRaw.collect { it.toString().trim() }.findAll { it }
} else if( metadataKeepTypesRaw ) {
    metadataKeepTypes = metadataKeepTypesRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def metadataIncludeRankRaw = metadataPlotsConfig.include_rank
List<String> metadataIncludeRank = []
if( metadataIncludeRankRaw instanceof List ) {
    metadataIncludeRank = metadataIncludeRankRaw.collect { it.toString().trim() }.findAll { it }
} else if( metadataIncludeRankRaw ) {
    metadataIncludeRank = metadataIncludeRankRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def metadataPlotsMitoThreshold = metadataPlotsConfig.mito_threshold_line != null ? (metadataPlotsConfig.mito_threshold_line as double) : 1000d
boolean metadataPlotsRunMicro = metadataPlotsConfig.containsKey('run_micro') ? (metadataPlotsConfig.run_micro as boolean) : true
boolean metadataPlotsRunMito = metadataPlotsConfig.containsKey('run_mito') ? (metadataPlotsConfig.run_mito as boolean) : true
if( metadataPlotsEnabled && !metadataPlotsRunMicro && !metadataPlotsRunMito ) {
    exit 1, "metadata_plots configured to skip both micro and mito outputs; disable metadata_plots.enabled instead."
}
boolean metadataForceMicroOnly = metadataPlotsRunMicro && !metadataPlotsRunMito
boolean metadataForceMitoOnly = metadataPlotsRunMito && !metadataPlotsRunMicro

def batchCorrectionConfig = config.batch_correction ?: [:]
boolean batchCorrectionEnabled = metadataPlotsEnabled && (batchCorrectionConfig.containsKey('enabled') ? (batchCorrectionConfig.enabled as boolean) : true)
def batchCorrectionOutputDir = batchCorrectionConfig.output_dir ?: 'batch_correction'
def batchCorrectionOutputDirAbs = new File(outputDir, batchCorrectionOutputDir).canonicalPath
def batchCorrectionBatchCol = batchCorrectionConfig.batch_col ?: 'batch'
def batchCorrectionOrientation = batchCorrectionConfig.asv_orientation ?: 'features_rows'
def batchBiologicalCovariates = batchCorrectionConfig.biological_covariates ? batchCorrectionConfig.biological_covariates.toString().trim() : ''
def batchBiologicalColorCols = batchCorrectionConfig.biological_color_col ?: 'Depth'
def batchColorPaletteCols = batchCorrectionConfig.color_palette_col ?: 'Color'
def batchUmapNeighbors = batchCorrectionConfig.umap_neighbors ? (batchCorrectionConfig.umap_neighbors as int) : 15
def batchUmapMinDist = batchCorrectionConfig.umap_min_dist != null ? (batchCorrectionConfig.umap_min_dist as double) : 0.1d
def batchHdbscanMinClusterSize = batchCorrectionConfig.hdbscan_min_cluster_size ? (batchCorrectionConfig.hdbscan_min_cluster_size as int) : 5
def batchHdbscanMinSamples = batchCorrectionConfig.hdbscan_min_samples != null ? (batchCorrectionConfig.hdbscan_min_samples as int) : null
def batchHdbscanSelectionMethod = batchCorrectionConfig.hdbscan_selection_method ?: 'eom'
boolean batchOptimize = (batchCorrectionConfig.optimize_clustering ?: false) as boolean
def batchTargetClusters = batchCorrectionConfig.target_clusters ?: '3-8'
def batchNFeaturesPlot = batchCorrectionConfig.n_features_plot ? (batchCorrectionConfig.n_features_plot as int) : 5
def batchRandomState = batchCorrectionConfig.random_state ? (batchCorrectionConfig.random_state as int) : 42

def outlierConfig = config.outlier_detection ?: [:]
boolean outlierEnabled = batchCorrectionEnabled && (outlierConfig.containsKey('enabled') ? (outlierConfig.enabled as boolean) : true)
def outlierOutputDir = outlierConfig.output_dir ?: 'outliers_corrected'
def outlierOutputDirAbs = new File(outputDir, outlierOutputDir).canonicalPath
def outlierGroupColsRaw = outlierConfig.group_cols
List<String> outlierGroupCols = []
if( outlierGroupColsRaw instanceof List ) {
    outlierGroupCols = outlierGroupColsRaw.collect { it.toString().trim() }.findAll { it }
} else if( outlierGroupColsRaw ) {
    outlierGroupCols = outlierGroupColsRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
if( outlierGroupCols.isEmpty() ) {
    outlierGroupCols = ['none']
}
def outlierTransform = outlierConfig.transform ?: 'clr'
def outlierOrientation = outlierConfig.asv_orientation ?: 'features_rows'
boolean outlierPreTransformed = (outlierConfig.pre_transformed ?: false) as boolean
boolean outlierScale = (outlierConfig.scale ?: false) as boolean
boolean outlierUseIso = (outlierConfig.use_iso ?: false) as boolean
boolean outlierUseSvm = (outlierConfig.use_svm ?: false) as boolean
boolean outlierUseHdb = (outlierConfig.use_hdb ?: false) as boolean
def outlierVoteThreshold = outlierConfig.vote_threshold ? (outlierConfig.vote_threshold as int) : 3
def outlierIsoContamination = outlierConfig.iso_contamination ?: 'auto'
def outlierIsoEstimators = outlierConfig.iso_estimators ? (outlierConfig.iso_estimators as int) : 100
def outlierIsoRandomState = outlierConfig.iso_random_state ? (outlierConfig.iso_random_state as int) : 42
def outlierSvmKernel = outlierConfig.svm_kernel ?: 'rbf'
def outlierSvmGamma = outlierConfig.svm_gamma ?: 'scale'
def outlierSvmNu = outlierConfig.svm_nu ? (outlierConfig.svm_nu as double) : 0.1d
def outlierHdbMinClusterSize = outlierConfig.hdbscan_min_cluster_size ? (outlierConfig.hdbscan_min_cluster_size as int) : 5
def outlierHdbMinSamples = outlierConfig.hdbscan_min_samples != null ? (outlierConfig.hdbscan_min_samples as int) : null
def outlierHdbMetric = outlierConfig.hdbscan_metric ?: 'euclidean'

def collectorsConfig = config.collectors_curve ?: [:]
boolean collectorsEnabled = metadataPlotsEnabled && (collectorsConfig.containsKey('enabled') ? (collectorsConfig.enabled as boolean) : true)
def collectorsGroupCol = collectorsConfig.group_col ?: 'Depth'
def collectorsColorCol = collectorsConfig.color_col ?: 'Color'
def collectorsPermutations = collectorsConfig.permutations ? (collectorsConfig.permutations as int) : 999
def collectorsSeed = collectorsConfig.seed ? (collectorsConfig.seed as int) : 42
def collectorsOutPrefix = collectorsConfig.out_prefix ?: 'metadata/collectors_curve'
def collectorsOutPrefixAbs = new File(outputDir, collectorsOutPrefix).canonicalPath
def collectorsTitle = collectorsConfig.title ?: ''
def collectorsFormats = collectorsConfig.formats ?: 'pdf'
def collectorsXpad = collectorsConfig.xpad != null ? (collectorsConfig.xpad as double) : 0.5d
def collectorsMaxCols = collectorsConfig.max_cols ? (collectorsConfig.max_cols as int) : 3
def collectorsShowPerms = collectorsConfig.show_perms ? (collectorsConfig.show_perms as int) : 10
def collectorsPresenceThreshold = collectorsConfig.presence_threshold != null ? (collectorsConfig.presence_threshold as double) : 0d

workflow {
    def fastp_result = FASTP_QC(raw_reads)
    def reads_after_qc = fastp_result.reads
    def reads_after_merge = MERGE_READS(reads_after_qc)
    def reads_after_filter = FILTER_READS(reads_after_merge)
    def relabeled_fasta_files = reads_after_filter.map { parts -> parts[1] }

    def concat_for_derep = relabeled_fasta_files
        .collectFile(name: 'concat.fasta', storeDir: dirMap.concat, newLine: true)
    def concat_for_counts = relabeled_fasta_files
        .collectFile(name: 'concat_counts.fasta', storeDir: dirMap.concat, newLine: true)

    def derep_input = DEREPLICATE(concat_for_derep)
    def sina_stage = SINA_TRIM(derep_input)
    def denoise_input = DENOISE(sina_stage.trimmed_fasta)
    def nochi_input = CHIMERA_CHECK(denoise_input)

    def count_matrix_stage = CREATE_COUNT_MATRIX(concat_for_counts, nochi_input)
    def count_matrix_channel = count_matrix_stage.count_matrix
    def asv_counts_for_sankey = count_matrix_channel.map { tuple -> tuple[0] }
    def filtered_stage = FILTER_TABLE(count_matrix_channel)
    def filtered_channel = filtered_stage.filtered
    def taxonomy_stage = TAXONOMY(filtered_channel)
    def runMitoStages = mitoEnabled || filterCountsEnabled
    def filter_counts_stage = null
    if( runMitoStages ) {
        def mitomaster_stage = MITOMASTER(filtered_channel)
        def mito_summary = MITO_DECONTAM(mitomaster_stage.mito_artifacts, taxonomy_stage.taxonomy_table)
        if( filterCountsEnabled ) {
            filter_counts_stage = FILTER_COUNTS(filtered_channel, taxonomy_stage.taxonomy_table, mito_summary.nontarget_table)
        }
    }
    if( metadataPlotsEnabled && !filter_counts_stage ) {
        exit 1, "metadata_plots.enabled requires filter_counts outputs but filter_counts stage was not executed"
    }
    def general_stats_stage = null
    if( generalStatsEnabled ) {
        general_stats_stage = GENERAL_STATS(concat_for_counts)
    }

def metadata_stage = null
def metaMicroForBatch = null
def metaMicroForOutlier = null
def metaMicroForCollectors = null
def asvMetaForBatch = null
def asvFinalForBatch = null
def asvFinalForCollectors = null
    if( metadataPlotsEnabled ) {
        metadata_stage = PLOT_METADATA(
            general_stats_stage.fastq_stats,
            filter_counts_stage.filtered_micro,
            filter_counts_stage.filtered_mito,
            taxonomy_stage.taxonomy_table
        )
        metaMicroForBatch = metadata_stage.metadata_micro
        metaMicroForOutlier = metadata_stage.metadata_micro
        metaMicroForCollectors = metadata_stage.metadata_micro
        asvMetaForBatch = metadata_stage.asv_meta_micro
        asvFinalForBatch = metadata_stage.asv_final_micro
        asvFinalForCollectors = metadata_stage.asv_final_micro
    }

    def batch_stage = null
    def asvClrForOutlier = null
    if( batchCorrectionEnabled ) {
        batch_stage = ASV_BATCH_CORRECTION(
            metaMicroForBatch,
            asvMetaForBatch,
            asvFinalForBatch
        )
        asvClrForOutlier = batch_stage.asv_clr_after
        umapResultsForTrajectory = batch_stage.umap_results
    }

    if( outlierEnabled ) {
        OUTLIER_CHECKER(
            asvClrForOutlier,
            metaMicroForOutlier
        )
    }
    if( collectorsEnabled ) {
        COLLECTORS_CURVE(
            asvFinalForCollectors,
            metaMicroForCollectors
        )
    }
    if( sankeyEnabled ) {
        SANKEY(
            general_stats_stage.fastq_stats,
            general_stats_stage.filtered_stats,
            asv_counts_for_sankey,
            filter_counts_stage.filtered_decon,
            filter_counts_stage.filtered_micro
        )
    }
}

process FASTP_QC {
    tag { meta.sample_id }
    cpus sampleThreads
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
    tuple val(meta), path(r1), path(r2)

    output:
    tuple val(meta), path("R1.fastq.gz"), path("R2.fastq.gz"), emit: reads
    path("fastp.json"), emit: fastp_json
    path("fastp.html"), emit: fastp_html

    script:
    def trimFrontR1 = fastpTrimValues.front_r1
    def trimTailR1  = fastpTrimValues.tail_r1
    def trimFrontR2 = fastpTrimValues.front_r2
    def trimTailR2  = fastpTrimValues.tail_r2
    if( meta.paired && r2 ) {
        return """
fastp \\
  -i "${r1}" -I "${r2}" \\
  -o R1.fastq.gz \\
  -O R2.fastq.gz \\
  -f ${trimFrontR1} -t ${trimTailR1} \\
  -F ${trimFrontR2} -T ${trimTailR2} \\
  -j fastp.json \\
  -h fastp.html \\
  -w ${task.cpus}
"""
    }
    return """
fastp \\
  -i "${r1}" \\
  -o R1.fastq.gz \\
  -f ${trimFrontR1} -t ${trimTailR1} \\
  -j fastp.json \\
  -h fastp.html \\
  -w ${task.cpus}
\nln -sf R1.fastq.gz R2.fastq.gz\n
"""
}

process MERGE_READS {
    tag { meta.sample_id }
    cpus sampleThreads
    conda "${condaEnvPath}"
    publishDir dirMap.merge, mode: 'copy', saveAs: { filename ->
        filename == 'merged.fastq' ? "${meta.sample_id}.merged.fastq" : filename
    }

    input:
    tuple val(meta), path(r1), path(r2)

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
    cpus sampleThreads
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

process SINA_TRIM {
    cpus sinaThreads
    conda "${sinaCondaEnvPath}"
    publishDir dirMap.sina, mode: 'copy', pattern: '*'

    input:
    path(derep_fasta)

    output:
    path("derep_trimmed.fasta"), emit: trimmed_fasta
    path("derep_SINA.fasta")
    path("derep_SINA.log")
    path("derep_v_regions.tsv")

    script:
    def parseVerboseArg = sinaVerbose ? ' --verbose' : ''
    def trimTargetArg = sinaTrimTarget ? " -t \"${sinaTrimTarget}\"" : ''
    def keepGapsArg = sinaKeepGaps ? ' --keep-gaps' : ''
    """
set -euo pipefail
sina \\
    -i "${derep_fasta}" \\
    -o derep_SINA.fasta \\
    -r "${sinaReferencePath}" \\
    -v \\
    -p ${task.cpus} \\
    --log-file derep_SINA.log

python "${parseSinaScriptPath}" \\
  --log derep_SINA.log \\
  --output derep_v_regions.tsv${parseVerboseArg}

python "${trimSinaScriptPath}" \\
  -m derep_v_regions.tsv \\
  -f derep_SINA.fasta \\
  -r "${sinaRegionsArg}"${trimTargetArg} \\
  -o derep_trimmed.fasta \\
  --id-column "${sinaIdColumn}" \\
  --threads ${task.cpus} \\
  --batch-size ${sinaBatchSize}${keepGapsArg}
"""
}
process DENOISE {
    cpus pipelineThreads
    conda "${condaEnvPath}"
    publishDir dirMap.denoise, mode: 'copy', pattern: '*'

    input:
    path(trimmed_fasta)

    output:
    path("centroids.fasta")

    script:
    def unoiseCfg = config.unoise ?: [:]
    """
vsearch --cluster_unoise "${trimmed_fasta}" \\
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

process CREATE_COUNT_MATRIX {
    cpus pipelineThreads
    conda "${condaEnvPath}"
    publishDir dirMap.asv, mode: 'copy', pattern: '*'

    input:
    path(concat_fasta)
    path(nochimeras)

    output:
    tuple path("ASV_counts.tsv"), path("ASVs.fasta"), emit: count_matrix

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
    tuple path("ASV_filtered.tsv"), path("ASVs_filtered.fasta"), emit: filtered

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

process TAXONOMY {
    cpus taxonomyThreads
    conda "${taxonomyCondaEnvPath}"
    publishDir dirMap.taxonomy, mode: 'copy', pattern: '*'

    input:
    tuple path(filtered_table), path(filtered_fasta)

    output:
    path("${taxonomyUppercaseName}")
    path("${taxonomyOutputName}"), emit: taxonomy_table
    path("${taxonomyStatsName}")

    script:
    """
set -euo pipefail
python - <<'PY' '${filtered_fasta}' '${taxonomyUppercaseName}'
import sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
with src.open() as inp, dst.open('w') as out:
    for line in inp:
        if line.startswith('>'):
            out.write(line)
        else:
            out.write(line.strip().upper() + '\\n')
PY

python "${taxonomyScriptPath}" \\
  --input-fasta "${taxonomyUppercaseName}" \\
  --ref-taxonomy "${taxonomyRefTaxonomy}" \\
  --ref-seqs "${taxonomyRefSequences}" \\
  --output-tsv "${taxonomyOutputName}" \\
  --stats-output "${taxonomyStatsName}" \\
  --threads ${task.cpus}
"""
}

process MITOMASTER {
    cpus mitoBlastThreads
    conda "${mitomasterCondaEnvPath}"
    publishDir mitoOutputDirPath, mode: 'copy', pattern: '*'

    input:
    tuple path(filtered_table), path(filtered_fasta)

    output:
    tuple path("mitomaster_output.tsv"), path("mito_ncbi.blast6.tsv"), path("ssu_pipeline_contaminants.blast6.tsv"), emit: mito_artifacts

    script:
    """
set -euo pipefail
rm -rf "${mitoChunkDirPath}"
mkdir -p "${mitoChunkDirPath}"

seqkit split -s ${mitoChunkSize} -O "${mitoChunkDirPath}" "${filtered_fasta}"

python "${mitomasterScriptPath}" \\
  --data-dir "${mitoChunkDirPath}" \\
  --glob-pattern "*.fasta" \\
  --output-file mitomaster_output.tsv \\
  --max-workers ${mitomasterWorkers} \\
  --timeout ${mitomasterTimeout} \\
  --retries ${mitomasterRetries} \\
  --header-mode "${mitomasterHeaderMode}" \\
  --overwrite

blastn -query "${filtered_fasta}" \\
  -db "${mitoBlastDbPath}" \\
  -outfmt "6 qseqid sseqid pident length qlen mismatch gapopen qstart qend sstart send evalue bitscore" \\
  -out mito_ncbi.blast6.tsv \\
  -num_threads ${task.cpus}

blastn -query "${filtered_fasta}" \\
  -db "${mitoBiofDbPath}" \\
  -outfmt "6 qseqid sseqid pident length qlen mismatch gapopen qstart qend sstart send evalue bitscore" \\
  -out ssu_pipeline_contaminants.blast6.tsv \\
  -num_threads ${task.cpus}
"""
}

process MITO_DECONTAM {
    cpus mitoBlastThreads
    conda "${mitoCheckerCondaEnvPath}"
    publishDir mitoOutputDirPath, mode: 'copy', pattern: '*'

    input:
    tuple path(mitomaster_file), path(mito_blast), path(biof_blast)
    path(taxonomy_table)

    output:
    path("${mitoPrefix}.master.tsv"), emit: nontarget_table

    script:
    def noPlotsFlag = mitoNoPlots ? ' --no-plots' : ''
    """
python "${mitoCheckerScriptPath}" \\
  --mitomaster-file "${mitomaster_file}" \\
  --mito-blast "${mito_blast}" \\
  --silva-tax "${taxonomy_table}" \\
  --biof-file "${biof_blast}" \\
  --output-dir "." \\
  --prefix "${mitoPrefix}" \\
  --formats "${mitoFormats}" \\
  --min-pident ${mitoMinPident} \\
  --min-percov ${mitoMinPercov} \\
  --mitochondria-substring "${mitoMitoSubstring}" \\
  --feature-col "${mitoFeatureCol}" \\
  --taxon-col "${mitoTaxonCol}" \\
  --consensus-col "${mitoConsensusCol}" \\
  --steps "${mitoSteps}" \\
  --host-first-step "${mitoHostFirstStep}" \\
  --figsize "${mitoFigsize}" \\
  --style "${mitoStyle}" \\
  --dpi ${mitoDpi} \\
  --overwrite${noPlotsFlag}
"""
}

process FILTER_COUNTS {
    cpus pipelineThreads
    conda "${filterCountsCondaEnvPath}"
    publishDir dirMap.asv, mode: 'copy', pattern: '*', saveAs: { filename ->
        filename.endsWith('.mito.tsv') ? null : filename
    }
    publishDir filterCountsMitoDir, mode: 'copy', pattern: '*.mito.tsv'

    when:
    filterCountsEnabled

    input:
    tuple path(count_table), path(asv_fasta)
    path(taxonomy_table)
    path(nontarget_table)

    output:
    path("${filterCountsOutputName}"), emit: filtered_counts
    path("${filterCountsOutputName}".replace('.tsv','.micro.tsv')), optional: true, emit: filtered_micro
    path("${filterCountsOutputName}".replace('.tsv','.mito.tsv')), emit: filtered_mito
    path("${filterCountsOutputName}".replace('.tsv','.decon.tsv')), optional: true, emit: filtered_decon

    script:
    def metadataArg = filterCountsMetadataPath ? """  --metadata "${filterCountsMetadataPath}" \\\n""" : ''
    def groupArg = filterCountsGroupCol ? """  --group-col "${filterCountsGroupCol}" \\\n""" : ''
    def saveInterArg = filterCountsSaveIntermediates ? "  --save-intermediates \\\n" : ''
    def mitoColsArg = (filterCountsMitoCols && !filterCountsMitoCols.isEmpty()) ?
        """  --mito-cols ${filterCountsMitoCols.collect { "\"${it}\"" }.join(' ')} \\\n""" : ''
    """
set -euo pipefail
python "${filterCountsScriptPath}" \\
  --count-table "${count_table}" \\
  --nontarget-table "${nontarget_table}" \\
  --taxonomy-table "${taxonomy_table}" \\
${metadataArg}${groupArg}  --min-group-size ${filterCountsMinGroup} \\
  --abundance-threshold ${filterCountsAbundance} \\
  --sample-id-col "${filterCountsSampleCol}" \\
  --min-consensus ${filterCountsMinConsensus} \\
  --taxon-col "${filterCountsTaxonCol}" \\
  --consensus-col "${filterCountsConsensusCol}" \\
  --biofactorial-col "${filterCountsBiofactorialCol}" \\
${mitoColsArg}  --mito-output-dir "." \\
  --output "${filterCountsOutputName}" \\
${saveInterArg}
"""
}

process SANKEY {
    cpus 1
    conda "${sankeyCondaEnvPath}"

    when:
    sankeyEnabled

    input:
    path(fastq_stats)
    path(filtered_stats)
    path(asv_counts)
    path(asv_decon_counts)
    path(asv_micro_counts)

    script:
    def keepTypesArg = sankeyKeepTypes && !sankeyKeepTypes.isEmpty() ? "  --keep-types \"${sankeyKeepTypes.join(',')}\" \\\n" : ''
    def labeledFlag = sankeyMakeLabeled ? "  --make-labeled \\\n" : ''
    def unlabeledFlag = sankeyMakeUnlabeled ? "  --make-unlabeled \\\n" : ''
    """
python3 "${sankeyScriptPath}" \\
  --data-dir "${outputDir}" \\
  --sub-dir "${sankeySubDir}" \\
  --metadata "${sankeyMetadataPath}" \\
  --samp-col "${sankeySampCol}" \\
  --group1-col "${sankeyGroupCol}" \\
  --color-col "${sankeyColorCol}" \\
${keepTypesArg}  --fastq-stats "${fastq_stats}" \\
  --filtered-stats "${filtered_stats}" \\
  --asv-raw "${asv_counts}" \\
  --asv-decon "${asv_decon_counts}" \\
  --asv-micro "${asv_micro_counts}" \\
  --title "${sankeyTitle}" \\
  --output-prefix "${sankeyOutputPrefix}" \\
${labeledFlag}${unlabeledFlag}  --verbose
"""
}

process GENERAL_STATS {
    cpus pipelineThreads
    conda "${generalStatsCondaEnvPath}"
    publishDir dirMap.stats, mode: 'copy', pattern: '*'

    input:
    path(concat_fasta)

    output:
    path("fastq_stats.tsv"), emit: fastq_stats
    path("fastp_fastqs.tsv"), emit: fastp_stats
    path("filtered_fastas.tsv"), emit: filtered_stats
    path("concat_fastas.tsv"), emit: concat_stats

    script:
    def rawArgs = generalStatsRawArgs
    def fastpArgs = generalStatsFastpArgs
    def filteredArgs = generalStatsFilteredArgs
    """
set -euo pipefail

run_seqkit() {
  local outfile="\$1"
  shift
  if [[ "\$#" -eq 0 ]]; then
    : > "\${outfile}"
    return
  fi
  seqkit stat -a -T -j ${task.cpus} -o "\${outfile}" "\$@"
}

run_seqkit fastq_stats.tsv ${rawArgs}
run_seqkit fastp_fastqs.tsv ${fastpArgs}
run_seqkit filtered_fastas.tsv ${filteredArgs}
run_seqkit concat_fastas.tsv "${concat_fasta}"
"""
}

process PLOT_METADATA {
    cpus pipelineThreads
    conda "${plotMetadataCondaEnvPath}"

    when:
    metadataPlotsEnabled

    input:
    path(fastq_stats)
    path(asv_micro)
    path(asv_mito)
    path(taxonomy_table)

    output:
    path("metadata_updated_micro.tsv"), emit: metadata_micro
    path("ASV_meta_micro.tsv"), emit: asv_meta_micro
    path("ASV_final.micro.tsv"), emit: asv_final_micro
    path("metadata_updated_mito.tsv"), optional: true, emit: metadata_mito
    path("ASV_meta_mito.tsv"), optional: true, emit: asv_meta_mito
    path("ASV_final.mito.tsv"), optional: true, emit: asv_final_mito

    script:
    def includeRankArgs = metadataIncludeRank && !metadataIncludeRank.isEmpty() ?
        metadataIncludeRank.collect { "  --include-rank \"${it}\" \\\n" }.join('') : ''
    def microFlag = metadataForceMicroOnly ? "  --make-micro \\\n" : ''
    def mitoFlag = metadataForceMitoOnly ? "  --make-mito \\\n" : ''
    def metadataMicroFile = "${outputDir}/metadata/metadata_updated_micro.tsv"
    def metadataMitoFile = "${outputDir}/mito/metadata/metadata_updated_mito.tsv"
    def asvMetaMicroFile = "${outputDir}/metadata/ASV_meta_micro.tsv"
    def asvMetaMitoFile = "${outputDir}/mito/metadata/ASV_meta_mito.tsv"
    def asvFinalMicroFile = "${outputDir}/ASVs/ASV_final.micro.tsv"
    def asvFinalMitoFile = "${outputDir}/mito/ASVs/ASV_final.mito.tsv"
    def asvTaxTable = "${outputDir}/taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv"
    """
set -euo pipefail

python "${plotMetadataScriptPath}" \\
  --data-dir "${outputDir}" \\
  --sub-dir "${metadataPlotsSubDir}" \\
  --metadata "${metadataPlotsMetadataPath}" \\
  --taxonomy "${asvTaxTable}" \\
  --asv-micro "${asvFinalMicroFile}" \\
  --asv-mito "${asvFinalMitoFile}" \\
  --group1-col "${metadataPlotsTypeCol}" \\
  --color-col "${metadataPlotsColorCol}" \\
  --sample-manifest ${manifestPath} \\
  ${includeRankArgs} \\
  ${microFlag} \\
  ${mitoFlag} \\
  --verbose

link_if_exists() {
  local src="\$1"
  local dest="\$2"
  if [[ -f "\${src}" ]]; then
    ln -sf "\${src}" "\${dest}"
  fi
}

link_if_exists "${metadataMicroFile}" "metadata_updated_micro.tsv"
link_if_exists "${asvMetaMicroFile}" "ASV_meta_micro.tsv"
link_if_exists "${asvFinalMicroFile}" "ASV_final.micro.tsv"
link_if_exists "${metadataMitoFile}" "metadata_updated_mito.tsv"
link_if_exists "${asvMetaMitoFile}" "ASV_meta_mito.tsv"
link_if_exists "${asvFinalMitoFile}" "ASV_final.mito.tsv"
"""
}

process ASV_BATCH_CORRECTION {
    cpus pipelineThreads
    conda "${batchCorrectionCondaEnvPath}"

    when:
    batchCorrectionEnabled

    input:
    path(metadata_table)
    path(asv_meta)
    path(asv_counts)

    output:
    path("asv_clr_after_correction.tsv"), emit: asv_clr_after
    path("umap_hdbscan_results.tsv"), emit: umap_results

    script:
    def bioCovArg = batchBiologicalCovariates ? """  --biological-covariates "${batchBiologicalCovariates}" \\\n""" : ''
    def minSamplesArg = batchHdbscanMinSamples != null ? "  --hdbscan-min-samples ${batchHdbscanMinSamples} \\\n" : ''
    def optimizeFlag = batchOptimize ? "  --optimize-clustering \\\n" : ''
    def asvClrAfterFile = "${batchCorrectionOutputDirAbs}/asv_clr_after_correction.tsv"
    def umapResultsFile = "${batchCorrectionOutputDirAbs}/umap_hdbscan_results.tsv"
    """
set -euo pipefail

python "${batchCorrectionScriptPath}" \\
  --data-dir "${outputDir}" \\
  --asv "${asv_counts}" \\
  --metadata "${metadata_table}" \\
  --asv-meta "${asv_meta}" \\
  --batch-col "${batchCorrectionBatchCol}" \\
  --output-dir "${batchCorrectionOutputDir}" \\
  --asv-orientation "${batchCorrectionOrientation}" \\
${bioCovArg}  --umap-neighbors ${batchUmapNeighbors} \\
  --umap-min-dist ${batchUmapMinDist} \\
  --hdbscan-min-cluster-size ${batchHdbscanMinClusterSize} \\
${minSamplesArg}  --hdbscan-selection-method "${batchHdbscanSelectionMethod}" \\
  --target-clusters "${batchTargetClusters}" \\
  --n-features-plot ${batchNFeaturesPlot} \\
  --biological-color-col "${batchBiologicalColorCols}" \\
  --color-palette-col "${batchColorPaletteCols}" \\
  --random-state ${batchRandomState} \\
${optimizeFlag}  --verbose

if [[ ! -f "${asvClrAfterFile}" ]]; then
  echo "Missing batch correction output: ${asvClrAfterFile}" >&2
  exit 1
fi
ln -sf "${asvClrAfterFile}" asv_clr_after_correction.tsv
if [[ -f "${umapResultsFile}" ]]; then
  ln -sf "${umapResultsFile}" umap_hdbscan_results.tsv
fi
"""
}

process OUTLIER_CHECKER {
    cpus pipelineThreads
    conda "${outlierCondaEnvPath}"

    when:
    outlierEnabled

    input:
    path(asv_clr)
    path(metadata_table)

    script:
    def groupColsArg = outlierGroupCols.join(',')
    def isoFlag = outlierUseIso ? "  --use-iso \\\n" : ''
    def svmFlag = outlierUseSvm ? "  --use-svm \\\n" : ''
    def hdbFlag = outlierUseHdb ? "  --use-hdb \\\n" : ''
    def preTransFlag = outlierPreTransformed ? "  --pre-transformed \\\n" : ''
    def scaleFlag = outlierScale ? "  --scale \\\n" : ''
    def hdbMinSamplesArg = outlierHdbMinSamples != null ? "  --hdbscan-min-samples ${outlierHdbMinSamples} \\\n" : ''
    """
set -euo pipefail

python "${outlierCheckerScriptPath}" \\
  --data-dir "${outputDir}" \\
  --asv "${asv_clr}" \\
  --metadata "${metadata_table}" \\
  --output-dir "${outlierOutputDirAbs}" \\
  --group-cols "${groupColsArg}" \\
  --asv-orientation "${outlierOrientation}" \\
  --transform "${outlierTransform}" \\
${preTransFlag}${scaleFlag}${isoFlag}${svmFlag}${hdbFlag}  --vote-threshold ${outlierVoteThreshold} \\
  --iso-contamination "${outlierIsoContamination}" \\
  --iso-estimators ${outlierIsoEstimators} \\
  --iso-random-state ${outlierIsoRandomState} \\
  --svm-kernel "${outlierSvmKernel}" \\
  --svm-gamma "${outlierSvmGamma}" \\
  --svm-nu ${outlierSvmNu} \\
  --hdbscan-min-cluster-size ${outlierHdbMinClusterSize} \\
${hdbMinSamplesArg}  --hdbscan-metric "${outlierHdbMetric}" \\
  --verbose
"""
}

process COLLECTORS_CURVE {
    cpus pipelineThreads
    conda "${collectorsCondaEnvPath}"

    when:
    collectorsEnabled

    input:
    path(asv_counts)
    path(metadata_table)

    script:
    """
set -euo pipefail

python "${collectorsCurveScriptPath}" \\
  --counts "${asv_counts}" \\
  --meta "${metadata_table}" \\
  --group-col "${collectorsGroupCol}" \\
  --color-col "${collectorsColorCol}" \\
  --permutations ${collectorsPermutations} \\
  --seed ${collectorsSeed} \\
  --out_prefix "${collectorsOutPrefixAbs}" \\
  --title "${collectorsTitle}" \\
  --formats "${collectorsFormats}" \\
  --xpad ${collectorsXpad} \\
  --max-cols ${collectorsMaxCols} \\
  --show-perms ${collectorsShowPerms} \\
  --presence-threshold ${collectorsPresenceThreshold}
"""
}

def downloadReference(String downloadUrl, File destination) {
    destination.parentFile?.mkdirs()
    def tmpFile = File.createTempFile("sina_ref", ".download", destination.parentFile ?: new File('.'))
    tmpFile.withOutputStream { out ->
        new URL(downloadUrl).withInputStream { ins ->
            out << ins
        }
    }
    if( downloadUrl?.toLowerCase()?.endsWith('.gz') ) {
        destination.withOutputStream { out ->
            tmpFile.withInputStream { tmpIn ->
                new GZIPInputStream(tmpIn).withCloseable { gz ->
                    out << gz
                }
            }
        }
        tmpFile.delete()
    } else {
        if( !tmpFile.renameTo(destination) ) {
            tmpFile.withInputStream { ins ->
                destination.withOutputStream { out ->
                    out << ins
                }
            }
            tmpFile.delete()
        }
    }
}

def ensureBlastDbExists(String basePath, String label){
    def baseFile = new File(basePath)
    if( baseFile.exists() ) {
        return
    }
    def suffixes = ['.nhr','.nin','.nsq','.fa','.fasta','.fna']
    if( suffixes.any { new File(basePath + it).exists() } ) {
        return
    }
    exit 1, "${label} BLAST database not found: ${basePath}"
}

def shellQuote(String value){
    if( value == null ){
        return "''"
    }
    return "'" + value.toString().replace("'", "'\"'\"'") + "'"
}

def joinShellArgs(List paths){
    if( !paths ) {
        return ''
    }
    return paths.collect { shellQuote(it.toString()) }.join(' ')
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

def loadManifestSamples(String manifestPath){
    File manifest = new File(manifestPath)
    if( !manifest.exists() ) {
        exit 1, "Manifest file not found: ${manifestPath}"
    }
    def records = []
    manifest.eachLine { line ->
        def trimmed = line.trim()
        if( !trimmed || trimmed.startsWith('#') ) {
            return
        }
        def parts = trimmed.split(/\t/)
        if( parts.length < 2 ) {
            exit 1, "Manifest line must contain sample_id and R1 path separated by tab: ${line}"
        }
        def sampleId = parts[0].trim()
        def r1Path = resolveOptionalPath(parts[1].trim(), manifest.parentFile)
        def r2Path = parts.length > 2 && parts[2].trim() ? resolveOptionalPath(parts[2].trim(), manifest.parentFile) : null
        if( !sampleId ) {
            exit 1, "Sample ID missing in manifest line: ${line}"
        }
        if( !r1Path ) {
            exit 1, "R1 path missing in manifest line: ${line}"
        }
        def r1File = new File(r1Path)
        if( !r1File.exists() ) {
            exit 1, "R1 file not found for sample ${sampleId}: ${r1Path}"
        }
        File r2File = null
        boolean paired = false
        if( r2Path ) {
            r2File = new File(r2Path)
            if( !r2File.exists() ) {
                exit 1, "R2 file not found for sample ${sampleId}: ${r2Path}"
            }
            paired = true
        }
        records << [
            sample_id: sampleId,
            paired: paired,
            r1: r1File.canonicalPath,
            r2: paired ? r2File.canonicalPath : null
        ]
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

def resolveOutputRelative(String pathValue, String baseOutputDir){
    if( !pathValue ) return baseOutputDir
    def candidate = new File(pathValue)
    if( candidate.isAbsolute() ) {
        return candidate.canonicalPath
    }
    return new File(baseOutputDir ?: '.', pathValue).canonicalPath
}
