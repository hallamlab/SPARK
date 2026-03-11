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
if( config.merge && !(config.merge instanceof Map) ) {
    log.warn "Ignoring non-map merge configuration (${config.merge.getClass()?.simpleName})"
}
def mergeConfigMap = (config.merge instanceof Map) ? config.merge : [:]
def mergeMaxDiffs = mergeConfigMap.max_diffs != null ? (mergeConfigMap.max_diffs as int) : 20
def mergeMinOverlap = mergeConfigMap.min_overlap != null ? (mergeConfigMap.min_overlap as int) : 5
def mergeTruncQuality = mergeConfigMap.trunc_quality != null ? (mergeConfigMap.trunc_quality as int) : 5
boolean mergeAllowStagger = (mergeConfigMap.allow_stagger ?: false) as boolean

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

def biochemPreAsvEnvConfigPath = config.environments?.biochem_pre_asv ?: config.environments?.biochem
def resolvedBiochemPreAsvEnvPath = biochemPreAsvEnvConfigPath ? resolveOptionalPath(biochemPreAsvEnvConfigPath, configRoot) : null
def defaultBiochemPreAsvEnvPath = new File("${projectDir}/envs/biochem.yml").canonicalPath
def biochemPreAsvCondaEnvPath = resolvedBiochemPreAsvEnvPath ?: defaultBiochemPreAsvEnvPath
def biochemPreAsvEnvFile = file(biochemPreAsvCondaEnvPath)
if( !biochemPreAsvEnvFile.exists() ) {
    exit 1, "Biochem pre-ASV conda environment YAML not found: ${biochemPreAsvCondaEnvPath}"
}
log.info "Using biochem pre-ASV Conda/Mamba env definition: ${biochemPreAsvCondaEnvPath}"

def resolveBiochemStepEnv = { String envKey ->
    def stepEnvCfg = config.environments?."${envKey}"
    def stepEnvPath = stepEnvCfg ? resolveOptionalPath(stepEnvCfg, configRoot) : biochemPreAsvCondaEnvPath
    def stepEnvFile = file(stepEnvPath)
    if( !stepEnvFile.exists() ) {
        exit 1, "Biochem step conda environment YAML not found for ${envKey}: ${stepEnvPath}"
    }
    log.info "Using biochem Conda/Mamba env for ${envKey}: ${stepEnvPath}"
    return stepEnvPath
}
def biochemMergeCondaEnvPath = resolveBiochemStepEnv('biochem_merge')
def biochemDensityCondaEnvPath = resolveBiochemStepEnv('biochem_density')
def biochemStratMetricsCondaEnvPath = resolveBiochemStepEnv('biochem_strat_metrics')
def biochemCustomCleanCondaEnvPath = resolveBiochemStepEnv('biochem_custom_clean')
def biochemEigenvectorsCondaEnvPath = resolveBiochemStepEnv('biochem_eigenvectors')
def biochemSelectkCondaEnvPath = resolveBiochemStepEnv('biochem_selectk')
def biochemGmmCondaEnvPath = resolveBiochemStepEnv('biochem_gmm')
def biochemO2SoftCondaEnvPath = resolveBiochemStepEnv('biochem_o2_soft')
def biochemHybridCondaEnvPath = resolveBiochemStepEnv('biochem_hybrid')
def biochemCompareCondaEnvPath = resolveBiochemStepEnv('biochem_compare')
def biochemSplitCondaEnvPath = resolveBiochemStepEnv('biochem_split_o2_by_gmm')
def biochemStratAnomalyCondaEnvPath = resolveBiochemStepEnv('biochem_strat_anomaly')
def biochemStateTransitionsCondaEnvPath = resolveBiochemStepEnv('biochem_state_transitions')
def biochemSuccessionCondaEnvPath = resolveBiochemStepEnv('biochem_succession')
def biochemFeatureAssocCondaEnvPath = resolveBiochemStepEnv('biochem_feature_assoc')
def biochemEofPipelineCondaEnvPath = resolveBiochemStepEnv('biochem_eof_pipeline')
def biochemEofStateCondaEnvPath = resolveBiochemStepEnv('biochem_eof_state_cluster')
def biochemEofModeCondaEnvPath = resolveBiochemStepEnv('biochem_eof_mode_plots')
def biochemWithinGmmCondaEnvPath = resolveBiochemStepEnv('biochem_within_gmm')

def defaultAdvancedEnvPath = new File("${projectDir}/envs/advanced.yml").canonicalPath
def diversityEnvConfigPath = config.environments?.diversity
def resolvedDiversityEnvPath = diversityEnvConfigPath ? resolveOptionalPath(diversityEnvConfigPath, configRoot) : null
def diversityCondaEnvPath = resolvedDiversityEnvPath ?: defaultAdvancedEnvPath
def diversityEnvFile = file(diversityCondaEnvPath)
if( !diversityEnvFile.exists() ) {
    exit 1, "Diversity conda environment YAML not found: ${diversityCondaEnvPath}"
}
log.info "Using diversity Conda/Mamba env definition: ${diversityCondaEnvPath}"

def indicspeciesEnvConfigPath = config.environments?.indicspecies
def resolvedIndicspeciesEnvPath = indicspeciesEnvConfigPath ? resolveOptionalPath(indicspeciesEnvConfigPath, configRoot) : null
def indicspeciesCondaEnvPath = resolvedIndicspeciesEnvPath ?: defaultAdvancedEnvPath
def indicspeciesEnvFile = file(indicspeciesCondaEnvPath)
if( !indicspeciesEnvFile.exists() ) {
    exit 1, "Indicspecies conda environment YAML not found: ${indicspeciesCondaEnvPath}"
}
log.info "Using indicspecies Conda/Mamba env definition: ${indicspeciesCondaEnvPath}"

def clustermapsEnvConfigPath = config.environments?.clustermaps
def resolvedClustermapsEnvPath = clustermapsEnvConfigPath ? resolveOptionalPath(clustermapsEnvConfigPath, configRoot) : null
def clustermapsCondaEnvPath = resolvedClustermapsEnvPath ?: defaultAdvancedEnvPath
def clustermapsEnvFile = file(clustermapsCondaEnvPath)
if( !clustermapsEnvFile.exists() ) {
    exit 1, "Clustermaps conda environment YAML not found: ${clustermapsCondaEnvPath}"
}
log.info "Using clustermaps Conda/Mamba env definition: ${clustermapsCondaEnvPath}"

def spieceasiEnvConfigPath = config.environments?.spieceasi
def resolvedSpieceasiEnvPath = spieceasiEnvConfigPath ? resolveOptionalPath(spieceasiEnvConfigPath, configRoot) : null
def spieceasiCondaEnvPath = resolvedSpieceasiEnvPath ?: defaultAdvancedEnvPath
def spieceasiEnvFile = file(spieceasiCondaEnvPath)
if( !spieceasiEnvFile.exists() ) {
    exit 1, "SPIEC-EASI conda environment YAML not found: ${spieceasiCondaEnvPath}"
}
log.info "Using SPIEC-EASI Conda/Mamba env definition: ${spieceasiCondaEnvPath}"

def networkEnvConfigPath = config.environments?.network
def resolvedNetworkEnvPath = networkEnvConfigPath ? resolveOptionalPath(networkEnvConfigPath, configRoot) : null
def networkCondaEnvPath = resolvedNetworkEnvPath ?: defaultAdvancedEnvPath
def networkEnvFile = file(networkCondaEnvPath)
if( !networkEnvFile.exists() ) {
    exit 1, "Network conda environment YAML not found: ${networkCondaEnvPath}"
}
log.info "Using network Conda/Mamba env definition: ${networkCondaEnvPath}"

def networkModulesEnvConfigPath = config.environments?.network_modules
def resolvedNetworkModulesEnvPath = networkModulesEnvConfigPath ? resolveOptionalPath(networkModulesEnvConfigPath, configRoot) : null
def networkModulesCondaEnvPath = resolvedNetworkModulesEnvPath ?: spieceasiCondaEnvPath
def networkModulesEnvFile = file(networkModulesCondaEnvPath)
if( !networkModulesEnvFile.exists() ) {
    exit 1, "Network modules conda environment YAML not found: ${networkModulesCondaEnvPath}"
}
log.info "Using network modules Conda/Mamba env definition: ${networkModulesCondaEnvPath}"

def masterSummaryEnvConfigPath = config.environments?.master_summary
def resolvedMasterSummaryEnvPath = masterSummaryEnvConfigPath ? resolveOptionalPath(masterSummaryEnvConfigPath, configRoot) : null
def masterSummaryCondaEnvPath = resolvedMasterSummaryEnvPath ?: defaultAdvancedEnvPath
def masterSummaryEnvFile = file(masterSummaryCondaEnvPath)
if( !masterSummaryEnvFile.exists() ) {
    exit 1, "Master summary conda environment YAML not found: ${masterSummaryCondaEnvPath}"
}
log.info "Using master summary Conda/Mamba env definition: ${masterSummaryCondaEnvPath}"

def plotUpsetEnvConfigPath = config.environments?.plot_upset
def resolvedPlotUpsetEnvPath = plotUpsetEnvConfigPath ? resolveOptionalPath(plotUpsetEnvConfigPath, configRoot) : null
def plotUpsetCondaEnvPath = resolvedPlotUpsetEnvPath ?: defaultAdvancedEnvPath
def plotUpsetEnvFile = file(plotUpsetCondaEnvPath)
if( !plotUpsetEnvFile.exists() ) {
    exit 1, "Plot Upset conda environment YAML not found: ${plotUpsetCondaEnvPath}"
}
log.info "Using Plot Upset Conda/Mamba env definition: ${plotUpsetCondaEnvPath}"

def bubbleplotterEnvConfigPath = config.environments?.bubbleplotter
def resolvedBubbleplotterEnvPath = bubbleplotterEnvConfigPath ? resolveOptionalPath(bubbleplotterEnvConfigPath, configRoot) : null
def bubbleplotterCondaEnvPath = resolvedBubbleplotterEnvPath ?: plotUpsetCondaEnvPath
def bubbleplotterEnvFile = file(bubbleplotterCondaEnvPath)
if( !bubbleplotterEnvFile.exists() ) {
    exit 1, "Bubbleplotter conda environment YAML not found: ${bubbleplotterCondaEnvPath}"
}
log.info "Using bubbleplotter Conda/Mamba env definition: ${bubbleplotterCondaEnvPath}"

def umapClusteringEnvConfigPath = config.environments?.umap_clustering
def resolvedUmapClusteringEnvPath = umapClusteringEnvConfigPath ? resolveOptionalPath(umapClusteringEnvConfigPath, configRoot) : null
def umapClusteringCondaEnvPath = resolvedUmapClusteringEnvPath ?: plotUpsetCondaEnvPath
def umapClusteringEnvFile = file(umapClusteringCondaEnvPath)
if( !umapClusteringEnvFile.exists() ) {
    exit 1, "UMAP clustering conda environment YAML not found: ${umapClusteringCondaEnvPath}"
}
log.info "Using UMAP clustering Conda/Mamba env definition: ${umapClusteringCondaEnvPath}"

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

def concatConfig = config.concat ?: [:]
def concatRelabelEnabled = concatConfig.containsKey('relabel') ? (concatConfig.relabel as boolean) : true
def concatLabelSep = concatConfig.label_sep ?: ':'

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
def calcDivScriptFile = new File("${projectDir}/calc_div.py")
if( !calcDivScriptFile.exists() ) {
    exit 1, "calc_div.py not found in project directory"
}
def calcDivScriptPath = calcDivScriptFile.canonicalPath
def plotDiversityScriptFile = new File("${projectDir}/plot_diversity.py")
if( !plotDiversityScriptFile.exists() ) {
    exit 1, "plot_diversity.py not found in project directory"
}
def plotDiversityScriptPath = plotDiversityScriptFile.canonicalPath
def plotUpsetScriptFile = new File("${projectDir}/plot_upset.py")
if( !plotUpsetScriptFile.exists() ) {
    exit 1, "plot_upset.py not found in project directory"
}
def plotUpsetScriptPath = plotUpsetScriptFile.canonicalPath
def bubbleplotterScriptFile = new File("${projectDir}/bubbleplotter.py")
if( !bubbleplotterScriptFile.exists() ) {
    exit 1, "bubbleplotter.py not found in project directory"
}
def bubbleplotterScriptPath = bubbleplotterScriptFile.canonicalPath
def umapClusteringScriptFile = new File("${projectDir}/umap_clustering.py")
if( !umapClusteringScriptFile.exists() ) {
    exit 1, "umap_clustering.py not found in project directory"
}
def umapClusteringScriptPath = umapClusteringScriptFile.canonicalPath
def indicspeciesScriptFile = new File("${projectDir}/run_indicspecies.R")
if( !indicspeciesScriptFile.exists() ) {
    exit 1, "run_indicspecies.R not found in project directory"
}
def indicspeciesScriptPath = indicspeciesScriptFile.canonicalPath
def plotIndicspeciesScriptFile = new File("${projectDir}/plot_indicspecies.py")
if( !plotIndicspeciesScriptFile.exists() ) {
    exit 1, "plot_indicspecies.py not found in project directory"
}
def plotIndicspeciesScriptPath = plotIndicspeciesScriptFile.canonicalPath
def clustermapsScriptFile = new File("${projectDir}/plot_clustermaps.py")
if( !clustermapsScriptFile.exists() ) {
    exit 1, "plot_clustermaps.py not found in project directory"
}
def clustermapsScriptPath = clustermapsScriptFile.canonicalPath
def spieceasiScriptFile = new File("${projectDir}/run_spieceasi.R")
if( !spieceasiScriptFile.exists() ) {
    exit 1, "run_spieceasi.R not found in project directory"
}
def spieceasiScriptPath = spieceasiScriptFile.canonicalPath
def networkModulesScriptFile = new File("${projectDir}/network_modules.R")
if( !networkModulesScriptFile.exists() ) {
    exit 1, "network_modules.R not found in project directory"
}
def networkModulesScriptPath = networkModulesScriptFile.canonicalPath
def graphNetworkScriptFile = new File("${projectDir}/graph_network.py")
if( !graphNetworkScriptFile.exists() ) {
    exit 1, "graph_network.py not found in project directory"
}
def graphNetworkScriptPath = graphNetworkScriptFile.canonicalPath
def masterSummaryScriptFile = new File("${projectDir}/summary/build_master_asv_summary.py")
if( !masterSummaryScriptFile.exists() ) {
    exit 1, "summary/build_master_asv_summary.py not found in project directory"
}
def masterSummaryScriptPath = masterSummaryScriptFile.canonicalPath
def emptyModulesScriptFile = new File("${projectDir}/empty_modules.tsv")
if( !emptyModulesScriptFile.exists() ) {
    exit 1, "empty_modules.tsv not found in project directory"
}
def emptyModulesPath = emptyModulesScriptFile.canonicalPath
def biochemMergeTablesScriptFile = new File("${projectDir}/biochem_modeling/merge_tables_ctd_nearest_depth.py")
if( !biochemMergeTablesScriptFile.exists() ) {
    exit 1, "biochem_modeling/merge_tables_ctd_nearest_depth.py not found in project directory"
}
def biochemMergeTablesScriptPath = biochemMergeTablesScriptFile.canonicalPath
def biochemCalcDensityScriptFile = new File("${projectDir}/biochem_modeling/env_calc_density.py")
if( !biochemCalcDensityScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_calc_density.py not found in project directory"
}
def biochemCalcDensityScriptPath = biochemCalcDensityScriptFile.canonicalPath
def biochemStratMetricsScriptFile = new File("${projectDir}/biochem_modeling/env_stratification_metrics.py")
if( !biochemStratMetricsScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_stratification_metrics.py not found in project directory"
}
def biochemStratMetricsScriptPath = biochemStratMetricsScriptFile.canonicalPath
def biochemCustomCleanerScriptFile = new File("${projectDir}/biochem_modeling/custom_density_cleaner.py")
if( !biochemCustomCleanerScriptFile.exists() ) {
    exit 1, "biochem_modeling/custom_density_cleaner.py not found in project directory"
}
def biochemCustomCleanerScriptPath = biochemCustomCleanerScriptFile.canonicalPath
def biochemEigenvectorsScriptFile = new File("${projectDir}/biochem_modeling/env_eigenvectors.py")
if( !biochemEigenvectorsScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_eigenvectors.py not found in project directory"
}
def biochemEigenvectorsScriptPath = biochemEigenvectorsScriptFile.canonicalPath
def biochemSelectkScriptFile = new File("${projectDir}/biochem_modeling/env_compartments_selectk.py")
if( !biochemSelectkScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_compartments_selectk.py not found in project directory"
}
def biochemSelectkScriptPath = biochemSelectkScriptFile.canonicalPath
def biochemGmmScriptFile = new File("${projectDir}/biochem_modeling/env_compartments_gmm.py")
if( !biochemGmmScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_compartments_gmm.py not found in project directory"
}
def biochemGmmScriptPath = biochemGmmScriptFile.canonicalPath
def biochemO2SoftScriptFile = new File("${projectDir}/biochem_modeling/env_compartments_o2_soft.py")
if( !biochemO2SoftScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_compartments_o2_soft.py not found in project directory"
}
def biochemO2SoftScriptPath = biochemO2SoftScriptFile.canonicalPath
def biochemHybridScriptFile = new File("${projectDir}/biochem_modeling/env_hybrid_compartment_builder.py")
if( !biochemHybridScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_hybrid_compartment_builder.py not found in project directory"
}
def biochemHybridScriptPath = biochemHybridScriptFile.canonicalPath
def biochemCompareScriptFile = new File("${projectDir}/biochem_modeling/env_compare_compartments.py")
if( !biochemCompareScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_compare_compartments.py not found in project directory"
}
def biochemCompareScriptPath = biochemCompareScriptFile.canonicalPath
def biochemSplitScriptFile = new File("${projectDir}/biochem_modeling/env_split_o2_by_gmm.py")
if( !biochemSplitScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_split_o2_by_gmm.py not found in project directory"
}
def biochemSplitScriptPath = biochemSplitScriptFile.canonicalPath
def biochemStratAnomalyScriptFile = new File("${projectDir}/biochem_modeling/env_stratification_anomaly_detection.py")
if( !biochemStratAnomalyScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_stratification_anomaly_detection.py not found in project directory"
}
def biochemStratAnomalyScriptPath = biochemStratAnomalyScriptFile.canonicalPath
def biochemStateTransitionScriptFile = new File("${projectDir}/biochem_modeling/env_state_transition_analysis.py")
if( !biochemStateTransitionScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_state_transition_analysis.py not found in project directory"
}
def biochemStateTransitionScriptPath = biochemStateTransitionScriptFile.canonicalPath
def biochemSuccessionScriptFile = new File("${projectDir}/biochem_modeling/env_succession_graph.py")
if( !biochemSuccessionScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_succession_graph.py not found in project directory"
}
def biochemSuccessionScriptPath = biochemSuccessionScriptFile.canonicalPath
def biochemFeatureAssocScriptFile = new File("${projectDir}/biochem_modeling/env_compartment_feature_assoc.py")
if( !biochemFeatureAssocScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_compartment_feature_assoc.py not found in project directory"
}
def biochemFeatureAssocScriptPath = biochemFeatureAssocScriptFile.canonicalPath
def biochemEofPipelineScriptFile = new File("${projectDir}/biochem_modeling/env_eof_pipeline.py")
if( !biochemEofPipelineScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_eof_pipeline.py not found in project directory"
}
def biochemEofPipelineScriptPath = biochemEofPipelineScriptFile.canonicalPath
def biochemEofStateScriptFile = new File("${projectDir}/biochem_modeling/eof_state_clustering.py")
if( !biochemEofStateScriptFile.exists() ) {
    exit 1, "biochem_modeling/eof_state_clustering.py not found in project directory"
}
def biochemEofStateScriptPath = biochemEofStateScriptFile.canonicalPath
def biochemEofModePlotScriptFile = new File("${projectDir}/biochem_modeling/eof_mode_plots.py")
if( !biochemEofModePlotScriptFile.exists() ) {
    exit 1, "biochem_modeling/eof_mode_plots.py not found in project directory"
}
def biochemEofModePlotScriptPath = biochemEofModePlotScriptFile.canonicalPath
def biochemWithinGmmScriptFile = new File("${projectDir}/biochem_modeling/env_within_gmm_hdbscan.py")
if( !biochemWithinGmmScriptFile.exists() ) {
    exit 1, "biochem_modeling/env_within_gmm_hdbscan.py not found in project directory"
}
def biochemWithinGmmScriptPath = biochemWithinGmmScriptFile.canonicalPath
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
def sankeyArrangement = (sankeyConfig.arrangement ?: 'snap').toString().trim().toLowerCase()
if( !(sankeyArrangement in ['snap', 'perpendicular', 'freeform', 'fixed']) ) {
    exit 1, "Invalid sankey.arrangement '${sankeyArrangement}'. Allowed: snap, perpendicular, freeform, fixed"
}
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
def metadataPlotsSampleCol = metadataPlotsConfig.sample_col ?: 'sampleID'
def metadataPlotsTypeCol = metadataPlotsConfig.type_col ?: (metadataPlotsConfig.group1_col ?: 'Depth')
def metadataPlotsColorCol = metadataPlotsConfig.color_col ?: 'Color'
def metadataPlotsBiochemAssignmentsPath = metadataPlotsConfig.biochem_assignments ? resolveOptionalPath(metadataPlotsConfig.biochem_assignments, configRoot) : null
def metadataPlotsBiochemSampleCol = metadataPlotsConfig.biochem_sample_col ?: 'cruise_year_month_depth'
def metadataPlotsStratificationTimeseriesPath = metadataPlotsConfig.stratification_timeseries ? resolveOptionalPath(metadataPlotsConfig.stratification_timeseries, configRoot) : null
def metadataPlotsStratMetaJoinCol = metadataPlotsConfig.strat_meta_join_col ?: 'Cruise'
def metadataPlotsStratJoinCol = metadataPlotsConfig.strat_join_col ?: 'Cruise'
def metadataBiochemIncludeRaw = metadataPlotsConfig.biochem_include_cols
List<String> metadataPlotsBiochemIncludeCols = []
if( metadataBiochemIncludeRaw instanceof List ) {
    metadataPlotsBiochemIncludeCols = metadataBiochemIncludeRaw.collect { it.toString().trim() }.findAll { it }
} else if( metadataBiochemIncludeRaw ) {
    metadataPlotsBiochemIncludeCols = metadataBiochemIncludeRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def metadataStratIncludeRaw = metadataPlotsConfig.strat_include_cols
List<String> metadataPlotsStratIncludeCols = []
if( metadataStratIncludeRaw instanceof List ) {
    metadataPlotsStratIncludeCols = metadataStratIncludeRaw.collect { it.toString().trim() }.findAll { it }
} else if( metadataStratIncludeRaw ) {
    metadataPlotsStratIncludeCols = metadataStratIncludeRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def metadataBiochemMetaJoinRaw = metadataPlotsConfig.biochem_meta_join_cols
List<String> metadataPlotsBiochemMetaJoinCols = []
if( metadataBiochemMetaJoinRaw instanceof List ) {
    metadataPlotsBiochemMetaJoinCols = metadataBiochemMetaJoinRaw.collect { it.toString().trim() }.findAll { it }
} else if( metadataBiochemMetaJoinRaw ) {
    metadataPlotsBiochemMetaJoinCols = metadataBiochemMetaJoinRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def metadataBiochemJoinRaw = metadataPlotsConfig.biochem_join_cols
List<String> metadataPlotsBiochemJoinCols = []
if( metadataBiochemJoinRaw instanceof List ) {
    metadataPlotsBiochemJoinCols = metadataBiochemJoinRaw.collect { it.toString().trim() }.findAll { it }
} else if( metadataBiochemJoinRaw ) {
    metadataPlotsBiochemJoinCols = metadataBiochemJoinRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
if( metadataPlotsBiochemMetaJoinCols.size() != metadataPlotsBiochemJoinCols.size() ) {
    exit 1, "metadata_plots.biochem_meta_join_cols and metadata_plots.biochem_join_cols must have the same number of entries"
}
def metadataKeepTypesRaw = metadataPlotsConfig.keep_types
List<String> metadataKeepTypes = []
if( metadataKeepTypesRaw instanceof List ) {
    metadataKeepTypes = metadataKeepTypesRaw.collect { it.toString().trim() }.findAll { it }
} else if( metadataKeepTypesRaw ) {
    metadataKeepTypes = metadataKeepTypesRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def metadataGroupOrderRaw = metadataPlotsConfig.group_order
List<String> metadataPlotsGroupOrder = []
if( metadataGroupOrderRaw instanceof List ) {
    metadataPlotsGroupOrder = metadataGroupOrderRaw.collect { it.toString().trim() }.findAll { it }
} else if( metadataGroupOrderRaw ) {
    metadataPlotsGroupOrder = metadataGroupOrderRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
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
def batchCorrectionSampleIdCol = batchCorrectionConfig.sample_id_col ?: metadataPlotsSampleCol
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
def batchConqurMode = batchCorrectionConfig.conqur_mode ?: 'libsize'
def batchConqurNumCore = batchCorrectionConfig.conqur_num_core ? (batchCorrectionConfig.conqur_num_core as int) : pipelineThreads
def batchConqurBatchRef = batchCorrectionConfig.conqur_batch_ref ? batchCorrectionConfig.conqur_batch_ref.toString().trim() : ''
boolean batchConqurLogisticLasso = (batchCorrectionConfig.conqur_logistic_lasso ?: false) as boolean
def batchConqurQuantileType = batchCorrectionConfig.conqur_quantile_type ?: 'standard'
boolean batchConqurSimpleMatch = (batchCorrectionConfig.conqur_simple_match ?: false) as boolean
def batchConqurLambdaQuantile = batchCorrectionConfig.conqur_lambda_quantile ?: '2p/n'
boolean batchConqurInterplt = (batchCorrectionConfig.conqur_interplt ?: false) as boolean
def batchConqurDelta = batchCorrectionConfig.conqur_delta != null ? (batchCorrectionConfig.conqur_delta as double) : 0.4999d
boolean batchConqurAutoInstall = (batchCorrectionConfig.conqur_auto_install ?: false) as boolean

def outlierConfig = config.outlier_detection ?: [:]
boolean outlierEnabled = batchCorrectionEnabled && (outlierConfig.containsKey('enabled') ? (outlierConfig.enabled as boolean) : true)
def outlierOutputDir = outlierConfig.output_dir ?: 'outliers_corrected'
def outlierOutputDirAbs = new File(outputDir, outlierOutputDir).canonicalPath
def outlierSampleIdCol = outlierConfig.sample_id_col ?: (outlierConfig.sample_col ?: metadataPlotsSampleCol)
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
def collectorsSampleCol = collectorsConfig.sample_col ?: metadataPlotsSampleCol
def collectorsGroupCol = collectorsConfig.group_col ?: (collectorsConfig.group1_col ?: metadataPlotsTypeCol)
def collectorsColorCol = collectorsConfig.color_col ?: 'Color'
def collectorsGroupOrderRaw = collectorsConfig.group_order ?: metadataPlotsGroupOrder
List<String> collectorsGroupOrder = []
if( collectorsGroupOrderRaw instanceof List ) {
    collectorsGroupOrder = collectorsGroupOrderRaw.collect { it.toString().trim() }.findAll { it }
} else if( collectorsGroupOrderRaw ) {
    collectorsGroupOrder = collectorsGroupOrderRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
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

def plotUpsetConfig = config.plot_upset ?: [:]
boolean plotUpsetRequested = plotUpsetConfig.containsKey('enabled') ? (plotUpsetConfig.enabled as boolean) : false
if( plotUpsetRequested && !metadataPlotsEnabled ) {
    exit 1, "plot_upset.enabled requires metadata_plots.enabled to be true"
}
boolean plotUpsetEnabled = plotUpsetRequested
def plotUpsetSubDir = plotUpsetConfig.sub_dir ?: '.'
def plotUpsetDomain = plotUpsetConfig.domain ?: 'micro'
def plotUpsetTaxonomyPath = plotUpsetConfig.taxonomy_path ? resolveOptionalPath(plotUpsetConfig.taxonomy_path, configRoot) : null
def plotUpsetSampleIdCol = plotUpsetConfig.sample_id_col ?: metadataPlotsSampleCol
def plotUpsetGroupCol = plotUpsetConfig.group_col ?: (plotUpsetConfig.group1_col ?: metadataPlotsTypeCol)
def plotUpsetColorCol = plotUpsetConfig.color_col ?: metadataPlotsColorCol
def plotUpsetGroupOrderRaw = plotUpsetConfig.group_order ?: metadataPlotsGroupOrder
List<String> plotUpsetGroupOrder = []
if( plotUpsetGroupOrderRaw instanceof List ) {
    plotUpsetGroupOrder = plotUpsetGroupOrderRaw.collect { it.toString().trim() }.findAll { it }
} else if( plotUpsetGroupOrderRaw ) {
    plotUpsetGroupOrder = plotUpsetGroupOrderRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def plotUpsetSubsetGroupsRaw = plotUpsetConfig.subset_groups
List<String> plotUpsetSubsetGroups = []
if( plotUpsetSubsetGroupsRaw instanceof List ) {
    plotUpsetSubsetGroups = plotUpsetSubsetGroupsRaw.collect { it.toString().trim() }.findAll { it }
} else if( plotUpsetSubsetGroupsRaw ) {
    plotUpsetSubsetGroups = plotUpsetSubsetGroupsRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
boolean plotUpsetSkipVenn = plotUpsetConfig.containsKey('skip_venn') ? (plotUpsetConfig.skip_venn as boolean) : true
def plotUpsetFormats = plotUpsetConfig.formats ?: 'pdf,svg,png'
def plotUpsetFontSize = plotUpsetConfig.font_size != null ? (plotUpsetConfig.font_size as double) : 12d
boolean plotUpsetRawOnly = plotUpsetConfig.containsKey('raw_only') ? (plotUpsetConfig.raw_only as boolean) : false
boolean plotUpsetFinalOnly = plotUpsetConfig.containsKey('final_only') ? (plotUpsetConfig.final_only as boolean) : false
if( plotUpsetRawOnly && plotUpsetFinalOnly ) {
    exit 1, "plot_upset.raw_only and plot_upset.final_only cannot both be true"
}

def bubbleplotterConfig = config.bubbleplotter ?: [:]
boolean bubbleplotterRequested = bubbleplotterConfig.containsKey('enabled') ? (bubbleplotterConfig.enabled as boolean) : false
if( bubbleplotterRequested && !metadataPlotsEnabled ) {
    exit 1, "bubbleplotter.enabled requires metadata_plots.enabled to be true"
}
boolean bubbleplotterEnabled = bubbleplotterRequested
def bubbleplotterOutputPrefix = bubbleplotterConfig.output_prefix ?: 'metadata/bubble_plot_asv'
def bubbleplotterOutputPrefixAbs = new File(outputDir, bubbleplotterOutputPrefix).canonicalPath
def bubbleplotterOutputDirAbs = (new File(bubbleplotterOutputPrefixAbs).parentFile ?: new File(outputDir)).canonicalPath
def bubbleplotterFormats = bubbleplotterConfig.formats ?: 'pdf,png'
def bubbleplotterCountCol = bubbleplotterConfig.count_col ?: 'count'
def bubbleplotterSampleCol = bubbleplotterConfig.sample_col ?: metadataPlotsSampleCol
def bubbleplotterDepthCol = bubbleplotterConfig.group1_col ?: (bubbleplotterConfig.depth_col ?: metadataPlotsTypeCol)
def bubbleplotterColorCol = bubbleplotterConfig.color_col ?: metadataPlotsColorCol
def bubbleplotterMonthCol = bubbleplotterConfig.group2_col ?: (bubbleplotterConfig.month_col ?: 'Month')
def bubbleplotterGroup1OrderRaw = bubbleplotterConfig.group1_order ?: metadataPlotsGroupOrder
List<String> bubbleplotterGroup1Order = []
if( bubbleplotterGroup1OrderRaw instanceof List ) {
    bubbleplotterGroup1Order = bubbleplotterGroup1OrderRaw.collect { it.toString().trim() }.findAll { it }
} else if( bubbleplotterGroup1OrderRaw ) {
    bubbleplotterGroup1Order = bubbleplotterGroup1OrderRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def bubbleplotterGroup2OrderRaw = bubbleplotterConfig.group2_order ?: (config.indicspecies?.group2_order ?: '')
List<String> bubbleplotterGroup2Order = []
if( bubbleplotterGroup2OrderRaw instanceof List ) {
    bubbleplotterGroup2Order = bubbleplotterGroup2OrderRaw.collect { it.toString().trim() }.findAll { it }
} else if( bubbleplotterGroup2OrderRaw ) {
    bubbleplotterGroup2Order = bubbleplotterGroup2OrderRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def bubbleplotterFigsize = bubbleplotterConfig.figsize ?: '32,60'
def bubbleplotterScale = bubbleplotterConfig.bubble_scale != null ? (bubbleplotterConfig.bubble_scale as double) : 10d
boolean bubbleplotterNoAutoSize = bubbleplotterConfig.containsKey('no_auto_size') ? (bubbleplotterConfig.no_auto_size as boolean) : true

def umapClusteringConfig = config.umap_clustering ?: [:]
boolean umapClusteringRequested = umapClusteringConfig.containsKey('enabled') ? (umapClusteringConfig.enabled as boolean) : false
if( umapClusteringRequested && !metadataPlotsEnabled ) {
    exit 1, "umap_clustering.enabled requires metadata_plots.enabled to be true"
}
boolean umapClusteringEnabled = umapClusteringRequested
def umapClusteringOutputPrefix = umapClusteringConfig.output_prefix ?: 'metadata/umap_clustering'
def umapClusteringOutputPrefixAbs = new File(outputDir, umapClusteringOutputPrefix).canonicalPath
def umapClusteringOutputDirAbs = (new File(umapClusteringOutputPrefixAbs).parentFile ?: new File(outputDir)).canonicalPath
def umapClusteringSampleCol = umapClusteringConfig.sample_col ?: metadataPlotsSampleCol
def umapClusteringCountCol = umapClusteringConfig.count_col ?: 'count'
def umapClusteringDepthCol = umapClusteringConfig.group1_col ?: (umapClusteringConfig.depth_col ?: metadataPlotsTypeCol)
def umapClusteringColorCol = umapClusteringConfig.color_col ?: metadataPlotsColorCol
def umapClusteringSecondaryCol = umapClusteringConfig.group2_col ?: (umapClusteringConfig.secondary_col ?: (umapClusteringConfig.month_col ?: 'Month'))
def umapClusteringGroup1OrderRaw = umapClusteringConfig.group1_order ?: metadataPlotsGroupOrder
List<String> umapClusteringGroup1Order = []
if( umapClusteringGroup1OrderRaw instanceof List ) {
    umapClusteringGroup1Order = umapClusteringGroup1OrderRaw.collect { it.toString().trim() }.findAll { it }
} else if( umapClusteringGroup1OrderRaw ) {
    umapClusteringGroup1Order = umapClusteringGroup1OrderRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def umapClusteringGroup2OrderRaw = umapClusteringConfig.group2_order ?: (config.indicspecies?.group2_order ?: '')
List<String> umapClusteringGroup2Order = []
if( umapClusteringGroup2OrderRaw instanceof List ) {
    umapClusteringGroup2Order = umapClusteringGroup2OrderRaw.collect { it.toString().trim() }.findAll { it }
} else if( umapClusteringGroup2OrderRaw ) {
    umapClusteringGroup2Order = umapClusteringGroup2OrderRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def umapClusteringFormats = umapClusteringConfig.formats ?: 'pdf,png'
def umapClusteringNormalize = umapClusteringConfig.normalize ?: 'clr'
def umapClusteringTransform = umapClusteringConfig.transform ?: 'sqrt'
def umapClusteringNeighbors = umapClusteringConfig.n_neighbors ? (umapClusteringConfig.n_neighbors as int) : 15
def umapClusteringMinDist = umapClusteringConfig.min_dist != null ? (umapClusteringConfig.min_dist as double) : 0.1d
def umapClusteringMetric = umapClusteringConfig.umap_metric ?: 'euclidean'
def umapClusteringHdbscanMetric = umapClusteringConfig.hdbscan_metric ?: 'euclidean'
def umapClusteringMinClusterSize = umapClusteringConfig.min_cluster_size ? (umapClusteringConfig.min_cluster_size as int) : 10
def umapClusteringMinSamples = umapClusteringConfig.min_samples ? (umapClusteringConfig.min_samples as int) : 5
boolean umapClusteringNoScale = (umapClusteringConfig.no_scale ?: false) as boolean
def umapClusteringRandomState = umapClusteringConfig.random_state ? (umapClusteringConfig.random_state as int) : 42

def biochemPreAsvConfig = config.biochem_pre_asv ?: [:]
boolean biochemPreAsvEnabled = biochemPreAsvConfig.containsKey('enabled') ? (biochemPreAsvConfig.enabled as boolean) : false
def biochemTableAPath = biochemPreAsvConfig.table_a ? resolveOptionalPath(biochemPreAsvConfig.table_a, configRoot) : resolveOptionalPath('../ref_db/new_biochem/SI_JA_Compiled_Geochem_Dec_09_Outlier_RM.csv', configRoot)
def biochemTableBPath = biochemPreAsvConfig.table_b ? resolveOptionalPath(biochemPreAsvConfig.table_b, configRoot) : resolveOptionalPath('../ref_db/new_biochem/SI_JA_Compiled_CTD_Data_Dec_18_2025_Outlier_RM.csv', configRoot)
if( biochemPreAsvEnabled && (!biochemTableAPath || !new File(biochemTableAPath).exists()) ) {
    exit 1, "biochem_pre_asv.table_a not found: ${biochemTableAPath}"
}
if( biochemPreAsvEnabled && (!biochemTableBPath || !new File(biochemTableBPath).exists()) ) {
    exit 1, "biochem_pre_asv.table_b not found: ${biochemTableBPath}"
}
def biochemOutputRoot = biochemPreAsvConfig.output_root ? resolveOutputRelative(biochemPreAsvConfig.output_root.toString(), outputDir) : outputDir
def biochemProcessingDirAbs = new File(biochemOutputRoot, 'biochem_processing').canonicalPath
def biochemStratMetricsDirAbs = new File(biochemProcessingDirAbs, 'stratification_metrics').canonicalPath
def biochemPcaDirAbs = new File(biochemOutputRoot, 'env_pca').canonicalPath
def biochemSelectkDirAbs = new File(biochemOutputRoot, 'env_compartments_selectk').canonicalPath
def biochemGmmDirAbs = new File(biochemOutputRoot, 'env_compartments_gmm').canonicalPath
def biochemO2DirAbs = new File(biochemOutputRoot, 'env_o2_soft_compartments').canonicalPath
def biochemHybridDirAbs = new File(biochemOutputRoot, 'env_hybrid_soft_compartments').canonicalPath
def biochemCompareDirAbs = new File(biochemOutputRoot, 'env_compare_compartments').canonicalPath
def biochemSplitDirAbs = new File(biochemOutputRoot, 'env_o2_split_by_gmm').canonicalPath
def biochemStratIndexDirAbs = new File(biochemOutputRoot, 'env_stratification_index').canonicalPath
def biochemStateTransitionsDirAbs = new File(biochemOutputRoot, 'env_state_transitions').canonicalPath
def biochemSuccessionDirAbs = new File(biochemOutputRoot, 'env_succession_graphs').canonicalPath
def biochemFeatureAssocDirAbs = new File(biochemOutputRoot, 'env_compartment_feature_assoc').canonicalPath
def biochemEofPcaDirAbs = new File(biochemOutputRoot, 'eof_pca').canonicalPath
def biochemEofStatesDirAbs = new File(biochemOutputRoot, 'eof_states').canonicalPath
def biochemEofPlotsDirAbs = new File(biochemOutputRoot, 'eof_plots').canonicalPath
def biochemWithinGmmDirAbs = new File(biochemGmmDirAbs, 'within_gmm_hdbscan').canonicalPath
def biochemMergedOxygenPath = new File(biochemProcessingDirAbs, '02_oxygen_best_available.tsv').canonicalPath
def biochemDensityPath = new File(biochemProcessingDirAbs, '02_oxygen_best_available_density.tsv').canonicalPath
def biochemDensityCleanedFile = biochemPreAsvConfig.cleaned_density_filename ?: '02_oxygen_best_available_density_RJM.tsv'
def biochemDensityCleanedPath = new File(biochemProcessingDirAbs, biochemDensityCleanedFile.toString()).canonicalPath
def biochemFeatureCols = biochemPreAsvConfig.feature_cols ?: 'Oxygen,Nitrate,Nitrite,Nitrous Oxide,Ammonium,Hydrogen Sulfide,Methane,Phosphate,Silicate,Temperature,Salinity,Density,Fe,Dimethyl Sulfide'
def biochemGmmKRaw = biochemPreAsvConfig.gmm_k
boolean biochemGmmKAuto = (biochemGmmKRaw == null) || (biochemGmmKRaw.toString().trim().equalsIgnoreCase('auto'))
def biochemGmmK = biochemGmmKAuto ? 5 : (biochemGmmKRaw as int)
def biochemEofPcs = biochemPreAsvConfig.eof_pcs ?: '1,2,4'
def biochemCleanKeepRaw = biochemPreAsvConfig.clean_keep_cols
List<String> biochemCleanKeepCols = []
if( biochemCleanKeepRaw instanceof List ) {
    biochemCleanKeepCols = biochemCleanKeepRaw.collect { it.toString().trim() }.findAll { it }
} else if( biochemCleanKeepRaw ) {
    biochemCleanKeepCols = biochemCleanKeepRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def biochemCleanDropRaw = biochemPreAsvConfig.clean_drop_cols
List<String> biochemCleanDropCols = []
if( biochemCleanDropRaw instanceof List ) {
    biochemCleanDropCols = biochemCleanDropRaw.collect { it.toString().trim() }.findAll { it }
} else if( biochemCleanDropRaw ) {
    biochemCleanDropCols = biochemCleanDropRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def biochemCleanRenameRaw = biochemPreAsvConfig.clean_rename_map
Map<String, String> biochemCleanRenameMap = [:]
if( biochemCleanRenameRaw instanceof Map ) {
    biochemCleanRenameRaw.each { k, v ->
        if( k != null && v != null ) {
            def kk = k.toString().trim()
            def vv = v.toString().trim()
            if( kk && vv ) {
                biochemCleanRenameMap[kk] = vv
            }
        }
    }
}

def diversityConfig = config.diversity ?: [:]
boolean diversityRequested = diversityConfig.containsKey('enabled') ? (diversityConfig.enabled as boolean) : false
if( diversityRequested && !metadataPlotsEnabled ) {
    exit 1, "diversity.enabled requires metadata_plots.enabled to be true"
}
boolean diversityEnabled = diversityRequested
def diversityOutputDir = diversityConfig.output_dir ?: 'diversity'
def diversityOutputDirAbs = new File(outputDir, diversityOutputDir).canonicalPath
def diversityMitoOutputDir = diversityConfig.mito_output_dir ?: 'mito/diversity'
def diversityMitoOutputDirAbs = new File(outputDir, diversityMitoOutputDir).canonicalPath
def diversityMitoInputPath = diversityConfig.mito_input ? resolveOptionalPath(diversityConfig.mito_input, configRoot) : new File(outputDir, 'mito/ASVs/ASV_target.mito.tsv').canonicalPath
def diversitySampleCol = diversityConfig.sample_col ?: metadataPlotsSampleCol
def diversityGroupCol = diversityConfig.group_col ?: (diversityConfig.group1_col ?: metadataPlotsTypeCol)
def diversityColorCol = diversityConfig.color_col ?: 'Color'
def diversitySecondaryCol = diversityConfig.group2_col ?: (diversityConfig.secondary_col ?: 'Month')
def diversityExcludeGroupsRaw = diversityConfig.exclude_groups
List<String> diversityExcludeGroups = []
if( diversityExcludeGroupsRaw instanceof List ) {
    diversityExcludeGroups = diversityExcludeGroupsRaw.collect { it.toString().trim() }.findAll { it }
} else if( diversityExcludeGroupsRaw ) {
    diversityExcludeGroups = diversityExcludeGroupsRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def diversityGroupOrderRaw = diversityConfig.group_order ?: metadataPlotsGroupOrder
List<String> diversityGroupOrder = []
if( diversityGroupOrderRaw instanceof List ) {
    diversityGroupOrder = diversityGroupOrderRaw.collect { it.toString().trim() }.findAll { it }
} else if( diversityGroupOrderRaw ) {
    diversityGroupOrder = diversityGroupOrderRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
boolean diversityRunMito = diversityConfig.containsKey('run_mito') ? (diversityConfig.run_mito as boolean) : true
def diversityUmapNeighbors = diversityConfig.umap_neighbors ? (diversityConfig.umap_neighbors as int) : 30
def diversityUmapMinDist = diversityConfig.umap_min_dist != null ? (diversityConfig.umap_min_dist as double) : 0.01d
def diversityPermutations = diversityConfig.permanova_perms ? (diversityConfig.permanova_perms as int) : 999
def diversityRandomState = diversityConfig.random_state ? (diversityConfig.random_state as int) : 42
def diversityBlockCol = diversityConfig.block_col ? diversityConfig.block_col.toString().trim() : ''
boolean diversityVerbose = diversityConfig.containsKey('verbose') ? (diversityConfig.verbose as boolean) : true

def indicspeciesConfig = config.indicspecies ?: [:]
boolean indicspeciesRequested = indicspeciesConfig.containsKey('enabled') ? (indicspeciesConfig.enabled as boolean) : false
if( indicspeciesRequested && !metadataPlotsEnabled ) {
    exit 1, "indicspecies.enabled requires metadata_plots.enabled to be true"
}
def indicspeciesGroupColsRaw = indicspeciesConfig.group_cols
List<String> indicspeciesGroupCols = []
if( indicspeciesGroupColsRaw instanceof List ) {
    indicspeciesGroupCols = indicspeciesGroupColsRaw.collect { it.toString().trim() }.findAll { it }
} else if( indicspeciesGroupColsRaw ) {
    indicspeciesGroupCols = indicspeciesGroupColsRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
} else {
    indicspeciesGroupCols = ['Depth', 'Month']
}
if( indicspeciesRequested && indicspeciesGroupCols.size() < 2 ) {
    exit 1, "indicspecies.group_cols must contain at least two groups when indicspecies.enabled is true"
}
boolean indicspeciesEnabled = indicspeciesRequested
def indicspeciesSampleCol = indicspeciesConfig.sample_col ?: metadataPlotsSampleCol
def indicspeciesPerms = indicspeciesConfig.perms ? (indicspeciesConfig.perms as int) : 999
def indicspeciesMinN = indicspeciesConfig.min_n ? (indicspeciesConfig.min_n as int) : 2
def indicspeciesBlockCol = indicspeciesConfig.block_col ? indicspeciesConfig.block_col.toString().trim() : ''
def indicspeciesGroup1 = indicspeciesGroupCols[0]
def indicspeciesGroup2 = indicspeciesGroupCols[1]
def indicspeciesOutputDirAbs = new File(outputDir, "indicspecies").canonicalPath
boolean indicspeciesPlotEnabled = indicspeciesConfig.containsKey('plot_enabled') ? (indicspeciesConfig.plot_enabled as boolean) : true
def indicspeciesPlotPairsMode = indicspeciesConfig.plot_pairs_mode ?: 'all'
def indicspeciesPlotOutputDir = indicspeciesConfig.plot_output_dir ?: 'indicspecies/plots'
def indicspeciesPlotOutputDirAbs = new File(outputDir, indicspeciesPlotOutputDir).canonicalPath
def indicspeciesPlotVennPath = indicspeciesConfig.venn ? resolveOptionalPath(indicspeciesConfig.venn, configRoot) : null
def indicspeciesPlotTaxonomyPath = indicspeciesConfig.taxonomy ? resolveOptionalPath(indicspeciesConfig.taxonomy, configRoot) : new File(outputDir, 'taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv').canonicalPath
def indicspeciesColorCol = indicspeciesConfig.color_col ?: metadataPlotsColorCol
def indicspeciesGroup1Palette = indicspeciesConfig.group1_palette ?: ''
def indicspeciesGroup2Palette = indicspeciesConfig.group2_palette ?: 'Non-Cancer=#FFFFFF,Cancer=#A50026,Cancer+Non-Cancer=#000000,not_indicator=#D3D3D3'
def indicspeciesGroup1OrderRaw = indicspeciesConfig.group1_order ?: metadataPlotsGroupOrder
List<String> indicspeciesGroup1Order = []
if( indicspeciesGroup1OrderRaw instanceof List ) {
    indicspeciesGroup1Order = indicspeciesGroup1OrderRaw.collect { it.toString().trim() }.findAll { it }
} else if( indicspeciesGroup1OrderRaw ) {
    indicspeciesGroup1Order = indicspeciesGroup1OrderRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def indicspeciesGroup2OrderRaw = indicspeciesConfig.group2_order
List<String> indicspeciesGroup2Order = []
if( indicspeciesGroup2OrderRaw instanceof List ) {
    indicspeciesGroup2Order = indicspeciesGroup2OrderRaw.collect { it.toString().trim() }.findAll { it }
} else if( indicspeciesGroup2OrderRaw ) {
    indicspeciesGroup2Order = indicspeciesGroup2OrderRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def indicspeciesFocusGroup1Label = indicspeciesConfig.focus_group1_label ? indicspeciesConfig.focus_group1_label.toString().trim() : ''
def indicspeciesFocusGroup2Label = indicspeciesConfig.focus_group2_label ? indicspeciesConfig.focus_group2_label.toString().trim() : ''
boolean indicspeciesLabelFocusedAsvs = indicspeciesConfig.containsKey('label_focused_asvs') ? (indicspeciesConfig.label_focused_asvs as boolean) : false

def clustermapsConfig = config.clustermaps ?: [:]
boolean clustermapsRequested = clustermapsConfig.containsKey('enabled') ? (clustermapsConfig.enabled as boolean) : false
if( clustermapsRequested && !metadataPlotsEnabled ) {
    exit 1, "clustermaps.enabled requires metadata_plots.enabled to be true"
}
boolean clustermapsEnabled = clustermapsRequested
def clustermapsOutputDir = clustermapsConfig.output_dir ?: 'clustermaps'
def clustermapsOutputDirAbs = new File(outputDir, clustermapsOutputDir).canonicalPath
def clustermapsMitoOutputDir = clustermapsConfig.mito_output_dir ?: 'mito/clustermaps'
def clustermapsMitoOutputDirAbs = new File(outputDir, clustermapsMitoOutputDir).canonicalPath
def clustermapsMitoInputPath = clustermapsConfig.mito_input ? resolveOptionalPath(clustermapsConfig.mito_input, configRoot) : new File(outputDir, 'mito/ASVs/ASV_target.mito.tsv').canonicalPath
def clustermapsIsaFile = clustermapsConfig.isa_file ? resolveOptionalPath(clustermapsConfig.isa_file, configRoot) : null
def clustermapsSampleCol = clustermapsConfig.sample_col ?: 'sample'
def clustermapsSampleCodeCol = clustermapsConfig.sample_code_col ?: 'sample_code'
def clustermapsAsvIdCol = clustermapsConfig.asv_id_col ?: 'ASV_ID'
def clustermapsGroup1Col = clustermapsConfig.group1_col ?: 'type_group'
def clustermapsGroup2Col = clustermapsConfig.group2_col ?: 'status'
def clustermapsGroup3Col = clustermapsConfig.containsKey('group3_col') ? (clustermapsConfig.group3_col ?: '') : 'kit'
def clustermapsGroup1OrderRaw = clustermapsConfig.group1_order ?: (clustermapsConfig.type_order ?: metadataPlotsGroupOrder)
List<String> clustermapsGroup1Order = []
if( clustermapsGroup1OrderRaw instanceof List ) {
    clustermapsGroup1Order = clustermapsGroup1OrderRaw.collect { it.toString().trim() }.findAll { it }
} else if( clustermapsGroup1OrderRaw ) {
    clustermapsGroup1Order = clustermapsGroup1OrderRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def clustermapsExcludeGroup1 = clustermapsConfig.exclude_group1 ?: (clustermapsConfig.exclude_types ?: '')
def clustermapsGroup1Palette = clustermapsConfig.group1_palette ?: (clustermapsConfig.type_palette ?: '')
def clustermapsGroup2Palette = clustermapsConfig.group2_palette ?: (clustermapsConfig.status_palette ?: 'Non-Cancer=#FFFFFF,Cancer=#A50026,Cancer+Non-Cancer=#000000,not_indicator=#D3D3D3')
def clustermapsGroup3Palette = clustermapsConfig.group3_palette ?: (clustermapsConfig.kit_palette ?: '')
def clustermapsRanks = clustermapsConfig.ranks ?: 'Phylum,Class,Order,Family,Genus,Species,ASV_ID'
def clustermapsTopN = clustermapsConfig.topN ?: 'Phylum=30,Class=30,Order=30,Family=30,Genus=30,Species=30,ASV_ID=6000'
def clustermapsCountCol = clustermapsConfig.count_col ?: 'corr_count'
def clustermapsIsaMinStat = clustermapsConfig.isa_min_stat != null ? (clustermapsConfig.isa_min_stat as double) : 0.6d
def clustermapsIsaSignificanceCols = clustermapsConfig.isa_significance_cols ?: ''
def clustermapsIsaStatCols = clustermapsConfig.isa_stat_cols ?: ''
def clustermapsMitoSampleMode = clustermapsConfig.mito_sample_mode ?: 'auto'
boolean clustermapsRunMito = clustermapsConfig.containsKey('run_mito') ? (clustermapsConfig.run_mito as boolean) : true

def spieceasiConfig = config.spieceasi ?: [:]
boolean spieceasiRequested = spieceasiConfig.containsKey('enabled') ? (spieceasiConfig.enabled as boolean) : false
if( spieceasiRequested && !metadataPlotsEnabled ) {
    exit 1, "spieceasi.enabled requires metadata_plots.enabled to be true"
}
boolean spieceasiEnabled = spieceasiRequested
def spieceasiOutputDir = spieceasiConfig.output_dir ?: 'spieceasi'
def spieceasiOutputDirAbs = new File(outputDir, spieceasiOutputDir).canonicalPath
def spieceasiPrefix = spieceasiConfig.prefix ?: 'spieceasi'
boolean spieceasiTranspose = spieceasiConfig.containsKey('transpose') ? (spieceasiConfig.transpose as boolean) : true
def spieceasiMinRelAbund = spieceasiConfig.min_rel_abund != null ? (spieceasiConfig.min_rel_abund as double) : 0d
def spieceasiMinPrevalence = spieceasiConfig.min_prevalence != null ? (spieceasiConfig.min_prevalence as double) : 0d
boolean spieceasiRemoveZeroVar = spieceasiConfig.containsKey('remove_zero_var') ? (spieceasiConfig.remove_zero_var as boolean) : true
def spieceasiMethod = spieceasiConfig.method ?: 'glasso'
def spieceasiLambdaMinRatio = spieceasiConfig.lambda_min_ratio != null ? (spieceasiConfig.lambda_min_ratio as double) : 1e-2d
def spieceasiNlambda = spieceasiConfig.nlambda ? (spieceasiConfig.nlambda as int) : 20
def spieceasiRepNum = spieceasiConfig.rep_num ? (spieceasiConfig.rep_num as int) : 50
def spieceasiThresh = spieceasiConfig.thresh != null ? (spieceasiConfig.thresh as double) : 0.1d
def spieceasiNcores = spieceasiConfig.ncores ? (spieceasiConfig.ncores as int) : pipelineThreads
def spieceasiSeed = spieceasiConfig.seed ? (spieceasiConfig.seed as int) : 10010
def spieceasiEdgeThreshold = spieceasiConfig.edge_threshold != null ? (spieceasiConfig.edge_threshold as double) : 0.1d
boolean spieceasiKeepNegative = spieceasiConfig.containsKey('keep_negative') ? (spieceasiConfig.keep_negative as boolean) : true
def spieceasiLayoutIters = spieceasiConfig.layout_iters ? (spieceasiConfig.layout_iters as int) : 1000
boolean spieceasiForceFilter = spieceasiConfig.containsKey('force_filter') ? (spieceasiConfig.force_filter as boolean) : false
boolean spieceasiForceSpieceasi = spieceasiConfig.containsKey('force_spieceasi') ? (spieceasiConfig.force_spieceasi as boolean) : false
boolean spieceasiForceGraphs = spieceasiConfig.containsKey('force_graphs') ? (spieceasiConfig.force_graphs as boolean) : true
boolean networkRequested = spieceasiConfig.containsKey('network_enabled') ? (spieceasiConfig.network_enabled as boolean) : false
if( networkRequested && !indicspeciesEnabled ) {
    exit 1, "spieceasi.network_enabled requires indicspecies.enabled to be true"
}
boolean networkEnabled = networkRequested && indicspeciesEnabled
def networkGraphAllPath = spieceasiConfig.graph_pos_all ? resolveOptionalPath(spieceasiConfig.graph_pos_all, configRoot) : new File(spieceasiOutputDirAbs, "${spieceasiPrefix}_network_pos_all.graphml").canonicalPath
def networkGraphThrPath = spieceasiConfig.graph_pos_sub ? resolveOptionalPath(spieceasiConfig.graph_pos_sub, configRoot) : new File(spieceasiOutputDirAbs, "${spieceasiPrefix}_network_pos_thr.graphml").canonicalPath
def networkNodeFeaturesPath = spieceasiConfig.node_features ? resolveOptionalPath(spieceasiConfig.node_features, configRoot) : new File(spieceasiOutputDirAbs, "${spieceasiPrefix}_node_features.csv").canonicalPath
if( networkEnabled && !spieceasiEnabled ) {
    [networkGraphAllPath, networkGraphThrPath, networkNodeFeaturesPath].each { p ->
        if( !new File(p).exists() ) {
            exit 1, "spieceasi.network_enabled is true while spieceasi.enabled is false, but required cached file is missing: ${p}"
        }
    }
}
def networkModesRaw = spieceasiConfig.network_modes
List<String> networkModes = []
if( networkModesRaw instanceof List ) {
    networkModes = networkModesRaw.collect { it.toString().trim() }.findAll { it }
} else if( networkModesRaw ) {
    networkModes = networkModesRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
if( networkModes.isEmpty() ) {
    networkModes = ['all']
}
def networkLayoutSeed = spieceasiConfig.layout_seed ? (spieceasiConfig.layout_seed as int) : 42
def networkLayoutScale = spieceasiConfig.layout_scale != null ? (spieceasiConfig.layout_scale as double) : 3.0d
def networkDegreeScale = spieceasiConfig.degree_scale != null ? (spieceasiConfig.degree_scale as double) : 80.0d
def networkEdgeWidthScale = spieceasiConfig.edge_width_scale != null ? (spieceasiConfig.edge_width_scale as double) : 5.0d
def networkIsaScale = spieceasiConfig.isa_scale != null ? (spieceasiConfig.isa_scale as double) : 700.0d
boolean networkModuleBestOnly = spieceasiConfig.containsKey('module_best_only') ? (spieceasiConfig.module_best_only as boolean) : true
def networkModuleBestMinSize = spieceasiConfig.module_best_min_size ? (spieceasiConfig.module_best_min_size as int) : 5
def networkModuleBestMinStability = spieceasiConfig.module_best_min_stability != null ? (spieceasiConfig.module_best_min_stability as double) : 0.7d
boolean networkModuleIsaOnly = spieceasiConfig.containsKey('module_isa_only') ? (spieceasiConfig.module_isa_only as boolean) : false
boolean networkModuleColorByIsa = spieceasiConfig.containsKey('module_color_by_isa') ? (spieceasiConfig.module_color_by_isa as boolean) : false
def networkModuleIsaSource = spieceasiConfig.module_isa_source ? spieceasiConfig.module_isa_source.toString().trim().toLowerCase() : 'group1'
if( !['group1','group2'].contains(networkModuleIsaSource) ) {
    networkModuleIsaSource = 'group1'
}
def networkModuleIsaMinStat = spieceasiConfig.module_isa_min_stat != null ? (spieceasiConfig.module_isa_min_stat as double) : 0.25d
def networkModuleIsaMaxQ = spieceasiConfig.module_isa_max_q != null ? (spieceasiConfig.module_isa_max_q as double) : 0.05d
def networkMetadataPath = spieceasiConfig.metadata ? resolveOptionalPath(spieceasiConfig.metadata, configRoot) : metadataPlotsMetadataPath
def networkColorCol = spieceasiConfig.color_col ?: indicspeciesColorCol
def networkGroup1Palette = spieceasiConfig.group1_palette ?: indicspeciesGroup1Palette
def networkGroup2Palette = spieceasiConfig.group2_palette ?: indicspeciesGroup2Palette
def networkGroup1OrderRaw = spieceasiConfig.group1_order ?: indicspeciesGroup1Order
List<String> networkGroup1Order = []
if( networkGroup1OrderRaw instanceof List ) {
    networkGroup1Order = networkGroup1OrderRaw.collect { it.toString().trim() }.findAll { it }
} else if( networkGroup1OrderRaw ) {
    networkGroup1Order = networkGroup1OrderRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def networkGroup2OrderRaw = spieceasiConfig.group2_order ?: indicspeciesGroup2Order
List<String> networkGroup2Order = []
if( networkGroup2OrderRaw instanceof List ) {
    networkGroup2Order = networkGroup2OrderRaw.collect { it.toString().trim() }.findAll { it }
} else if( networkGroup2OrderRaw ) {
    networkGroup2Order = networkGroup2OrderRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def networkFocusGroup1Label = spieceasiConfig.focus_group1_label ? spieceasiConfig.focus_group1_label.toString().trim() : indicspeciesFocusGroup1Label
def networkFocusGroup2Label = spieceasiConfig.focus_group2_label ? spieceasiConfig.focus_group2_label.toString().trim() : indicspeciesFocusGroup2Label
boolean networkModulesEnabled = networkEnabled && (spieceasiConfig.containsKey('modules_enabled') ? (spieceasiConfig.modules_enabled as boolean) : false)
def networkModuleMethodsRaw = spieceasiConfig.module_methods ?: 'leiden,louvain'
List<String> networkModuleMethods = []
if( networkModuleMethodsRaw instanceof List ) {
    networkModuleMethods = networkModuleMethodsRaw.collect { it.toString().trim().toLowerCase() }.findAll { it }
} else if( networkModuleMethodsRaw ) {
    networkModuleMethods = networkModuleMethodsRaw.toString().split(/[,|]/).collect { it.trim().toLowerCase() }.findAll { it }
}
if( networkModuleMethods.isEmpty() ) {
    networkModuleMethods = ['leiden','louvain']
}
def networkModulePrimaryMethod = spieceasiConfig.module_primary_method ? spieceasiConfig.module_primary_method.toString().trim().toLowerCase() : networkModuleMethods[0]
if( !networkModuleMethods.contains(networkModulePrimaryMethod) ) {
    networkModulePrimaryMethod = networkModuleMethods[0]
}
def networkModuleResolutionsRaw = spieceasiConfig.module_resolutions ?: '0.5,1.0,1.5'
List<String> networkModuleResolutions = []
if( networkModuleResolutionsRaw instanceof List ) {
    networkModuleResolutions = networkModuleResolutionsRaw.collect { it.toString().trim() }.findAll { it }
} else if( networkModuleResolutionsRaw ) {
    networkModuleResolutions = networkModuleResolutionsRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
if( networkModuleResolutions.isEmpty() ) {
    networkModuleResolutions = ['1.0']
}
def networkModuleReps = spieceasiConfig.module_reps ? (spieceasiConfig.module_reps as int) : 25
def networkModuleConsensusThreshold = spieceasiConfig.module_consensus_threshold != null ? (spieceasiConfig.module_consensus_threshold as double) : 0.8d
def networkModuleSeed = spieceasiConfig.module_seed ? (spieceasiConfig.module_seed as int) : networkLayoutSeed
def networkModulesSubPath = spieceasiConfig.modules_sub ? resolveOptionalPath(spieceasiConfig.modules_sub, configRoot) : new File(spieceasiOutputDirAbs, "${spieceasiPrefix}_modules_sub.tsv").canonicalPath
def networkModulesAllPath = spieceasiConfig.modules_all ? resolveOptionalPath(spieceasiConfig.modules_all, configRoot) : new File(spieceasiOutputDirAbs, "${spieceasiPrefix}_modules_all.tsv").canonicalPath

def masterSummaryConfig = config.master_summary ?: [:]
boolean masterSummaryEnabled = masterSummaryConfig.containsKey('enabled') ? (masterSummaryConfig.enabled as boolean) : false
if( masterSummaryEnabled && !metadataPlotsEnabled ) {
    exit 1, "master_summary.enabled requires metadata_plots.enabled to be true"
}
def masterSummaryOutputDir = masterSummaryConfig.output_dir ?: 'summary/tables'
def masterSummaryOutputDirAbs = resolveOutputRelative(masterSummaryOutputDir.toString(), outputDir)
def masterSummaryClustermapsDir = masterSummaryConfig.clustermaps_dir ?: 'clustermaps'
def masterSummaryClustermapsDirAbs = resolveOutputRelative(masterSummaryClustermapsDir.toString(), outputDir)
def masterSummaryIndicspeciesDir = masterSummaryConfig.indicspecies_dir ?: 'indicspecies'
def masterSummaryIndicspeciesDirAbs = resolveOutputRelative(masterSummaryIndicspeciesDir.toString(), outputDir)
def masterSummarySpieceasiDir = masterSummaryConfig.spieceasi_dir ?: 'spieceasi'
def masterSummarySpieceasiDirAbs = resolveOutputRelative(masterSummarySpieceasiDir.toString(), outputDir)
def masterSummaryWhitelistRaw = masterSummaryConfig.whitelist
List<String> masterSummaryWhitelist = []
if( masterSummaryWhitelistRaw instanceof List ) {
    masterSummaryWhitelist = masterSummaryWhitelistRaw.collect { it.toString().trim() }.findAll { it }
} else if( masterSummaryWhitelistRaw ) {
    masterSummaryWhitelist = masterSummaryWhitelistRaw.toString().split(/[,|]/).collect { it.trim() }.findAll { it }
}
def masterSummaryWhitelistCsv = masterSummaryWhitelist ? masterSummaryWhitelist.join(',') : ''
def masterSummaryMaxDirectCols = masterSummaryConfig.max_direct_cols ? (masterSummaryConfig.max_direct_cols as int) : 300

workflow {
    def biochemReady = Channel.value(true)
    if( biochemPreAsvEnabled ) {
        def b0 = BIOCHEM_MERGE()
        def b1 = BIOCHEM_DENSITY(b0.done)
        def b2 = BIOCHEM_STRAT_METRICS(b1.done)
        def b3 = BIOCHEM_CUSTOM_CLEAN(b2.done)
        def b4 = BIOCHEM_EIGENVECTORS(b3.done)
        def b5 = BIOCHEM_SELECTK(b4.done)
        def b6 = BIOCHEM_GMM(b5.done, b5.selected_k)
        def b7 = BIOCHEM_O2_SOFT(b6.done)
        def b8 = BIOCHEM_HYBRID(b7.done)
        def b9 = BIOCHEM_COMPARE(b8.done)
        def b10 = BIOCHEM_SPLIT_O2_BY_GMM(b9.done)
        def b11 = BIOCHEM_STRAT_ANOMALY(b10.done)
        def b12 = BIOCHEM_STATE_TRANSITIONS(b11.done)
        def b13 = BIOCHEM_SUCCESSION_GRAPH(b12.done)
        def b14 = BIOCHEM_FEATURE_ASSOC(b13.done)
        def b15 = BIOCHEM_EOF_PIPELINE(b14.done)
        def b16 = BIOCHEM_EOF_STATE_CLUSTER(b15.done)
        def b17 = BIOCHEM_EOF_MODE_PLOTS(b16.done)
        def b18 = BIOCHEM_WITHIN_GMM_HDBSCAN(b17.done)
        biochemReady = b18.done.collect().map { true }
    }
    def rawReadsForAsv = raw_reads
        .combine(biochemReady)
        .map { meta, r1, r2, _ready -> tuple(meta, r1, r2) }
    def fastp_result = FASTP_QC(rawReadsForAsv)
    def reads_after_qc = fastp_result.reads
    def reads_after_merge = MERGE_READS(reads_after_qc)
    def reads_after_filter = FILTER_READS(reads_after_merge)
    def reads_for_concat = reads_after_filter
    if( concatRelabelEnabled ) {
        def relabeled_stage = RELABEL_FILTERED(reads_after_filter)
        reads_for_concat = relabeled_stage.relabeled
    }
    def relabeled_fasta_files = reads_for_concat.map { parts -> parts[1] }

    def concat_for_derep = relabeled_fasta_files
        .collectFile(name: 'concat.fasta', storeDir: dirMap.concat, newLine: true)
    def concat_for_counts = relabeled_fasta_files
        .collectFile(name: 'concat_counts.fasta', storeDir: dirMap.concat, newLine: true)

    def derep_input = DEREPLICATE(concat_for_derep)
    def denoise_input = DENOISE(derep_input)
    def nochi_input = CHIMERA_CHECK(denoise_input)

    def count_matrix_stage = CREATE_COUNT_MATRIX(concat_for_counts, nochi_input)
    def count_matrix_channel = count_matrix_stage.count_matrix
    def asv_counts_for_sankey = count_matrix_channel.map { tuple -> tuple[0] }
    def filtered_stage = FILTER_TABLE(count_matrix_channel)
    def filtered_channel = filtered_stage.filtered
    def filtered_fasta_for_taxonomy = filtered_channel.map { tuple -> tuple[1] }
    def sina_stage = SINA_TRIM(filtered_fasta_for_taxonomy)
    def taxonomy_stage = TAXONOMY(sina_stage.trimmed_fasta)
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
def asvMetaForClustermaps = null
def asvMetaForBubbleUmap = null
def asvFinalForBatch = null
def asvFinalForCollectors = null
def asvFinalForSpieceasi = null
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
        asvMetaForClustermaps = metadata_stage.asv_meta_micro
        asvMetaForBubbleUmap = metadata_stage.asv_meta_micro
        asvFinalForBatch = metadata_stage.asv_final_micro
        asvFinalForCollectors = metadata_stage.asv_final_micro
        asvFinalForSpieceasi = metadata_stage.asv_final_micro
    }

    if( plotUpsetEnabled ) {
        PLOT_UPSET(metaMicroForCollectors)
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
        asvFinalForCollectors = batch_stage.asv_corrected_counts_int
        asvFinalForSpieceasi = batch_stage.asv_corrected_counts_int
        umapResultsForTrajectory = batch_stage.umap_results
        if( bubbleplotterEnabled || umapClusteringEnabled || clustermapsEnabled ) {
            def corrected_asv_meta_stage = ASV_META_FROM_CORRECTED(
                asvMetaForClustermaps,
                batch_stage.asv_corrected_counts_int
            )
            asvMetaForClustermaps = corrected_asv_meta_stage.asv_meta_corrected
            asvMetaForBubbleUmap = corrected_asv_meta_stage.asv_meta_corrected
        }
    }
    if( bubbleplotterEnabled ) {
        BUBBLEPLOTTER(asvMetaForBubbleUmap)
    }
    if( umapClusteringEnabled ) {
        UMAP_CLUSTERING(asvMetaForBubbleUmap)
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
    def diversity_stage = null
    if( diversityEnabled ) {
        diversity_stage = DIVERSITY_ANALYSIS(
            metaMicroForCollectors,
            asvFinalForCollectors
        )
    }
    def indicspecies_stage = null
    def indicspecies_plots_stage = null
    if( indicspeciesEnabled ) {
        indicspecies_stage = INDICSPECIES(
            metaMicroForCollectors,
            asvFinalForCollectors
        )
        if( indicspeciesPlotEnabled ) {
            indicspecies_plots_stage = INDICSPECIES_PLOTS(
                metaMicroForCollectors,
                indicspecies_stage.all_tables.collect()
            )
        }
    }
    if( clustermapsEnabled ) {
        CLUSTERMAPS(
            asvMetaForClustermaps,
            metaMicroForCollectors
        )
    }
    def spieceasi_stage = null
    def graphAllForNetwork = null
    def graphThrForNetwork = null
    def nodeFeaturesForNetwork = null
    def modulesSubForNetwork = null
    def modulesAllForNetwork = null
    if( spieceasiEnabled ) {
        spieceasi_stage = SPIECEASI(
            asvFinalForSpieceasi
        )
        graphAllForNetwork = spieceasi_stage.graph_all
        graphThrForNetwork = spieceasi_stage.graph_thr
        nodeFeaturesForNetwork = spieceasi_stage.node_features
    } else if( networkEnabled ) {
        graphAllForNetwork = Channel.value(file(networkGraphAllPath))
        graphThrForNetwork = Channel.value(file(networkGraphThrPath))
        nodeFeaturesForNetwork = Channel.value(file(networkNodeFeaturesPath))
    }
    if( networkEnabled ) {
        if( networkModulesEnabled ) {
            def network_modules_stage = NETWORK_MODULES(
                graphAllForNetwork,
                graphThrForNetwork
            )
            modulesSubForNetwork = network_modules_stage.modules_sub
            modulesAllForNetwork = network_modules_stage.modules_all
        } else {
            def modulesSubFile = new File(networkModulesSubPath)
            def modulesAllFile = new File(networkModulesAllPath)
            modulesSubForNetwork = modulesSubFile.exists() ? Channel.value(file(networkModulesSubPath)) : Channel.value(file(emptyModulesPath))
            modulesAllForNetwork = modulesAllFile.exists() ? Channel.value(file(networkModulesAllPath)) : Channel.value(file(emptyModulesPath))
        }
    }
    def graph_network_stage = null
    if( networkEnabled ) {
        graph_network_stage = GRAPH_NETWORK(
            graphAllForNetwork,
            graphThrForNetwork,
            nodeFeaturesForNetwork,
            asvFinalForSpieceasi,
            taxonomy_stage.taxonomy_table,
            indicspecies_stage.group1_summary,
            indicspecies_stage.group2_summary,
            modulesSubForNetwork,
            modulesAllForNetwork
        )
    }
    def sankey_stage = null
    if( sankeyEnabled ) {
        sankey_stage = SANKEY(
            general_stats_stage.fastq_stats,
            general_stats_stage.filtered_stats,
            asv_counts_for_sankey,
            filter_counts_stage.filtered_decon,
            filter_counts_stage.filtered_micro
        )
    }
    if( masterSummaryEnabled ) {
        if( !asvMetaForClustermaps || !asvFinalForSpieceasi ) {
            exit 1, "master_summary.enabled requires ASV_meta and ASV_final inputs from metadata/batch stages"
        }
        def masterSummaryNetworkDone = graph_network_stage ? graph_network_stage.done : Channel.value(file(emptyModulesPath))
        def masterSummarySankeyDone = sankey_stage ? sankey_stage.done : Channel.value(file(emptyModulesPath))
        MASTER_SUMMARY(
            asvMetaForClustermaps,
            asvFinalForSpieceasi,
            masterSummaryNetworkDone,
            masterSummarySankeyDone
        )
    }
}

process BIOCHEM_MERGE {
    cpus pipelineThreads
    conda "${biochemMergeCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    output:
    path("biochem_merge.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemProcessingDirAbs}"
python "${biochemMergeTablesScriptPath}" \\
  --table-a "${biochemTableAPath}" \\
  --table-b "${biochemTableBPath}" \\
  --outdir "${biochemProcessingDirAbs}"
[[ -f "${biochemMergedOxygenPath}" ]] || { echo "Missing ${biochemMergedOxygenPath}" >&2; exit 1; }
touch biochem_merge.done
"""
}

process BIOCHEM_DENSITY {
    cpus pipelineThreads
    conda "${biochemDensityCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_density.done"), emit: done

    script:
    """
set -euo pipefail
python "${biochemCalcDensityScriptPath}" \\
  --input "${biochemMergedOxygenPath}" \\
  --salinity-col Salinity \\
  --temperature-col Temperature \\
  --depth-col Depth \\
  --latitude-col Latitude \\
  --longitude-col Longitude \\
  --sigma0
[[ -f "${biochemDensityPath}" ]] || { echo "Missing ${biochemDensityPath}" >&2; exit 1; }
touch biochem_density.done
"""
}

process BIOCHEM_STRAT_METRICS {
    cpus pipelineThreads
    conda "${biochemStratMetricsCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_strat_metrics.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemStratMetricsDirAbs}"
python "${biochemStratMetricsScriptPath}" \\
  --input "${biochemDensityPath}" \\
  --output-dir "${biochemStratMetricsDirAbs}" \\
  --salinity-col Salinity \\
  --temperature-col Temperature \\
  --depth-col Depth \\
  --latitude-col Latitude \\
  --longitude-col Longitude \\
  --profile-cols Cruise \\
  --date-col Date \\
  --layer-split-mode mld125
[[ -f "${biochemStratMetricsDirAbs}/stratification_summary.tsv" ]] || { echo "Missing ${biochemStratMetricsDirAbs}/stratification_summary.tsv" >&2; exit 1; }
touch biochem_strat_metrics.done
"""
}

process BIOCHEM_CUSTOM_CLEAN {
    cpus pipelineThreads
    conda "${biochemCustomCleanCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_custom_clean.done"), emit: done
    path("biochem_density_cleaned.tsv"), emit: cleaned_density

    script:
    def biochemRenamePairs = biochemCleanRenameMap ? biochemCleanRenameMap.collect { k, v -> "${k}:${v}" }.join(',') : ''
    def biochemKeepArg = biochemCleanKeepCols && !biochemCleanKeepCols.isEmpty() ? """ --keep-cols "${biochemCleanKeepCols.join(',')}" """ : ''
    def biochemDropArg = biochemCleanDropCols && !biochemCleanDropCols.isEmpty() ? """ --drop-cols "${biochemCleanDropCols.join(',')}" """ : ''
    def biochemRenameArg = biochemRenamePairs ? """ --rename-map "${biochemRenamePairs}" """ : ''
    """
set -euo pipefail
python "${biochemCustomCleanerScriptPath}" --input "${biochemDensityPath}" --output "${biochemDensityCleanedPath}"${biochemKeepArg}${biochemDropArg}${biochemRenameArg}
[[ -f "${biochemDensityCleanedPath}" ]] || { echo "Missing ${biochemDensityCleanedPath}" >&2; exit 1; }
ln -sf "${biochemDensityCleanedPath}" biochem_density_cleaned.tsv
touch biochem_custom_clean.done
"""
}

process BIOCHEM_EIGENVECTORS {
    cpus pipelineThreads
    conda "${biochemEigenvectorsCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_eigenvectors.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemPcaDirAbs}"
python "${biochemEigenvectorsScriptPath}" \\
  --input "${biochemDensityCleanedPath}" \\
  --outdir "${biochemPcaDirAbs}" \\
  --feature-cols "${biochemFeatureCols}" \\
  --pc-selection \\
  --anchor-depths
[[ -f "${biochemPcaDirAbs}/tables/eigenvectors_scores.csv" ]] || { echo "Missing ${biochemPcaDirAbs}/tables/eigenvectors_scores.csv" >&2; exit 1; }
[[ -f "${biochemPcaDirAbs}/tables/pc_keep_decision.csv" ]] || { echo "Missing ${biochemPcaDirAbs}/tables/pc_keep_decision.csv" >&2; exit 1; }
touch biochem_eigenvectors.done
"""
}

process BIOCHEM_SELECTK {
    cpus pipelineThreads
    conda "${biochemSelectkCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_selectk.done"), emit: done
    path("selected_k.txt"), emit: selected_k

    script:
    """
set -euo pipefail
mkdir -p "${biochemSelectkDirAbs}"
python "${biochemSelectkScriptPath}" \\
  --eigenvectors "${biochemPcaDirAbs}/tables/eigenvectors_scores.csv" \\
  --pc-keep "${biochemPcaDirAbs}/tables/pc_keep_decision.csv" \\
  --outdir "${biochemSelectkDirAbs}" \\
  --sep "," \\
  --stability-block-col Cruise \\
  --min-cluster-frac 0.02
[[ -s "${biochemSelectkDirAbs}/SELECTED_K.txt" ]] || { echo "Missing ${biochemSelectkDirAbs}/SELECTED_K.txt" >&2; exit 1; }
ln -sf "${biochemSelectkDirAbs}/SELECTED_K.txt" selected_k.txt
touch biochem_selectk.done
"""
}

process BIOCHEM_GMM {
    cpus pipelineThreads
    conda "${biochemGmmCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)
    path(selected_k_file)

    output:
    path("biochem_gmm.done"), emit: done

    script:
    def biochemGmmKAutoFlag = biochemGmmKAuto ? '1' : '0'
    """
set -euo pipefail
mkdir -p "${biochemGmmDirAbs}"
gmm_k="${biochemGmmK}"
if [[ "${biochemGmmKAutoFlag}" == "1" ]]; then
  gmm_k="\$(tr -d '[:space:]' < "${selected_k_file}")"
  case "\$gmm_k" in
    ''|*[!0-9]*) echo "Invalid selected K: \$gmm_k" >&2; exit 1 ;;
  esac
fi
python "${biochemGmmScriptPath}" \\
  --eigenvectors "${biochemPcaDirAbs}/tables/eigenvectors_scores.csv" \\
  --pc-keep "${biochemPcaDirAbs}/tables/pc_keep_decision.csv" \\
  --outdir "${biochemGmmDirAbs}" \\
  --sep "," \\
  --pc-use-mode keep \\
  --standardize-pc-space \\
  --episodic-smoothing \\
  --random-state 42 \\
  --K "\$gmm_k" \\
  --matrix-cleaned "${biochemPcaDirAbs}/tables/matrix_cleaned_with_sparse.csv"
[[ -f "${biochemGmmDirAbs}/tables/compartments_assignments_smoothed.csv" ]] || { echo "Missing ${biochemGmmDirAbs}/tables/compartments_assignments_smoothed.csv" >&2; exit 1; }
touch biochem_gmm.done
"""
}

process BIOCHEM_O2_SOFT {
    cpus pipelineThreads
    conda "${biochemO2SoftCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_o2_soft.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemO2DirAbs}"
python "${biochemO2SoftScriptPath}" \\
  --input "${biochemPcaDirAbs}/tables/matrix_cleaned.csv" \\
  --outdir "${biochemO2DirAbs}" \\
  --o2-col Oxygen \\
  --T-oxic-dyso 90 \\
  --T-dyso-sub 20 \\
  --T-sub-anox 1 \\
  --episodic-smoothing \\
  --episodic-block-col Cruise \\
  --episodic-sort-cols Depth_anchored \\
  --episodic-sticky-prob 0.85 \\
  --episodic-apply-to all
[[ -f "${biochemO2DirAbs}/tables/o2_compartments_assignments_smoothed.csv" ]] || { echo "Missing ${biochemO2DirAbs}/tables/o2_compartments_assignments_smoothed.csv" >&2; exit 1; }
touch biochem_o2_soft.done
"""
}

process BIOCHEM_HYBRID {
    cpus pipelineThreads
    conda "${biochemHybridCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_hybrid.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemHybridDirAbs}"
python "${biochemHybridScriptPath}" \\
  --gmm-assign "${biochemGmmDirAbs}/tables/compartments_assignments_smoothed.csv" \\
  --o2-assign "${biochemO2DirAbs}/tables/o2_compartments_assignments_smoothed.csv" \\
  --outdir "${biochemHybridDirAbs}" \\
  --join-key cruise_year_month_depth \\
  --make-plot \\
  --cmap rainbow \\
  --make-umap \\
  --umap-n-neighbors 15 \\
  --umap-min-dist 0.1 \\
  --umap-seed 42
[[ -f "${biochemHybridDirAbs}/tables/cruise_composition_hybrid.csv" ]] || { echo "Missing ${biochemHybridDirAbs}/tables/cruise_composition_hybrid.csv" >&2; exit 1; }
touch biochem_hybrid.done
"""
}

process BIOCHEM_COMPARE {
    cpus pipelineThreads
    conda "${biochemCompareCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_compare.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemCompareDirAbs}"
python "${biochemCompareScriptPath}" \\
  --matrix-cleaned "${biochemPcaDirAbs}/tables/matrix_cleaned_with_sparse.csv" \\
  --eigenvectors "${biochemPcaDirAbs}/tables/eigenvectors_scores.csv" \\
  --assignments "${biochemGmmDirAbs}/tables/compartments_assignments_smoothed.csv" \\
  --o2-assignments "${biochemO2DirAbs}/tables/o2_compartments_assignments_smoothed.csv" \\
  --o2-compartment-col compartment_name \\
  --outdir "${biochemCompareDirAbs}" \\
  --sep-matrix "," \\
  --sep-eig "," \\
  --sep-assign "," \\
  --sep-o2-assign "," \\
  --pca-tables-dir "${biochemPcaDirAbs}/tables" \\
  --key-mode composite \\
  --key-cols "Cruise,Year,Month,Day,Depth" \\
  --pc-cols "PC1,PC2"
[[ -f "${biochemCompareDirAbs}/tables/umap_embedding.csv" ]] || { echo "Missing ${biochemCompareDirAbs}/tables/umap_embedding.csv" >&2; exit 1; }
touch biochem_compare.done
"""
}

process BIOCHEM_SPLIT_O2_BY_GMM {
    cpus pipelineThreads
    conda "${biochemSplitCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_split.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemSplitDirAbs}"
python "${biochemSplitScriptPath}" \\
  --matrix-cleaned "${biochemPcaDirAbs}/tables/matrix_cleaned_with_sparse.csv" \\
  --eigenvectors "${biochemPcaDirAbs}/tables/eigenvectors_scores.csv" \\
  --assignments "${biochemGmmDirAbs}/tables/compartments_assignments_smoothed.csv" \\
  --o2-assignments "${biochemO2DirAbs}/tables/o2_compartments_assignments_smoothed.csv" \\
  --o2-compartment-col compartment_name \\
  --outdir "${biochemSplitDirAbs}" \\
  --sep-matrix "," \\
  --sep-eig "," \\
  --sep-assign "," \\
  --sep-o2-assign "," \\
  --key-mode composite \\
  --key-cols "Cruise,Year,Month,Day,Depth" \\
  --pc-cols "PC1,PC2" \\
  --plots \\
  --plot-formats "pdf,png,svg" \\
  --umap-embedding "${biochemCompareDirAbs}/tables/umap_embedding.csv" \\
  --reassign \\
  --borderline-mode other_or_low_conf \\
  --borderline-max-prob 0.70 \\
  --core-min-prob 0.90 \\
  --reassign-radius-quantile 0.95 \\
  --reassign-min-core-n 30 \\
  --min-subcluster-size 20
[[ -f "${biochemSplitDirAbs}/tables/merged_o2_split_by_gmm.csv" ]] || { echo "Missing ${biochemSplitDirAbs}/tables/merged_o2_split_by_gmm.csv" >&2; exit 1; }
touch biochem_split.done
"""
}

process BIOCHEM_STRAT_ANOMALY {
    cpus pipelineThreads
    conda "${biochemStratAnomalyCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_strat_anomaly.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemStratIndexDirAbs}"
python "${biochemStratAnomalyScriptPath}" \\
  --input "${biochemPcaDirAbs}/tables/matrix_cleaned.csv" \\
  --sample-id-col cruise_year_month_depth \\
  --date-col date \\
  --month-col Month \\
  --year-col Year \\
  --depth-col Depth \\
  --output-dir "${biochemStratIndexDirAbs}" \\
  --pea-metrics "${biochemStratMetricsDirAbs}/stratification_summary.tsv"
[[ -f "${biochemStratIndexDirAbs}/stratification_timeseries.tsv" ]] || { echo "Missing ${biochemStratIndexDirAbs}/stratification_timeseries.tsv" >&2; exit 1; }
touch biochem_strat_anomaly.done
"""
}

process BIOCHEM_STATE_TRANSITIONS {
    cpus pipelineThreads
    conda "${biochemStateTransitionsCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_state_transitions.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemStateTransitionsDirAbs}"
python "${biochemStateTransitionScriptPath}" \\
  --o2 "${biochemHybridDirAbs}/tables/cruise_composition_o2.csv" \\
  --gmm "${biochemHybridDirAbs}/tables/cruise_composition_gmm.csv" \\
  --hybrid "${biochemHybridDirAbs}/tables/cruise_composition_hybrid.csv" \\
  --outdir "${biochemStateTransitionsDirAbs}" \\
  --changepoint-metric braycurtis \\
  --changepoint-threshold 0.35 \\
  --coupling \\
  --coupling-method spearman \\
  --coupling-cluster-threshold 0.30 \\
  --coupling-edge-threshold 0.60 \\
  --strat-timeseries "${biochemStratIndexDirAbs}/stratification_timeseries.tsv" \\
  --eof-states "${biochemStratIndexDirAbs}/stratification_timeseries.tsv" \\
  --eof-state-col anomaly_type
touch biochem_state_transitions.done
"""
}

process BIOCHEM_SUCCESSION_GRAPH {
    cpus pipelineThreads
    conda "${biochemSuccessionCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_succession.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemSuccessionDirAbs}"
python "${biochemSuccessionScriptPath}" \\
  --o2 "${biochemHybridDirAbs}/tables/cruise_composition_o2.csv" \\
  --gmm "${biochemHybridDirAbs}/tables/cruise_composition_gmm.csv" \\
  --hybrid "${biochemHybridDirAbs}/tables/cruise_composition_hybrid.csv" \\
  --outdir "${biochemSuccessionDirAbs}" \\
  --make-plots \\
  --top-n 1 \\
  --no-keep-self \\
  --min-prob 0.1
touch biochem_succession.done
"""
}

process BIOCHEM_FEATURE_ASSOC {
    cpus pipelineThreads
    conda "${biochemFeatureAssocCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_feature_assoc.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemFeatureAssocDirAbs}"
python "${biochemFeatureAssocScriptPath}" \\
  --matrix-cleaned "${biochemPcaDirAbs}/tables/matrix_cleaned_with_sparse.csv" \\
  --assignments-gmm "${biochemGmmDirAbs}/tables/compartments_assignments_smoothed.csv" \\
  --assignments-o2 "${biochemO2DirAbs}/tables/o2_compartments_assignments_smoothed.csv" \\
  --assignments-hybrid "${biochemHybridDirAbs}/tables/compartments_assignments_hybrid.csv" \\
  --outdir "${biochemFeatureAssocDirAbs}" \\
  --sep-matrix "," \\
  --sep-assign "," \\
  --bootstrap-B 500 \\
  --top-n-each-side 8 \\
  --min-n-comp 20 \\
  --min-n-rest 50 \\
  --depth-adjust \\
  --hybrid-split-table "${biochemSplitDirAbs}/tables/merged_o2_split_by_gmm.csv"
touch biochem_feature_assoc.done
"""
}

process BIOCHEM_EOF_PIPELINE {
    cpus pipelineThreads
    conda "${biochemEofPipelineCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_eof_pipeline.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemEofPcaDirAbs}"
python "${biochemEofPipelineScriptPath}" \\
  --matrix-cleaned "${biochemPcaDirAbs}/tables/matrix_cleaned_with_sparse.csv" \\
  --core-loadings "${biochemPcaDirAbs}/tables/pca_loadings.csv" \\
  --outdir "${biochemEofPcaDirAbs}" \\
  --sep "," \\
  --pc-selection
[[ -f "${biochemEofPcaDirAbs}/tables/eof_eigenvectors_scores_by_cruise.csv" ]] || { echo "Missing ${biochemEofPcaDirAbs}/tables/eof_eigenvectors_scores_by_cruise.csv" >&2; exit 1; }
touch biochem_eof_pipeline.done
"""
}

process BIOCHEM_EOF_STATE_CLUSTER {
    cpus pipelineThreads
    conda "${biochemEofStateCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_eof_state_cluster.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemEofStatesDirAbs}"
python "${biochemEofStateScriptPath}" \\
  --scores "${biochemEofPcaDirAbs}/tables/eof_eigenvectors_scores_by_cruise.csv" \\
  --pcs "${biochemEofPcs}" \\
  --k auto \\
  --k-min 2 \\
  --k-max 10 \\
  --covariance-type full \\
  --standardize-pc-space \\
  --n-init 20 \\
  --max-iter 500 \\
  --cv-folds 5 \\
  --stability-R 200 \\
  --stability-block-col Cruise \\
  --stability-oob-min 10 \\
  --stability-min-ari 0.25 \\
  --min-cluster-frac 0.01 \\
  --select-by icl \\
  --select-delta 5 \\
  --sep "," \\
  --outdir "${biochemEofStatesDirAbs}" \\
  --sticky-smoothing \\
  --time-col date \\
  --sticky-prob 0.85 \\
  --apply-to all
touch biochem_eof_state_cluster.done
"""
}

process BIOCHEM_EOF_MODE_PLOTS {
    cpus pipelineThreads
    conda "${biochemEofModeCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_eof_mode_plots.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemEofPlotsDirAbs}"
python "${biochemEofModePlotScriptPath}" \\
  --loadings "${biochemEofPcaDirAbs}/tables/eof_pca_loadings.csv" \\
  --explained "${biochemEofPcaDirAbs}/tables/eof_pca_explained_variance.csv" \\
  --outdir "${biochemEofPlotsDirAbs}" \\
  --eofs "${biochemEofPcs}" \\
  --top-n 100 \\
  --sep ","
touch biochem_eof_mode_plots.done
"""
}

process BIOCHEM_WITHIN_GMM_HDBSCAN {
    cpus pipelineThreads
    conda "${biochemWithinGmmCondaEnvPath}"

    when:
    biochemPreAsvEnabled

    input:
    path(prev_done)

    output:
    path("biochem_within_gmm.done"), emit: done

    script:
    """
set -euo pipefail
mkdir -p "${biochemWithinGmmDirAbs}"
python "${biochemWithinGmmScriptPath}" \\
  --eigenvectors "${biochemPcaDirAbs}/tables/eigenvectors_scores.csv" \\
  --assignments "${biochemGmmDirAbs}/tables/compartments_assignments_smoothed.csv" \\
  --outdir "${biochemWithinGmmDirAbs}" \\
  --sep "," \\
  --pc-cols "PC1,PC2" \\
  --standardize-pc-space \\
  --hdbscan-min-cluster-size 10 \\
  --hdbscan-metric euclidean \\
  --min-rows-per-component 10 \\
  --high-conf-only \\
  --high-conf-maxprob 0.80 \\
  --strict-unique-ids
touch biochem_within_gmm.done
"""
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
    def allowStagger = mergeAllowStagger ? '--fastq_allowmergestagger' : ''
    if( meta.paired && r2 ) {
        """
vsearch --fastq_mergepairs "${r1}" \\
        --reverse "${r2}" \\
        --fastqout merged.fastq \\
        --fastq_maxdiffs ${mergeMaxDiffs} \\
        --fastq_minovlen ${mergeMinOverlap} \\
        --fastq_truncqual ${mergeTruncQuality} \\
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

process RELABEL_FILTERED {
    tag { meta.sample_id }
    cpus 1
    conda "${condaEnvPath}"
    publishDir dirMap.concat, mode: 'copy', pattern: '*.fasta', saveAs: { filename ->
        filename == 'filtered_relabel.fasta' ? "${meta.sample_id}.filtered.relabel.fasta" : filename
    }

    input:
    tuple val(meta), path(filtered_fasta)

    output:
    tuple val(meta), path("filtered_relabel.fasta"), emit: relabeled

    script:
    def labelSep = concatLabelSep
    """
awk -v pref="${meta.sample_id}" -v sep="${labelSep}" '{
  if (\$0 ~ /^>/) {
    sub(/^>[^:]*:/, ">" pref sep, \$0)
    if (\$0 !~ "^>" pref sep) \$0 = ">" pref sep substr(\$0, 2)
  }
  print
}' "${filtered_fasta}" > filtered_relabel.fasta
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
    path(filtered_fasta)

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
    def filteredFastaPath = filtered_fasta.toString().trim()
    """
set -euo pipefail
rm -rf "${mitoChunkDirPath}"
mkdir -p "${mitoChunkDirPath}"
FILTERED_FASTA=\$(realpath "${filteredFastaPath}")

seqkit split -s ${mitoChunkSize} -O "${mitoChunkDirPath}" "\${FILTERED_FASTA}"

python "${mitomasterScriptPath}" \\
  --data-dir "${mitoChunkDirPath}" \\
  --glob-pattern "*.fa*" \\
  --recursive \\
  --output-file mitomaster_output.tsv \\
  --max-workers ${mitomasterWorkers} \\
  --timeout ${mitomasterTimeout} \\
  --retries ${mitomasterRetries} \\
  --header-mode "${mitomasterHeaderMode}" \\
  --overwrite

blastn -query "\${FILTERED_FASTA}" \\
  -db "${mitoBlastDbPath}" \\
  -outfmt "6 qseqid sseqid pident length qlen mismatch gapopen qstart qend sstart send evalue bitscore" \\
  -out mito_ncbi.blast6.tsv \\
  -num_threads ${task.cpus}

blastn -query "\${FILTERED_FASTA}" \\
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

    output:
    path("sankey.done"), emit: done

    script:
    def keepTypesArg = sankeyKeepTypes && !sankeyKeepTypes.isEmpty() ? "  --keep-types \"${sankeyKeepTypes.join(',')}\" \\\n" : ''
    def labeledFlag = sankeyMakeLabeled ? "  --make-labeled \\\n" : ''
    def unlabeledFlag = sankeyMakeUnlabeled ? "  --make-unlabeled \\\n" : ''
    """
set -euo pipefail

python3 "${sankeyScriptPath}" \\
  --data-dir "${outputDir}" \\
  --sub-dir "${sankeySubDir}" \\
  --metadata "${sankeyMetadataPath}" \\
  --sample-manifest "${manifestPath}" \\
  --samp-col "${sankeySampCol}" \\
  --group1-col "${sankeyGroupCol}" \\
  --color-col "${sankeyColorCol}" \\
${keepTypesArg}  --fastq-stats stats/"${fastq_stats}" \\
  --filtered-stats stats/"${filtered_stats}" \\
  --asv-raw ASVs/"${asv_counts}" \\
  --asv-decon ASVs/"${asv_decon_counts}" \\
  --asv-micro ASVs/"${asv_micro_counts}" \\
  --title "${sankeyTitle}" \\
  --arrangement "${sankeyArrangement}" \\
  --output-prefix "${sankeyOutputPrefix}" \\
${labeledFlag}${unlabeledFlag}  --verbose

touch sankey.done
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
    def includeRankAppend = metadataIncludeRank && !metadataIncludeRank.isEmpty() ?
        metadataIncludeRank.collect { "cmd+=( --include-rank \"${it}\" )" }.join('\n') : ''
    def metadataBiochemTable = metadataPlotsBiochemAssignmentsPath ?: (biochemPreAsvEnabled ? "${biochemSplitDirAbs}/tables/merged_o2_split_by_gmm.csv" : '')
    def metadataBiochemIncludeCsv = metadataPlotsBiochemIncludeCols && !metadataPlotsBiochemIncludeCols.isEmpty() ? metadataPlotsBiochemIncludeCols.join(',') : ''
    def metadataBiochemMetaJoinCsv = metadataPlotsBiochemMetaJoinCols && !metadataPlotsBiochemMetaJoinCols.isEmpty() ? metadataPlotsBiochemMetaJoinCols.join(',') : ''
    def metadataBiochemJoinCsv = metadataPlotsBiochemJoinCols && !metadataPlotsBiochemJoinCols.isEmpty() ? metadataPlotsBiochemJoinCols.join(',') : ''
    def metadataStratTable = metadataPlotsStratificationTimeseriesPath ?: (biochemPreAsvEnabled ? "${biochemStratIndexDirAbs}/stratification_timeseries.tsv" : '')
    def metadataStratIncludeCsv = metadataPlotsStratIncludeCols && !metadataPlotsStratIncludeCols.isEmpty() ? metadataPlotsStratIncludeCols.join(',') : ''
    def metadataKeepTypesCsv = metadataKeepTypes && !metadataKeepTypes.isEmpty() ? metadataKeepTypes.join(',') : ''
    def metadataGroupOrderCsv = metadataPlotsGroupOrder && !metadataPlotsGroupOrder.isEmpty() ? metadataPlotsGroupOrder.join(',') : ''
    def metadataMicroFile = "${outputDir}/metadata/metadata_updated_micro.tsv"
    def metadataMitoFile = "${outputDir}/mito/metadata/metadata_updated_mito.tsv"
    def asvMetaMicroFile = "${outputDir}/metadata/ASV_meta_micro.tsv"
    def asvMetaMitoFile = "${outputDir}/mito/metadata/ASV_meta_mito.tsv"
    def asvTargetMicroFile = "${outputDir}/ASVs/ASV_target.micro.tsv"
    def asvTargetMitoFile = "${outputDir}/mito/ASVs/ASV_target.mito.tsv"
    def asvFinalMicroFile = "${outputDir}/ASVs/ASV_final.micro.tsv"
    def asvFinalMitoFile = "${outputDir}/mito/ASVs/ASV_final.mito.tsv"
    def asvTaxTable = "${outputDir}/taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv"
    """
set -euo pipefail

cmd=(
  python "${plotMetadataScriptPath}"
  --data-dir "${outputDir}"
  --sub-dir "${metadataPlotsSubDir}"
  --metadata "${metadataPlotsMetadataPath}"
  --taxonomy "${asvTaxTable}"
  --asv-micro "${asvTargetMicroFile}"
  --asv-mito "${asvTargetMitoFile}"
  --sample-id-col "${metadataPlotsSampleCol}"
  --group1-col "${metadataPlotsTypeCol}"
  --color-col "${metadataPlotsColorCol}"
  --sample-manifest "${manifestPath}"
  --make-micro
  --make-mito
  --verbose
)
${includeRankAppend}
if [[ -n "${metadataKeepTypesCsv}" ]]; then
  cmd+=( --keep-types "${metadataKeepTypesCsv}" )
fi
if [[ -n "${metadataGroupOrderCsv}" ]]; then
  cmd+=( --group-order "${metadataGroupOrderCsv}" )
fi

if [[ -n "${metadataBiochemTable}" ]]; then
  if [[ -f "${metadataBiochemTable}" ]]; then
    cmd+=( --biochem-assignments "${metadataBiochemTable}" )
    cmd+=( --biochem-sample-col "${metadataPlotsBiochemSampleCol}" )
    if [[ -n "${metadataBiochemIncludeCsv}" ]]; then
      cmd+=( --biochem-include-cols "${metadataBiochemIncludeCsv}" )
    fi
    if [[ -n "${metadataBiochemMetaJoinCsv}" || -n "${metadataBiochemJoinCsv}" ]]; then
      if [[ -n "${metadataBiochemMetaJoinCsv}" && -n "${metadataBiochemJoinCsv}" ]]; then
        cmd+=( --biochem-meta-join-cols "${metadataBiochemMetaJoinCsv}" )
        cmd+=( --biochem-join-cols "${metadataBiochemJoinCsv}" )
      else
        echo "[w] Incomplete biochem join config; both metadata and biochem join col lists are required. Falling back to sample-id join." >&2
      fi
    fi
  else
    echo "[w] metadata biochem assignments table not found; skipping merge: ${metadataBiochemTable}" >&2
  fi
fi

if [[ -n "${metadataStratTable}" ]]; then
  if [[ -f "${metadataStratTable}" ]]; then
    cmd+=( --stratification-timeseries "${metadataStratTable}" )
    cmd+=( --stratification-meta-join-col "${metadataPlotsStratMetaJoinCol}" )
    cmd+=( --stratification-join-col "${metadataPlotsStratJoinCol}" )
    if [[ -n "${metadataStratIncludeCsv}" ]]; then
      cmd+=( --stratification-include-cols "${metadataStratIncludeCsv}" )
    fi
  else
    echo "[w] metadata stratification table not found; skipping merge: ${metadataStratTable}" >&2
  fi
fi

"\${cmd[@]}"

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

process PLOT_UPSET {
    cpus pipelineThreads
    conda "${plotUpsetCondaEnvPath}"

    when:
    plotUpsetEnabled

    input:
    path(metadata_table)

    output:
    path("plot_upset.done"), emit: done

    script:
    def taxonomyArg = plotUpsetTaxonomyPath ? """  --taxonomy-path "${plotUpsetTaxonomyPath}" \\\n""" : ''
    def groupOrderArg = plotUpsetGroupOrder && !plotUpsetGroupOrder.isEmpty() ? """  --group-order "${plotUpsetGroupOrder.join(',')}" \\\n""" : ''
    def subsetGroupsArg = plotUpsetSubsetGroups && !plotUpsetSubsetGroups.isEmpty() ? """  --subset-groups "${plotUpsetSubsetGroups.join(',')}" \\\n""" : ''
    def skipVennArg = plotUpsetSkipVenn ? "  --skip-venn \\\n" : ''
    def rawOnlyArg = plotUpsetRawOnly ? "  --raw-only \\\n" : ''
    def finalOnlyArg = plotUpsetFinalOnly ? "  --final-only \\\n" : ''
    """
set -euo pipefail

python "${plotUpsetScriptPath}" \\
  --data-dir "${outputDir}" \\
  --subdir "${plotUpsetSubDir}" \\
  --domain "${plotUpsetDomain}" \\
${taxonomyArg}  --sample-id-col "${plotUpsetSampleIdCol}" \\
  --group-col "${plotUpsetGroupCol}" \\
  --color-col "${plotUpsetColorCol}" \\
${groupOrderArg}${subsetGroupsArg}${skipVennArg}${rawOnlyArg}${finalOnlyArg}  --formats "${plotUpsetFormats}" \\
  --font-size ${plotUpsetFontSize}

touch plot_upset.done
"""
}

process BUBBLEPLOTTER {
    cpus pipelineThreads
    conda "${bubbleplotterCondaEnvPath}"

    when:
    bubbleplotterEnabled

    input:
    path(asv_meta)

    output:
    path("bubbleplotter.done"), emit: done

    script:
    def noAutoSizeArg = bubbleplotterNoAutoSize ? "  --no-auto-size \\\n" : ''
    def bubbleplotterGroup1OrderArg = bubbleplotterGroup1Order && !bubbleplotterGroup1Order.isEmpty() ? """  --group1-order "${bubbleplotterGroup1Order.join(',')}" \\\n""" : ''
    def bubbleplotterGroup2OrderArg = bubbleplotterGroup2Order && !bubbleplotterGroup2Order.isEmpty() ? """  --group2-order "${bubbleplotterGroup2Order.join(',')}" \\\n""" : ''
    """
set -euo pipefail
mkdir -p "${bubbleplotterOutputDirAbs}"

python "${bubbleplotterScriptPath}" \\
  --input "${asv_meta}" \\
  --output-prefix "${bubbleplotterOutputPrefixAbs}" \\
  --count-col "${bubbleplotterCountCol}" \\
  --sample-col "${bubbleplotterSampleCol}" \\
  --group1-col "${bubbleplotterDepthCol}" \\
  --color-col "${bubbleplotterColorCol}" \\
  --group2-col "${bubbleplotterMonthCol}" \\
${bubbleplotterGroup1OrderArg}${bubbleplotterGroup2OrderArg}${noAutoSizeArg}  --formats "${bubbleplotterFormats}" \\
  --figsize "${bubbleplotterFigsize}" \\
  --bubble-scale ${bubbleplotterScale}

touch bubbleplotter.done
"""
}

process UMAP_CLUSTERING {
    cpus pipelineThreads
    conda "${umapClusteringCondaEnvPath}"

    when:
    umapClusteringEnabled

    input:
    path(asv_meta)

    output:
    path("umap_clustering.done"), emit: done

    script:
    def umapGroup1OrderArg = umapClusteringGroup1Order && !umapClusteringGroup1Order.isEmpty() ? """  --group1-order "${umapClusteringGroup1Order.join(',')}" \\\n""" : ''
    def umapGroup2OrderArg = umapClusteringGroup2Order && !umapClusteringGroup2Order.isEmpty() ? """  --group2-order "${umapClusteringGroup2Order.join(',')}" \\\n""" : ''
    def umapNoScaleArg = umapClusteringNoScale ? "  --no-scale \\\n" : ''
    """
set -euo pipefail
mkdir -p "${umapClusteringOutputDirAbs}"

python "${umapClusteringScriptPath}" \\
  --input "${asv_meta}" \\
  --output-prefix "${umapClusteringOutputPrefixAbs}" \\
  --count-col "${umapClusteringCountCol}" \\
  --sample-col "${umapClusteringSampleCol}" \\
  --group1-col "${umapClusteringDepthCol}" \\
  --color-col "${umapClusteringColorCol}" \\
  --group2-col "${umapClusteringSecondaryCol}" \\
${umapGroup1OrderArg}${umapGroup2OrderArg}  --formats "${umapClusteringFormats}" \\
  --normalize "${umapClusteringNormalize}" \\
  --transform "${umapClusteringTransform}" \\
  --n-neighbors ${umapClusteringNeighbors} \\
  --min-dist ${umapClusteringMinDist} \\
  --umap-metric "${umapClusteringMetric}" \\
  --min-cluster-size ${umapClusteringMinClusterSize} \\
  --min-samples ${umapClusteringMinSamples} \\
  --hdbscan-metric "${umapClusteringHdbscanMetric}" \\
${umapNoScaleArg}  --random-state ${umapClusteringRandomState}

touch umap_clustering.done
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
    path("asv_corrected_abundance.features_rows.tsv"), emit: asv_corrected_counts
    path("asv_corrected_pseudocount.features_rows.tsv"), emit: asv_corrected_counts_int
    path("batch_correction_countspace_preservation.png"), emit: countspace_plot
    path("batch_correction_countspace_preservation_metrics.tsv"), emit: countspace_metrics
    path("batch_correction_umap_comparison.png"), emit: umap_plot
    path("batch_correction_statistics.tsv"), emit: correction_stats
    path("umap_hdbscan_results.tsv"), emit: umap_results

    script:
    def bioCovArg = batchBiologicalCovariates ? """  --biological-covariates "${batchBiologicalCovariates}" \\\n""" : ''
    def minSamplesArg = batchHdbscanMinSamples != null ? "  --hdbscan-min-samples ${batchHdbscanMinSamples} \\\n" : ''
    def optimizeFlag = batchOptimize ? "  --optimize-clustering \\\n" : ''
    def conqurBatchRefArg = batchConqurBatchRef ? """  --conqur-batch-ref "${batchConqurBatchRef}" \\\n""" : ''
    def conqurLogisticLassoFlag = batchConqurLogisticLasso ? "  --conqur-logistic-lasso \\\n" : ''
    def conqurSimpleMatchFlag = batchConqurSimpleMatch ? "  --conqur-simple-match \\\n" : ''
    def conqurInterpltFlag = batchConqurInterplt ? "  --conqur-interplt \\\n" : ''
    def conqurAutoInstallFlag = batchConqurAutoInstall ? "  --conqur-auto-install \\\n" : ''
    def asvClrAfterFile = "${batchCorrectionOutputDirAbs}/asv_clr_after_correction.tsv"
    def asvCorrectedFeaturesFile = "${batchCorrectionOutputDirAbs}/asv_corrected_abundance.features_rows.tsv"
    def asvCorrectedPseudoFeaturesFile = "${batchCorrectionOutputDirAbs}/asv_corrected_pseudocount.features_rows.tsv"
    def countspacePlotFile = "${batchCorrectionOutputDirAbs}/batch_correction_countspace_preservation.png"
    def countspaceMetricsFile = "${batchCorrectionOutputDirAbs}/batch_correction_countspace_preservation_metrics.tsv"
    def umapComparisonPngFile = "${batchCorrectionOutputDirAbs}/batch_correction_umap_comparison.png"
    def batchCorrectionStatsFile = "${batchCorrectionOutputDirAbs}/batch_correction_statistics.tsv"
    def umapResultsFile = "${batchCorrectionOutputDirAbs}/umap_hdbscan_results.tsv"
    def asv_final = "ASVs/${asv_counts}"
    def updated_metadata = "metadata/${metadata_table}"
    def asv_metadata = "metadata/${asv_meta}"
    """
set -euo pipefail

python "${batchCorrectionScriptPath}" \\
  --data-dir "${outputDir}" \\
  --asv "${asv_final}" \\
  --metadata "${updated_metadata}" \\
  --asv-meta "${asv_metadata}" \\
  --sample-id-col "${batchCorrectionSampleIdCol}" \\
  --batch-col "${batchCorrectionBatchCol}" \\
  --output-dir "${batchCorrectionOutputDir}" \\
  --asv-orientation "${batchCorrectionOrientation}" \\
  --conqur-mode "${batchConqurMode}" \\
  --conqur-num-core ${batchConqurNumCore} \\
${conqurBatchRefArg}${conqurLogisticLassoFlag}  --conqur-quantile-type "${batchConqurQuantileType}" \\
${conqurSimpleMatchFlag}  --conqur-lambda-quantile "${batchConqurLambdaQuantile}" \\
${conqurInterpltFlag}  --conqur-delta ${batchConqurDelta} \\
${conqurAutoInstallFlag}${bioCovArg}  --umap-neighbors ${batchUmapNeighbors} \\
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
if [[ ! -f "${asvCorrectedFeaturesFile}" ]]; then
  echo "Missing batch correction output: ${asvCorrectedFeaturesFile}" >&2
  exit 1
fi
ln -sf "${asvCorrectedFeaturesFile}" asv_corrected_abundance.features_rows.tsv
if [[ ! -f "${asvCorrectedPseudoFeaturesFile}" ]]; then
  echo "Missing batch correction output: ${asvCorrectedPseudoFeaturesFile}" >&2
  exit 1
fi
ln -sf "${asvCorrectedPseudoFeaturesFile}" asv_corrected_pseudocount.features_rows.tsv
if [[ ! -f "${countspacePlotFile}" ]]; then
  echo "Missing batch correction output: ${countspacePlotFile}" >&2
  exit 1
fi
ln -sf "${countspacePlotFile}" batch_correction_countspace_preservation.png
if [[ ! -f "${countspaceMetricsFile}" ]]; then
  echo "Missing batch correction output: ${countspaceMetricsFile}" >&2
  exit 1
fi
ln -sf "${countspaceMetricsFile}" batch_correction_countspace_preservation_metrics.tsv
if [[ ! -f "${umapComparisonPngFile}" ]]; then
  echo "Missing batch correction output: ${umapComparisonPngFile}" >&2
  exit 1
fi
ln -sf "${umapComparisonPngFile}" batch_correction_umap_comparison.png
if [[ ! -f "${batchCorrectionStatsFile}" ]]; then
  echo "Missing batch correction output: ${batchCorrectionStatsFile}" >&2
  exit 1
fi
ln -sf "${batchCorrectionStatsFile}" batch_correction_statistics.tsv
if [[ ! -f "${umapResultsFile}" ]]; then
  echo "Missing batch correction output: ${umapResultsFile}" >&2
  exit 1
fi
ln -sf "${umapResultsFile}" umap_hdbscan_results.tsv
"""
}

process ASV_META_FROM_CORRECTED {
    cpus 1
    conda "${batchCorrectionCondaEnvPath}"

    when:
    batchCorrectionEnabled && (bubbleplotterEnabled || umapClusteringEnabled || clustermapsEnabled)

    input:
    path(asv_meta)
    path(corrected_counts)

    output:
    path("ASV_meta_micro.corrected.tsv"), emit: asv_meta_corrected

    script:
    """
set -euo pipefail

python - "${asv_meta}" "${corrected_counts}" "ASV_meta_micro.corrected.tsv" <<'PY'
import sys
import pandas as pd

asv_meta_path, corrected_counts_path, out_path = sys.argv[1:4]
sample_col = "${batchCorrectionSampleIdCol}"
asv_col = "ASV_ID"
count_col = "count"
count_alias_col = "corr_count"

meta = pd.read_csv(asv_meta_path, sep='\\t')
if sample_col not in meta.columns or asv_col not in meta.columns:
    raise ValueError(f"Expected columns '{sample_col}' and '{asv_col}' in {asv_meta_path}")

corr = pd.read_csv(corrected_counts_path, sep='\\t', index_col=0)
corr.index = corr.index.astype(str)
corr.columns = corr.columns.astype(str)

corr_long = corr.stack().rename(count_col).reset_index()
corr_long.columns = [asv_col, sample_col, count_col]
corr_long[count_col] = pd.to_numeric(corr_long[count_col], errors='coerce').fillna(0.0).clip(lower=0.0)

candidate_cols = [c for c in meta.columns if c not in {sample_col, asv_col, count_col, count_alias_col}]
asv_only_cols = []
sample_only_cols = []
for col in candidate_cols:
    asv_n = meta.groupby(asv_col, dropna=False)[col].nunique(dropna=False).max()
    sample_n = meta.groupby(sample_col, dropna=False)[col].nunique(dropna=False).max()
    if asv_n <= 1 and sample_n > 1:
        asv_only_cols.append(col)
    else:
        sample_only_cols.append(col)

sample_meta = meta[[sample_col] + sample_only_cols].drop_duplicates(subset=[sample_col])
asv_meta_df = meta[[asv_col] + asv_only_cols].drop_duplicates(subset=[asv_col])

out = corr_long.merge(sample_meta, on=sample_col, how='left')
out = out.merge(asv_meta_df, on=asv_col, how='left')
out = out[out[count_col] > 0]
out[count_alias_col] = out[count_col]

front = [sample_col, asv_col, count_col, count_alias_col]
remaining = [c for c in out.columns if c not in front]
out = out[front + remaining]
out.to_csv(out_path, sep='\\t', index=False)

print(f"[i] Wrote corrected ASV meta table: {out_path}")
print(f"[i] Rows: {len(out)}")
PY
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
    def updated_metadata = "metadata/${metadata_table}"
    def asvClrAfterFile = "${batchCorrectionOutputDirAbs}/asv_clr_after_correction.tsv"


    """
set -euo pipefail

python "${outlierCheckerScriptPath}" \\
  --data-dir "${outputDir}" \\
  --asv "${asvClrAfterFile}" \\
  --metadata "${updated_metadata}" \\
  --sample-id-col "${outlierSampleIdCol}" \\
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
    def collectorsGroupOrderArg = collectorsGroupOrder && !collectorsGroupOrder.isEmpty() ? """  --group-order "${collectorsGroupOrder.join(',')}" \\\n""" : ''
    """
set -euo pipefail

python "${collectorsCurveScriptPath}" \\
  --counts "${asv_counts}" \\
  --meta "${metadata_table}" \\
  --sample-col "${collectorsSampleCol}" \\
  --group-col "${collectorsGroupCol}" \\
  --color-col "${collectorsColorCol}" \\
${collectorsGroupOrderArg}  --permutations ${collectorsPermutations} \\
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

process DIVERSITY_ANALYSIS {
    cpus pipelineThreads
    conda "${diversityCondaEnvPath}"

    when:
    diversityEnabled

    input:
    path(metadata_table)
    path(asv_counts)

    output:
    path("diversity.done"), emit: done

    script:
    def secondaryColArg = diversitySecondaryCol ? """  --secondary-col "${diversitySecondaryCol}" \\\n""" : ''
    def excludeGroupsArg = diversityExcludeGroups && !diversityExcludeGroups.isEmpty() ? """  --exclude-groups "${diversityExcludeGroups.join(',')}" \\\n""" : ''
    def groupOrderArg = diversityGroupOrder && !diversityGroupOrder.isEmpty() ? """  --group-order "${diversityGroupOrder.join(',')}" \\\n""" : ''
    def diversityBlockArg = diversityBlockCol ? """  --block-col "${diversityBlockCol}" \\\n""" : ''
    def verboseFlag = diversityVerbose ? "  --verbose\n" : ''
    def diversityRunMitoFlag = diversityRunMito ? '1' : '0'
    """
set -euo pipefail
mkdir -p "${diversityOutputDirAbs}"
mkdir -p "${diversityMitoOutputDirAbs}"

if [[ "${diversityRunMitoFlag}" == "1" && -f "${diversityMitoInputPath}" ]]; then
  python "${calcDivScriptPath}" \\
    --micro-table "${asv_counts}" \\
    --mito-table "${diversityMitoInputPath}" \\
    --outdir "${diversityOutputDirAbs}" \\
    --mito-outdir "${diversityMitoOutputDirAbs}"
else
  python "${calcDivScriptPath}" \\
    --micro-table "${asv_counts}" \\
    --outdir "${diversityOutputDirAbs}"
fi

python "${plotDiversityScriptPath}" \\
  --metadata "${metadata_table}" \\
  --sample-col "${diversitySampleCol}" \\
  --group-col "${diversityGroupCol}" \\
  --color-col "${diversityColorCol}" \\
${secondaryColArg}${excludeGroupsArg}${groupOrderArg}  --alpha-table "${diversityOutputDirAbs}/shannon.tsv" \\
  --distance-bray "${diversityOutputDirAbs}/bray.tsv" \\
  --distance-jaccard "${diversityOutputDirAbs}/jaccard.tsv" \\
  --output-dir "${diversityOutputDirAbs}" \\
  --umap-neighbors ${diversityUmapNeighbors} \\
  --umap-min-dist ${diversityUmapMinDist} \\
${diversityBlockArg}  --permanova-perms ${diversityPermutations} \\
  --random-state ${diversityRandomState} \\
${verboseFlag}

if [[ "${diversityRunMitoFlag}" == "1" && -f "${diversityMitoOutputDirAbs}/shannon.mito.tsv" && -f "${diversityMitoOutputDirAbs}/bray.mito.tsv" && -f "${diversityMitoOutputDirAbs}/jaccard.mito.tsv" ]]; then
  python "${plotDiversityScriptPath}" \\
    --metadata "${metadata_table}" \\
    --sample-col "${diversitySampleCol}" \\
    --group-col "${diversityGroupCol}" \\
    --color-col "${diversityColorCol}" \\
${secondaryColArg}${excludeGroupsArg}${groupOrderArg}    --alpha-table "${diversityOutputDirAbs}/shannon.tsv" \\
    --distance-bray "${diversityOutputDirAbs}/bray.tsv" \\
    --distance-jaccard "${diversityOutputDirAbs}/jaccard.tsv" \\
    --output-dir "${diversityOutputDirAbs}" \\
    --mito-mode \\
    --mito-alpha "${diversityMitoOutputDirAbs}/shannon.mito.tsv" \\
    --mito-bray "${diversityMitoOutputDirAbs}/bray.mito.tsv" \\
    --mito-jaccard "${diversityMitoOutputDirAbs}/jaccard.mito.tsv" \\
    --mito-output-dir "${diversityMitoOutputDirAbs}" \\
    --umap-neighbors ${diversityUmapNeighbors} \\
    --umap-min-dist ${diversityUmapMinDist} \\
${diversityBlockArg}    --permanova-perms ${diversityPermutations} \\
    --random-state ${diversityRandomState} \\
${verboseFlag}
fi

touch diversity.done
"""
}

process INDICSPECIES {
    cpus pipelineThreads
    conda "${indicspeciesCondaEnvPath}"

    when:
    indicspeciesEnabled

    input:
    path(metadata_table)
    path(asv_counts)

    output:
    path("indicspecies_group1_summary.tsv"), emit: group1_summary
    path("indicspecies_group2_summary.tsv"), emit: group2_summary
    path("indicspecies_group1_results.tsv"), emit: group1_results
    path("indicspecies_group2_results.tsv"), emit: group2_results
    path("indicspecies_tables/*.tsv"), emit: all_tables
    path("indicspecies.done"), emit: done

    script:
    def indicspeciesGroupColsArg = indicspeciesGroupCols.join(',')
    def indicspeciesBlockArg = indicspeciesBlockCol ? """  --block-col "${indicspeciesBlockCol}" \\\n""" : ''
    def group1SummaryPath = "${indicspeciesOutputDirAbs}/${indicspeciesGroup1}_indicator_species_summary.tsv"
    def group2SummaryPath = "${indicspeciesOutputDirAbs}/${indicspeciesGroup2}_indicator_species_summary.tsv"
    def group1ResultsPath = "${indicspeciesOutputDirAbs}/${indicspeciesGroup1}_indicator_species_results.tsv"
    def group2ResultsPath = "${indicspeciesOutputDirAbs}/${indicspeciesGroup2}_indicator_species_results.tsv"
    """
set -euo pipefail
mkdir -p "${indicspeciesOutputDirAbs}"

Rscript "${indicspeciesScriptPath}" \\
  --asv "${asv_counts}" \\
  --meta "${metadata_table}" \\
  --sample-col "${indicspeciesSampleCol}" \\
  --group-cols "${indicspeciesGroupColsArg}" \\
${indicspeciesBlockArg}  --perms ${indicspeciesPerms} \\
  --min-n ${indicspeciesMinN} \\
  --outdir "${outputDir}"

python - <<'PY'
from pathlib import Path
import sys

group1_summary = Path("${group1SummaryPath}")
group2_summary = Path("${group2SummaryPath}")
group1_results = Path("${group1ResultsPath}")
group2_results = Path("${group2ResultsPath}")
required = [group1_summary, group2_summary, group1_results, group2_results]

missing = [str(p) for p in required if not p.is_file()]
if missing:
    for p in missing:
        print(f"Missing expected indicspecies output: {p}", file=sys.stderr)
    raise SystemExit(1)

tables_dir = Path("indicspecies_tables")
tables_dir.mkdir(parents=True, exist_ok=True)
all_tables = sorted(Path("${indicspeciesOutputDirAbs}").glob("*_indicator_species*.tsv"))
if not all_tables:
    print("No indicspecies tables were generated in ${indicspeciesOutputDirAbs}", file=sys.stderr)
    raise SystemExit(1)

for src in all_tables:
    dst = tables_dir / src.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())

link_map = {
    "indicspecies_group1_summary.tsv": group1_summary,
    "indicspecies_group2_summary.tsv": group2_summary,
    "indicspecies_group1_results.tsv": group1_results,
    "indicspecies_group2_results.tsv": group2_results,
}
for dst_name, src in link_map.items():
    dst = Path(dst_name)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())
PY

touch indicspecies.done
"""
}

process INDICSPECIES_PLOTS {
    cpus pipelineThreads
    conda "${indicspeciesCondaEnvPath}"

    when:
    indicspeciesEnabled && indicspeciesPlotEnabled

    input:
    path(metadata_table)
    path(indicspecies_tables)

    output:
    path("indicspecies_plots.done"), emit: done

    script:
    def plotVennPath = indicspeciesPlotVennPath ?: ''
    def plotTaxPath = indicspeciesPlotTaxonomyPath ?: ''
    def plotPairsMode = indicspeciesPlotPairsMode?.toString()?.trim()?.toLowerCase() ?: 'all'
    """
set -euo pipefail
mkdir -p "${indicspeciesPlotOutputDirAbs}"

python - <<'PY'
from pathlib import Path
import itertools
import re
import subprocess
import sys

plot_script = Path("${plotIndicspeciesScriptPath}")
out_root = Path("${indicspeciesPlotOutputDirAbs}")
pairs_mode = "${plotPairsMode}"
plot_tax = Path("${plotTaxPath}") if "${plotTaxPath}" else None
plot_venn = Path("${plotVennPath}") if "${plotVennPath}" else None
metadata_path = Path("${metadataPlotsMetadataPath}") if "${metadataPlotsMetadataPath}" else None
preferred_group1 = "${indicspeciesGroup1}".strip()
preferred_group2 = "${indicspeciesGroup2}".strip()
metadata_color_col = "${indicspeciesColorCol}".strip()
group1_palette_cfg = "${indicspeciesGroup1Palette}"
group2_palette_cfg = "${indicspeciesGroup2Palette}"
group1_order_cfg = "${indicspeciesGroup1Order.join(',')}"
group2_order_cfg = "${indicspeciesGroup2Order.join(',')}"
focus_group1_cfg = "${indicspeciesFocusGroup1Label}"
focus_group2_cfg = "${indicspeciesFocusGroup2Label}"
label_focused_asvs = ${indicspeciesLabelFocusedAsvs ? 'True' : 'False'}

result_files = sorted(Path(".").glob("*_indicator_species*_results.tsv"))
if len(result_files) < 2:
    print(
        f"[w] Need at least 2 indicspecies result tables to build ISA plots; found {len(result_files)}. Skipping.",
        file=sys.stderr,
    )
    raise SystemExit(0)

def clean_group_name(raw_name: str) -> str:
    name = raw_name
    if name.endswith("_results.tsv"):
        name = name[: -len("_results.tsv")]
    for suffix in ("_indicator_species_DULEG", "_indicator_species"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name

def orient_pair(a: Path, b: Path) -> tuple[Path, Path]:
    an = clean_group_name(a.name)
    bn = clean_group_name(b.name)
    if an == preferred_group1 and bn == preferred_group2:
        return a, b
    if an == preferred_group2 and bn == preferred_group1:
        return b, a
    if an == preferred_group1:
        return a, b
    if bn == preferred_group1:
        return b, a
    return a, b

if pairs_mode == "first_vs_rest":
    raw_pairs = [(result_files[0], f) for f in result_files[1:]]
else:
    raw_pairs = list(itertools.combinations(result_files, 2))
pair_iter = [orient_pair(a, b) for a, b in raw_pairs]

def has_col(path: Path, col: str) -> bool:
    if not col:
        return False
    try:
        header = path.read_text(encoding="utf-8").splitlines()[0].split("\t")
    except Exception:
        return False
    return col in header

for g1_file, g2_file in pair_iter:
    g1_name = clean_group_name(g1_file.name)
    g2_name = clean_group_name(g2_file.name)
    if g1_name == g2_name:
        print(
            f"[w] Skipping same-group ISA pair: {g1_file.name} vs {g2_file.name}",
            file=sys.stderr,
        )
        continue
    pair_slug = re.sub(r"[^0-9A-Za-z._-]", "_", f"{g1_name}__{g2_name}")
    pair_out = out_root / pair_slug
    pair_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(plot_script),
        "--group1-results",
        str(g1_file),
        "--group2-results",
        str(g2_file),
        "--group1-name",
        g1_name,
        "--group2-name",
        g2_name,
        "--outdir",
        str(pair_out),
    ]
    if metadata_path and metadata_path.is_file():
        cmd.extend(["--metadata", str(metadata_path)])
        cmd.extend(["--group1-meta-label-col", g1_name])
        cmd.extend(["--group2-meta-label-col", g2_name])
        if g1_name == preferred_group1 and metadata_color_col:
            cmd.extend(["--group1-meta-color-col", metadata_color_col])

    if has_col(g1_file, g1_name):
        cmd.extend(["--group1-label-col", g1_name])
    if has_col(g1_file, f"{g1_name}_Color"):
        cmd.extend(["--group1-color-col", f"{g1_name}_Color"])
    elif has_col(g1_file, "Color"):
        cmd.extend(["--group1-color-col", "Color"])

    if has_col(g2_file, g2_name):
        cmd.extend(["--group2-label-col", g2_name])
    if has_col(g2_file, f"{g2_name}_Color"):
        cmd.extend(["--group2-color-col", f"{g2_name}_Color"])
    elif has_col(g2_file, "Color"):
        cmd.extend(["--group2-color-col", "Color"])

    if has_col(g2_file, f"{g2_name}_Marker"):
        cmd.extend(["--group2-marker-col", f"{g2_name}_Marker"])
    if g1_name == preferred_group1 and group1_palette_cfg:
        cmd.extend(["--group1-palette", group1_palette_cfg])
    if g2_name == preferred_group2 and group2_palette_cfg:
        cmd.extend(["--group2-palette", group2_palette_cfg])
    if g1_name == preferred_group1 and group1_order_cfg:
        cmd.extend(["--group1-order", group1_order_cfg])
    if g2_name == preferred_group2 and group2_order_cfg:
        cmd.extend(["--group2-order", group2_order_cfg])
    if g1_name == preferred_group1 and focus_group1_cfg:
        cmd.extend(["--focus-group1-label", focus_group1_cfg])
    if g2_name == preferred_group2 and focus_group2_cfg:
        cmd.extend(["--focus-group2-label", focus_group2_cfg])
    if label_focused_asvs:
        cmd.append("--label-focused-asvs")
    if plot_tax and plot_tax.is_file():
        cmd.extend(["--taxonomy", str(plot_tax)])
    if plot_venn and plot_venn.is_file():
        cmd.extend(["--venn", str(plot_venn)])

    subprocess.run(cmd, check=True)
PY

touch indicspecies_plots.done
"""
}

process CLUSTERMAPS {
    cpus pipelineThreads
    conda "${clustermapsCondaEnvPath}"

    when:
    clustermapsEnabled

    input:
    path(asv_meta)
    path(metadata_table)

    output:
    path("clustermaps.done"), emit: done

    script:
    def group3ColArg = clustermapsGroup3Col ? """  --group3-col "${clustermapsGroup3Col}" \\\n""" : ''
    def group3PaletteArg = clustermapsGroup3Palette ? """  --group3-palette "${clustermapsGroup3Palette}" \\\n""" : ''
    def clustermapsGroup1OrderArg = clustermapsGroup1Order && !clustermapsGroup1Order.isEmpty() ? """  --group1-order "${clustermapsGroup1Order.join(',')}" \\\n""" : ''
    def clustermapsRunMitoFlag = clustermapsRunMito ? '1' : '0'
    def isaFileArg = clustermapsIsaFile ?: ''
    def isaSigColsArg = clustermapsIsaSignificanceCols ? """  --isa-significance-cols "${clustermapsIsaSignificanceCols}" \\\n""" : ''
    def isaStatColsArg = clustermapsIsaStatCols ? """  --isa-stat-cols "${clustermapsIsaStatCols}" \\\n""" : ''
    """
set -euo pipefail
mkdir -p "${clustermapsOutputDirAbs}"
mkdir -p "${clustermapsMitoOutputDirAbs}"

ISA_ARGS=()
if [[ -n "${isaFileArg}" && -f "${isaFileArg}" ]]; then
  ISA_ARGS=(--isa "${isaFileArg}")
fi

if [[ "${clustermapsRunMitoFlag}" == "1" && -f "${clustermapsMitoInputPath}" ]]; then
  python "${clustermapsScriptPath}" \\
    --asv-meta "${asv_meta}" \\
    --metadata "${metadata_table}" \\
    --outdir "${clustermapsOutputDirAbs}" \\
    --sample-col "${clustermapsSampleCol}" \\
    --sample-code-col "${clustermapsSampleCodeCol}" \\
    --asv-id-col "${clustermapsAsvIdCol}" \\
    --group1-col "${clustermapsGroup1Col}" \\
    --group2-col "${clustermapsGroup2Col}" \\
${group3ColArg}${clustermapsGroup1OrderArg}    --exclude-group1 "${clustermapsExcludeGroup1}" \\
    --group1-palette "${clustermapsGroup1Palette}" \\
    --group2-palette "${clustermapsGroup2Palette}" \\
${group3PaletteArg}    --ranks "${clustermapsRanks}" \\
    --topN "${clustermapsTopN}" \\
    --count-col "${clustermapsCountCol}" \\
    --isa-min-stat ${clustermapsIsaMinStat} \\
${isaSigColsArg}${isaStatColsArg}    --mito-sample-mode "${clustermapsMitoSampleMode}" \\
    --mito-asv "${clustermapsMitoInputPath}" \\
    --mito-outdir "${clustermapsMitoOutputDirAbs}" \\
    \${ISA_ARGS[@]}
else
  python "${clustermapsScriptPath}" \\
    --asv-meta "${asv_meta}" \\
    --metadata "${metadata_table}" \\
    --outdir "${clustermapsOutputDirAbs}" \\
    --sample-col "${clustermapsSampleCol}" \\
    --sample-code-col "${clustermapsSampleCodeCol}" \\
    --asv-id-col "${clustermapsAsvIdCol}" \\
    --group1-col "${clustermapsGroup1Col}" \\
    --group2-col "${clustermapsGroup2Col}" \\
${group3ColArg}${clustermapsGroup1OrderArg}    --exclude-group1 "${clustermapsExcludeGroup1}" \\
    --group1-palette "${clustermapsGroup1Palette}" \\
    --group2-palette "${clustermapsGroup2Palette}" \\
${group3PaletteArg}    --ranks "${clustermapsRanks}" \\
    --topN "${clustermapsTopN}" \\
    --count-col "${clustermapsCountCol}" \\
    --isa-min-stat ${clustermapsIsaMinStat} \\
${isaSigColsArg}${isaStatColsArg}    --mito-sample-mode "${clustermapsMitoSampleMode}" \\
    \${ISA_ARGS[@]}
fi

touch clustermaps.done
"""
}

process SPIECEASI {
    cpus pipelineThreads
    conda "${spieceasiCondaEnvPath}"

    when:
    spieceasiEnabled

    input:
    path(asv_counts)

    output:
    path("spieceasi_network_pos_all.graphml"), emit: graph_all
    path("spieceasi_network_pos_thr.graphml"), emit: graph_thr
    path("spieceasi_node_features.csv"), emit: node_features
    path("spieceasi.done"), emit: done

    script:
    def transposeFlag = spieceasiTranspose ? 'TRUE' : 'FALSE'
    def removeZeroVarFlag = spieceasiRemoveZeroVar ? 'TRUE' : 'FALSE'
    def keepNegativeFlag = spieceasiKeepNegative ? 'TRUE' : 'FALSE'
    def forceFilterFlag = spieceasiForceFilter ? 'TRUE' : 'FALSE'
    def forceSpieceasiFlag = spieceasiForceSpieceasi ? 'TRUE' : 'FALSE'
    def forceGraphsFlag = spieceasiForceGraphs ? 'TRUE' : 'FALSE'
    """
set -euo pipefail
mkdir -p "${spieceasiOutputDirAbs}"

Rscript "${spieceasiScriptPath}" \\
  --counts "${asv_counts}" \\
  --outdir "${spieceasiOutputDirAbs}" \\
  --prefix "${spieceasiPrefix}" \\
  --transpose ${transposeFlag} \\
  --min-rel-abund ${spieceasiMinRelAbund} \\
  --min-prevalence ${spieceasiMinPrevalence} \\
  --remove-zero-var ${removeZeroVarFlag} \\
  --method "${spieceasiMethod}" \\
  --lambda-min-ratio ${spieceasiLambdaMinRatio} \\
  --nlambda ${spieceasiNlambda} \\
  --rep-num ${spieceasiRepNum} \\
  --thresh ${spieceasiThresh} \\
  --ncores ${spieceasiNcores} \\
  --seed ${spieceasiSeed} \\
  --edge-threshold ${spieceasiEdgeThreshold} \\
  --keep-negative ${keepNegativeFlag} \\
  --layout-iters ${spieceasiLayoutIters} \\
  --force-filter ${forceFilterFlag} \\
  --force-spieceasi ${forceSpieceasiFlag} \\
  --force-graphs ${forceGraphsFlag}

GRAPH_ALL="${spieceasiOutputDirAbs}/${spieceasiPrefix}_network_pos_all.graphml"
GRAPH_THR="${spieceasiOutputDirAbs}/${spieceasiPrefix}_network_pos_thr.graphml"
NODE_FEATURES="${spieceasiOutputDirAbs}/${spieceasiPrefix}_node_features.csv"

for f in "\${GRAPH_ALL}" "\${GRAPH_THR}" "\${NODE_FEATURES}"; do
  if [[ ! -f "\${f}" ]]; then
    echo "Missing expected SPIEC-EASI output: \${f}" >&2
    exit 1
  fi
done

ln -sf "\${GRAPH_ALL}" spieceasi_network_pos_all.graphml
ln -sf "\${GRAPH_THR}" spieceasi_network_pos_thr.graphml
ln -sf "\${NODE_FEATURES}" spieceasi_node_features.csv
touch spieceasi.done
"""
}

process NETWORK_MODULES {
    cpus pipelineThreads
    conda "${networkModulesCondaEnvPath}"

    when:
    networkEnabled && networkModulesEnabled

    input:
    path(graph_all)
    path(graph_thr)

    output:
    path("network_modules_sub.tsv"), emit: modules_sub
    path("network_modules_all.tsv"), emit: modules_all
    path("network_modules_summary.tsv"), emit: summary
    path("network_modules_runs.tsv"), emit: runs
    path("network_modules.done"), emit: done

    script:
    def methodsCsv = networkModuleMethods.join(',')
    def resolutionsCsv = networkModuleResolutions.join(',')
    """
set -euo pipefail
mkdir -p "${spieceasiOutputDirAbs}"

Rscript "${networkModulesScriptPath}" \\
  --graph-sub "${graph_thr}" \\
  --graph-all "${graph_all}" \\
  --outdir "${spieceasiOutputDirAbs}" \\
  --prefix "${spieceasiPrefix}" \\
  --methods "${methodsCsv}" \\
  --primary-method "${networkModulePrimaryMethod}" \\
  --reps ${networkModuleReps} \\
  --resolutions "${resolutionsCsv}" \\
  --consensus-threshold ${networkModuleConsensusThreshold} \\
  --seed ${networkModuleSeed}

MODULES_SUB="${spieceasiOutputDirAbs}/${spieceasiPrefix}_modules_sub.tsv"
MODULES_ALL="${spieceasiOutputDirAbs}/${spieceasiPrefix}_modules_all.tsv"
MODULE_SUMMARY="${spieceasiOutputDirAbs}/${spieceasiPrefix}_module_summary.tsv"
MODULE_RUNS="${spieceasiOutputDirAbs}/${spieceasiPrefix}_module_runs.tsv"

for f in "\${MODULES_SUB}" "\${MODULES_ALL}" "\${MODULE_SUMMARY}" "\${MODULE_RUNS}"; do
  if [[ ! -f "\${f}" ]]; then
    echo "Missing expected network module output: \${f}" >&2
    exit 1
  fi
done

ln -sf "\${MODULES_SUB}" network_modules_sub.tsv
ln -sf "\${MODULES_ALL}" network_modules_all.tsv
ln -sf "\${MODULE_SUMMARY}" network_modules_summary.tsv
ln -sf "\${MODULE_RUNS}" network_modules_runs.tsv
touch network_modules.done
"""
}

process GRAPH_NETWORK {
    cpus pipelineThreads
    conda "${networkCondaEnvPath}"

    when:
    networkEnabled

    input:
    path(graph_all)
    path(graph_thr)
    path(node_features)
    path(asv_counts)
    path(taxonomy_table)
    path(group1_summary)
    path(group2_summary)
    path(modules_sub)
    path(modules_all)

    output:
    path("network.done"), emit: done

    script:
    def networkModesArg = networkModes && !networkModes.isEmpty() ? """  --modes ${networkModes.collect { "\"${it}\"" }.join(' ')} \\\n""" : ''
    def networkGroup1PaletteArg = networkGroup1Palette ? """  --group1-palette "${networkGroup1Palette}" \\\n""" : ''
    def networkGroup2PaletteArg = networkGroup2Palette ? """  --group2-palette "${networkGroup2Palette}" \\\n""" : ''
    def networkGroup1OrderArg = networkGroup1Order && !networkGroup1Order.isEmpty() ? """  --group1-order "${networkGroup1Order.join(',')}" \\\n""" : ''
    def networkGroup2OrderArg = networkGroup2Order && !networkGroup2Order.isEmpty() ? """  --group2-order "${networkGroup2Order.join(',')}" \\\n""" : ''
    def networkFocusGroup1Arg = networkFocusGroup1Label ? """  --focus-group1-label "${networkFocusGroup1Label}" \\\n""" : ''
    def networkFocusGroup2Arg = networkFocusGroup2Label ? """  --focus-group2-label "${networkFocusGroup2Label}" \\\n""" : ''
    def networkModuleBestOnlyArg = networkModuleBestOnly ? """  --module-best-only \\\n""" : ''
    def networkModuleIsaOnlyArg = networkModuleIsaOnly ? """  --module-isa-only \\\n""" : ''
    def networkModuleColorByIsaArg = networkModuleColorByIsa ? """  --module-color-by-isa \\\n""" : ''
    """
set -euo pipefail
mkdir -p "${spieceasiOutputDirAbs}"

python "${graphNetworkScriptPath}" \\
  --data-dir "${outputDir}" \\
  --outdir "${spieceasiOutputDirAbs}" \\
  --graph-pos-all "${graph_all}" \\
  --graph-pos-sub "${graph_thr}" \\
  --node-features "${node_features}" \\
  --asv-counts "${asv_counts}" \\
  --taxonomy "${taxonomy_table}" \\
  --group1-summary "${group1_summary}" \\
  --group2-summary "${group2_summary}" \\
  --group1-name "${indicspeciesGroup1}" \\
  --group2-name "${indicspeciesGroup2}" \\
  --metadata "${networkMetadataPath}" \\
  --sample-col "${indicspeciesSampleCol}" \\
  --group1-col "${indicspeciesGroup1}" \\
  --group2-col "${indicspeciesGroup2}" \\
  --color-col "${networkColorCol}" \\
${networkGroup1PaletteArg}${networkGroup2PaletteArg}${networkGroup1OrderArg}${networkGroup2OrderArg}${networkFocusGroup1Arg}${networkFocusGroup2Arg}${networkModuleBestOnlyArg}${networkModuleIsaOnlyArg}${networkModuleColorByIsaArg}  --module-best-min-size ${networkModuleBestMinSize} \\
  --module-best-min-stability ${networkModuleBestMinStability} \\
  --module-isa-source "${networkModuleIsaSource}" \\
  --module-isa-min-stat ${networkModuleIsaMinStat} \\
  --module-isa-max-q ${networkModuleIsaMaxQ} \\
  --modules-sub "${modules_sub}" \\
  --modules-all "${modules_all}" \\
${networkModesArg}  --layout-seed ${networkLayoutSeed} \\
  --layout-scale ${networkLayoutScale} \\
  --degree-scale ${networkDegreeScale} \\
  --edge-width-scale ${networkEdgeWidthScale} \\
  --isa-scale ${networkIsaScale}

touch network.done
"""
}

process MASTER_SUMMARY {
    cpus 1
    conda "${masterSummaryCondaEnvPath}"

    when:
    masterSummaryEnabled

    input:
    path(asv_meta)
    path(asv_counts)
    path(dep_network)
    path(dep_sankey)

    output:
    path("ASV_master_long.tsv"), optional: true, emit: master_long
    path("ASV_master_count_wide.tsv"), optional: true, emit: master_count
    path("ASV_master_source_manifest.tsv"), optional: true, emit: master_manifest
    path("ASV_master_column_mapping.tsv"), optional: true, emit: master_colmap
    path("ASV_master_column_collisions_original.tsv"), optional: true, emit: master_collisions
    path("master_summary.done"), emit: done

    script:
    def whitelistArg = masterSummaryWhitelistCsv ? """  --whitelist "${masterSummaryWhitelistCsv}" \\\n""" : ''
    """
set -euo pipefail
mkdir -p "${masterSummaryOutputDirAbs}"

python "${masterSummaryScriptPath}" \\
  --data-dir "${outputDir}" \\
  --asv-meta "${asv_meta}" \\
  --asv-counts "${asv_counts}" \\
  --clustermaps-dir "${masterSummaryClustermapsDirAbs}" \\
  --indicspecies-dir "${masterSummaryIndicspeciesDirAbs}" \\
  --spieceasi-dir "${masterSummarySpieceasiDirAbs}" \\
${whitelistArg}  --outdir "${masterSummaryOutputDirAbs}" \\
  --max-direct-cols ${masterSummaryMaxDirectCols}

link_if_exists() {
  local src="\$1"
  local dest="\$2"
  if [[ -f "\${src}" ]]; then
    ln -sf "\${src}" "\${dest}"
  fi
}

link_if_exists "${masterSummaryOutputDirAbs}/ASV_master_long.tsv" "ASV_master_long.tsv"
link_if_exists "${masterSummaryOutputDirAbs}/ASV_master_count_wide.tsv" "ASV_master_count_wide.tsv"
link_if_exists "${masterSummaryOutputDirAbs}/ASV_master_source_manifest.tsv" "ASV_master_source_manifest.tsv"
link_if_exists "${masterSummaryOutputDirAbs}/ASV_master_column_mapping.tsv" "ASV_master_column_mapping.tsv"
link_if_exists "${masterSummaryOutputDirAbs}/ASV_master_column_collisions_original.tsv" "ASV_master_column_collisions_original.tsv"

touch master_summary.done
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
