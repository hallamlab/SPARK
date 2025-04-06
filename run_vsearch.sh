#!/bin/bash

# ================================
# ASV Processing Script with Skip Flags
# ================================

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display help message
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --skip-fastp          Skip Step 1: Fastp QC reads"
    echo "  --skip-merge          Skip Step 2: Merging paired-end reads"
    echo "  --skip-filter         Skip Step 3: Filtering merged reads"
    echo "  --skip-concat         Skip Step 4: Concatenating all filtered reads"
    echo "  --skip-derep          Skip Step 5: Dereplicating sequences"
    echo "  --skip-denoise        Skip Step 6: ASV denoising with UNOISE"
    echo "  --skip-chimera        Skip Step 7: ASV chimera check with uchime3_denovo"
    echo "  --skip-swarm          Skip Step 8: Swarm clustering to remove artifacts"
    echo "  --skip-nontarget      Skip Step 9: Removing non-targets"
    echo "  --skip-count          Skip Step 10: Assigning counts and creating ASV count matrix"
    echo "  -h, --help            Display this help message and exit"
    echo ""
    echo "Example:"
    echo "  $0 --skip-merge --skip-filter"
    exit 1
}

# Initialize skip flags to 0 (do not skip)
SKIP_QC=0
SKIP_MERGE=0
SKIP_FILTER=0
SKIP_CONCAT=0
SKIP_DEREP=0
SKIP_DENOISE=0
SKIP_CHIMERA=0
SKIP_SWARM=0
SKIP_NONTARGET=0
SKIP_COUNT=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-fastp)
            SKIP_QC=1
            shift
            ;;
        --skip-merge)
            SKIP_MERGE=1
            shift
            ;;
        --skip-filter)
            SKIP_FILTER=1
            shift
            ;;
        --skip-concat)
            SKIP_CONCAT=1
            shift
            ;;
        --skip-derep)
            SKIP_DEREP=1
            shift
            ;;
        --skip-denoise)
            SKIP_DENOISE=1
            shift
            ;;
        --skip-chimera)
            SKIP_CHIMERA=1
            shift
            ;;
        --skip-swarm)
            SKIP_SWARM=1
            shift
            ;;
        --skip-nontarget)
            SKIP_NONTARGET=1
            shift
            ;;
        --skip-count)
            SKIP_COUNT=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Define directories
INPUT_DIR="fastq_input"
REFDB_DIR="ref_db"
OUTPUT_DIR="vsearch_output"
QC_DIR="${OUTPUT_DIR}/fastp"
MERGED_DIR="${OUTPUT_DIR}/merged"
FILTERED_DIR="${OUTPUT_DIR}/filtered"
CONCAT_DIR="${OUTPUT_DIR}/concat"
DEREP_DIR="${OUTPUT_DIR}/derep"
DEN_DIR="${OUTPUT_DIR}/denoise"
NOC_DIR="${OUTPUT_DIR}/nochimeras"
SWM_DIR="${OUTPUT_DIR}/swarm"
ASV_DIR="${OUTPUT_DIR}/ASVs"
LOG_DIR="${OUTPUT_DIR}/logs"
THREADS="$(nproc --all)"


# Define primers with degenerate bases using regex
#PRIMER_F="GTGYCAGCMGCCGCGGTAA"
#PRIMER_R="GGACTACNVGGGTWTCTAAT"

# Create necessary output directories
mkdir -p ${QC_DIR} ${MERGED_DIR} ${FILTERED_DIR} ${CONCAT_DIR} ${DEREP_DIR} ${DEN_DIR} ${NOC_DIR} ${ASV_DIR} ${LOG_DIR} ${SWM_DIR}

fastp_qc() {
    echo "Step 1: FastP QC..."
    for R1 in ${INPUT_DIR}/*_R1_001.fastq.gz; do
        SAMPLE=$(basename ${R1} _R1_001.fastq.gz)
        R2="${INPUT_DIR}/${SAMPLE}_R2_001.fastq.gz"
        
        # Check if R2 exists
        if [[ ! -f "${R2}" ]]; then
            echo "Warning: Reverse read ${R2} not found for sample ${SAMPLE}. Skipping."
            continue
        fi

        fastp \
            -i ${INPUT_DIR}/${SAMPLE}_R1_001.fastq.gz \
            -I ${INPUT_DIR}/${SAMPLE}_R2_001.fastq.gz \
            -o ${QC_DIR}/${SAMPLE}_R1.fastq.gz \
            -O ${QC_DIR}/${SAMPLE}_R2.fastq.gz \
            -f 19 -t 80 \
            -F 20 -T 80 \
            -j ${QC_DIR}/${SAMPLE}_report.json \
            -h ${QC_DIR}/${SAMPLE}_report.html \
            -w ${THREADS}
    done 2>&1 | tee -a "${LOG_DIR}/fastp_log.txt"
    echo "Step 1: FastP QC completed."   
}

# Function to merge paired-end reads
merge_reads() {
    echo "Step 2: Merging paired-end reads..."
    for R1 in ${QC_DIR}/*_R1.fastq.gz; do
        SAMPLE=$(basename ${R1} _R1.fastq.gz)
        R2="${QC_DIR}/${SAMPLE}_R2.fastq.gz"
        
        # Check if R2 exists
        if [[ ! -f "${R2}" ]]; then
            echo "Warning: Reverse read ${R2} not found for sample ${SAMPLE}. Skipping."
            continue
        fi

        vsearch --fastq_mergepairs ${R1} \
                --reverse ${R2} \
                --fastqout ${MERGED_DIR}/${SAMPLE}.merged.fastq \
                --fastq_maxdiffs 20 \
                --fastq_minovlen 5 \
                --fastq_allowmergestagger \
                --threads ${THREADS}
    done 2>&1 | tee -a "${LOG_DIR}/merging_log.txt"
    echo "Step 2: Merging completed."
}

# Function to filter merged reads
filter_reads() {
    echo "Step 3: Filtering merged reads..."
    for M in ${MERGED_DIR}/*.merged.fastq; do
        SAMPLE=$(basename ${M} .merged.fastq)
        vsearch --fastx_filter ${M} \
            --fastq_maxee 1.0 \
            --fastq_minlen 200 \
            --fastaout ${FILTERED_DIR}/${SAMPLE}.filtered.fasta
    done 2>&1 | tee -a "${LOG_DIR}/filtering_log.txt"
    echo "Step 3: Filtering completed."
}

# Function to concatenate all filtered reads
concatenate_reads() {
    echo "Step 4: Concatenating all filtered reads..."
    rm -f ${CONCAT_DIR}/concat.fasta
    for F in ${FILTERED_DIR}/*.filtered.fasta; do
        SAMPLE=$(basename ${F} .filtered.fasta)
        
        # Extract the first sequence header and modify it
        HEADER=$(head -n 1 "${F}")
        SUBSTRING=$(echo ${HEADER} | cut -d'>' -f2 | cut -d':' -f1)
        NEW_HEADER=">${SAMPLE}"
        
        # Replace the old header with the new header and append to concat.fasta
        sed "s/${SUBSTRING}/${SAMPLE}/g" ${F} >> ${CONCAT_DIR}/concat.fasta
    done 2>&1 | tee -a "${LOG_DIR}/concat_log.txt"
    echo "Step 4: Concatenating completed."
}

# Function to dereplicate sequences
dereplicate_sequences() {
    echo "Step 5: Dereplicating sequences..."
    vsearch --derep_fulllength ${CONCAT_DIR}/concat.fasta \
            --output ${DEREP_DIR}/derep.fasta \
            --sizeout \
            --threads ${THREADS} \
            --log ${LOG_DIR}/derep_log.txt
    echo "Step 5: Dereplication completed."
}

# Function for ASV denoising with UNOISE
denoise_asv() {
    echo "Step 6: Performing ASV denoising with UNOISE..."
    COUNT=$(awk '/^>/ {count++} END {print count}' "${CONCAT_DIR}/concat.fasta")
    #MINSIZE=$(echo "($COUNT * 0.00005)/1" | bc) # https://www.nature.com/articles/nmeth.2276
    MINSIZE=3
    vsearch --cluster_unoise ${DEREP_DIR}/derep.fasta \
            --centroids ${DEN_DIR}/centroids.fasta \
            --sizein \
            --sizeout \
            --minsize ${MINSIZE} \
            --threads ${THREADS} \
            --log ${LOG_DIR}/denoise_log.txt
    echo "Step 6: ASV denoising completed."
}

# Function to perform chimera checking
chimera_check() {
    echo "Step 7: Performing chimera check with uchime3_denovo..."
    vsearch --uchime3_denovo ${DEN_DIR}/centroids.fasta \
            --nonchimeras ${NOC_DIR}/nochimeras.fasta \
            --sizein \
            --sizeout \
            --threads ${THREADS} \
            --log ${LOG_DIR}/nochimera_log.txt
    echo "Step 7: Chimera checking completed."
}

# Function for Swarm clustering
swarm_clustering() {
    echo "Step 8: Performing Swarm clustering to remove artifacts..."
    swarm -d 1 -f -t ${THREADS} -z \
          -i ${SWM_DIR}/swarm_reps.struct \
          -j ${SWM_DIR}/swarm_reps.network \
          -s ${SWM_DIR}/swarm_reps.stats.txt \
          -w ${SWM_DIR}/swarm_reps.fasta \
          -o ${SWM_DIR}/swarm_reps.swarms \
          ${NOC_DIR}/nochimeras.fasta 2>&1 | tee -a "${LOG_DIR}/swarm_log.txt"
    echo "Step 8: Swarm clustering completed."
}

# Function to remove nontarget
remove_nontarget() {
    echo "Step 9: Removing Non-target..."
    vsearch --usearch_global ${NOC_DIR}/nochimeras.fasta \
            --db ./ssu_pipeline_contaminants_mito.fasta \
            --matched ${ASV_DIR}/ASVs.nontarget.fasta \
            --notmatched ${ASV_DIR}/ASVs.fasta \
            --id 0.99 \
            --sizein \
            --relabel ASV \
            --threads ${THREADS} \
            --log ${LOG_DIR}/nontarget_log.txt
    echo "Step 9: Non-target removed."
}

# Function to assign counts and create ASV count matrix
create_count_matrix() {
    echo "Step 10: Creating ASV count matrix..."
    vsearch --usearch_global ${CONCAT_DIR}/concat.fasta \
            --db ${ASV_DIR}/ASVs.fasta \
            --id 0.999 \
            --otutabout ${ASV_DIR}/ASV_counts.tsv \
            --threads ${THREADS} \
            --log ${LOG_DIR}/count_log.txt
    
    echo "Step 10: Filtering samples by ASV sum..."
    python filter_ASV_table.py \
            ${ASV_DIR}/ASV_counts.tsv \
            ${REFDB_DIR}/spark_metadata.tsv \
            ${ASV_DIR}/ASV_filtered.tsv \
            5000 0.005
    echo "Step 10: ASV count matrix created and filtered."
}


# Execute steps based on skip flags
if [[ ${SKIP_QC} -eq 0 ]]; then
    fastp_qc
else
    echo "Skipping Step 1: FastP QC as per user request."
fi

if [[ ${SKIP_MERGE} -eq 0 ]]; then
    merge_reads
else
    echo "Skipping Step 2: Merging paired-end reads as per user request."
fi

if [[ ${SKIP_FILTER} -eq 0 ]]; then
    filter_reads
else
    echo "Skipping Step 3: Filtering merged reads as per user request."
fi

if [[ ${SKIP_CONCAT} -eq 0 ]]; then
    concatenate_reads
else
    echo "Skipping Step 4: Concatenating all filtered reads as per user request."
fi

if [[ ${SKIP_DEREP} -eq 0 ]]; then
    dereplicate_sequences
else
    echo "Skipping Step 5: Dereplicating sequences as per user request."
fi

if [[ ${SKIP_DENOISE} -eq 0 ]]; then
    denoise_asv
else
    echo "Skipping Step 6: ASV denoising with UNOISE as per user request."
fi

if [[ ${SKIP_CHIMERA} -eq 0 ]]; then
    chimera_check
else
    echo "Skipping Step 7: ASV chimera check with uchime3_denovo as per user request."
fi

if [[ ${SKIP_SWARM} -eq 0 ]]; then
    swarm_clustering
else
    echo "Skipping Step 8: Swarm clustering as per user request."
fi

if [[ ${SKIP_NONTARGET} -eq 0 ]]; then
    remove_nontarget
else
    echo "Skipping Step 9: Removing non-targets as per user request."
fi

if [[ ${SKIP_COUNT} -eq 0 ]]; then
    create_count_matrix
else
    echo "Skipping Step 10: Creating ASV count matrix as per user request."
fi

echo "Amplicon processing with VSEARCH completed successfully."
