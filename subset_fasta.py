#!/usr/bin/env python3
import sys
import pandas as pd
import os
from Bio import SeqIO

def subset_fasta(defs_file, fasta_in, tax_file, outdir):
    
    defs_df = pd.read_csv(defs_file, sep='\t', header=0)
    tax_df = pd.read_csv(tax_file, sep='\t', header=0)
    for group in defs_df['grouping'].unique():
        grp = group.replace(' ', '_')
        out_file = os.path.join(outdir, grp + '.venn.fasta')
        defs = defs_df.loc[defs_df['grouping'] == group]['ASV_ID'].to_list()
        tax_dict = {x: (y,z) for x,y,z in zip(tax_df['Feature ID'], tax_df['Taxon'], tax_df['Consensus'])}
        out_records = []
        for rec in SeqIO.parse(fasta_in, "fasta"):
            if rec.id in defs:
                # append the extra info to the record ID
                rec.id = f"{rec.id}||{tax_dict[rec.id][0]}||{tax_dict[rec.id][1]}"
                # also update .name and .description so nothing weird shows up
                rec.name = rec.id
                rec.description = ""
                out_records.append(rec)
        SeqIO.write(out_records, out_file, "fasta")

if __name__ == '__main__':
#    if len(sys.argv) != 4:
#        print(f"Usage: {sys.argv[0]} defs.txt input.fasta output.fasta", file=sys.stderr)
#        sys.exit(1)
#    subset_fasta(sys.argv[1], sys.argv[2], sys.argv[3])
    data_dir = '/home/ryan/Projects/UBC/LMP/SPARK_data/'
    venn_file = os.path.join(data_dir, "vsearch_output/metadata/venn3_presence_table.tsv")
    asv_fasta = os.path.join(data_dir, "vsearch_output/ASVs/ASV_filtered.micro.fasta")
    asv_tax = os.path.join(data_dir, "vsearch_output/taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv")

    outdir = os.path.join(data_dir, "vsearch_output/ASVs")
    subset_fasta(venn_file, asv_fasta, asv_tax, outdir)