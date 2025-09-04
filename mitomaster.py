#!/usr/bin/env python3
import os
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


def process_fasta(fasta_path, first=False):
    try:
        with open(fasta_path, 'rb') as f:
            files = {
                "file": f,
                "fileType": ('', 'sequences'),
                "output": ('', 'hsd')
            }
            response = requests.post("https://mitomap.org/mitomaster/websrvc.cgi", files=files)
            response.raise_for_status()
            output = response.text

        # Write output and update checkpoint
        with lock:
            with open(output_file, 'a') as out_f:
                if first:
                    out_f.write(output)
                else:
                    out_f.write('\n' + '\n'.join(output.splitlines()[1:]))
            with open(checkpoint_file, 'a') as chk_f:
                chk_f.write(fasta_path.name + '\n')

        print(f"✅ Done: {fasta_path.name}")
    except Exception as e:
        print(f"❌ Error: {fasta_path.name}: {e}")

data_dir = '/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_combined_output/ASVs/chunks'
output_file = '/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_combined_output/mito/mitomap/mitomaster_combined.tsv'
checkpoint_file = output_file + '.done'

fasta_files = sorted(Path(data_dir).glob("*.fasta"))

# Read already completed files
done = set()
if Path(checkpoint_file).exists():
    with open(checkpoint_file) as f:
        done.update(line.strip() for line in f)

# Lock for writing to files safely across threads
lock = Lock()

# Filter only unprocessed files
remaining = [f for f in fasta_files if f.name not in done]
print(f"Found {len(remaining)} unprocessed FASTA files")

if remaining:
    first = True
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for fasta in remaining:
            futures.append(executor.submit(process_fasta, fasta, first))
            first = False  # Only keep header from first
        for f in as_completed(futures):
            f.result()  # Trigger exception if any
else:
    print("✅ All files already processed.")

print(f"📝 Results saved to {output_file}")
