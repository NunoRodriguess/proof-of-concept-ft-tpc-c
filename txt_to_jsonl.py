input_path = "tpch_facts_small.txt"      # Change to your .txt file path
output_path = "tpch_small.jsonl"  # Change to desired .jsonl output path

import json

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.strip()
        if line:
            json.dump({"text": line}, outfile)
            outfile.write("\n")