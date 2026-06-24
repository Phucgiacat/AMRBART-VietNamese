import os
import json
import re
import pyvi.ViTokenizer as ViTokenizer
from collections import defaultdict

def preprocess_amr_file(input_file, output_jsonl):
    print(f"Processing {input_file} -> {output_jsonl}")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().split('\n\n')

    with open(output_jsonl, 'w', encoding='utf-8') as out_f:
        for block in content:
            if not block.strip():
                continue
            lines = block.strip().split('\n')
            snt = ""
            graph_lines = []
            for line in lines:
                if line.startswith('# ::snt'):
                    snt = line.replace('# ::snt', '').strip()
                elif line.startswith('#'):
                    continue
                else:
                    graph_lines.append(line)
            
            if not snt or not graph_lines:
                continue

            # Segment sentence using Pyvi
            segmented_snt = ViTokenizer.tokenize(snt)
            gt_words = set(segmented_snt.split())

            # Linearized graph string
            graph_str = " ".join(graph_lines)
            graph_str = re.sub(r'\s+', ' ', graph_str)

            # Write out JSONL
            record = {
                "amr": graph_str,
                "sent": segmented_snt  # Use segmented sentence for BARTpho/AMRBART
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    data_dir = "../../data_amrbart"
    for split in ["train", "dev", "test"]:
        in_file = os.path.join(data_dir, f"{split}.amr")
        out_file = os.path.join(data_dir, f"{split}.jsonl")
        if split == "dev":
            out_file = os.path.join(data_dir, "val.jsonl")
        if os.path.exists(in_file):
            preprocess_amr_file(in_file, out_file)
