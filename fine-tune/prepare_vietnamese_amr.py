import os
import json
import re
import pyvi.ViTokenizer as ViTokenizer
from collections import defaultdict

try:
    import phonlp
    print("Loading PhoNLP model...")
    # Tải model PhoNLP vào thư mục hiện tại nếu chưa có
    if not os.path.exists('./phonlp'):
        phonlp.download(save_dir='./phonlp')
    phonlp_model = phonlp.load(save_dir='./phonlp')
    HAS_PHONLP = True
except ImportError:
    print("WARNING: phonlp is not installed. Will output dummy dependency matrices.")
    HAS_PHONLP = False

def extract_dependency_matrix(segmented_snt):
    words = segmented_snt.split()
    n = len(words)
    matrix = [[0] * n for _ in range(n)]
    # Tự liên kết chính nó (self-loop)
    for i in range(n):
        matrix[i][i] = 1
        
    if not HAS_PHONLP or n == 0:
        return matrix

    try:
        # PhoNLP input is a raw string or segmented string?
        # PhoNLP annotate takes segmented string or unsegmented? Usually it can take unsegmented and segment it, 
        # but to match exact tokens, we should pass the pre-segmented words or just let PhoNLP do it.
        # However, AMRBART needs alignment. Let's just pass the segmented string.
        annotations = phonlp_model.annotate(segmented_snt)
        # annotations: [sentences] -> sentence: (words, pos, ner, dependencies)
        # Assuming single sentence for AMR
        if len(annotations[0]) > 0:
            words, pos, ner, deps = annotations[0][0], annotations[1][0], annotations[2][0], annotations[3][0]
            # Mismatch check
            if len(words) == n:
                for i, (head_idx, rel) in enumerate(deps):
                    # head_idx is 1-based, 0 means root
                    head_idx = int(head_idx)
                    if head_idx > 0:
                        # Symmetric adjacency
                        matrix[i][head_idx - 1] = 1
                        matrix[head_idx - 1][i] = 1
    except Exception as e:
        pass
        
    return matrix

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
            
            # Extract Adjacency Matrix
            dep_matrix = extract_dependency_matrix(segmented_snt)

            # Write out JSONL
            record = {
                "amr": graph_str,
                "sent": segmented_snt,  # Use segmented sentence
                "dependency_matrix": dep_matrix # Add dependency matrix
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
