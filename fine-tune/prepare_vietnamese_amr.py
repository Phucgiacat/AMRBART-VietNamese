import os
import json
import re
from collections import defaultdict
from tqdm import tqdm
import sys

# --- MONKEY PATCH REQUESTS FOR HUGGINGFACE BUG ---
import requests
orig_request = requests.Session.request
def patched_request(self, method, url, *args, **kwargs):
    if url.startswith('/api/'):
        url = 'https://huggingface.co' + url
    return orig_request(self, method, url, *args, **kwargs)
requests.Session.request = patched_request
# -------------------------------------------------

try:
    import phonlp
    print("Loading PhoNLP model...")
    if not os.path.exists('./phonlp'):
        phonlp.download(save_dir='./phonlp')
    phonlp_model = phonlp.load(save_dir='./phonlp')
    HAS_PHONLP = True
except ImportError:
    print("WARNING: phonlp is not installed. Will output dummy dependency matrices.")
    HAS_PHONLP = False

def preprocess_amr_file(input_file, output_jsonl):
    print(f"Processing {input_file} -> {output_jsonl}")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().split('\n\n')

    blocks_data = []
    texts_to_annotate = []
    
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

        graph_str = " ".join(graph_lines)
        graph_str = re.sub(r'\s+', ' ', graph_str)
        
        blocks_data.append({
            "snt": snt,
            "graph_str": graph_str
        })
        texts_to_annotate.append(snt)

    all_words = []
    all_deps = []
    
    if HAS_PHONLP and texts_to_annotate:
        print(f"Annotating {len(texts_to_annotate)} sentences with PhoNLP...")
        old_stderr = sys.stderr
        with open(os.devnull, 'w') as fnull:
            sys.stderr = fnull
            for text in tqdm(texts_to_annotate, desc="PhoNLP Annotation", file=old_stderr):
                try:
                    annotations = phonlp_model.annotate(text)
                    if len(annotations[0]) > 0:
                        all_words.append(annotations[0][0])
                        all_deps.append(annotations[3][0])
                    else:
                        all_words.append([])
                        all_deps.append([])
                except Exception:
                    all_words.append([])
                    all_deps.append([])
            sys.stderr = old_stderr
    else:
        # Dummy if no PhoNLP
        for t in texts_to_annotate:
            w = t.split()
            all_words.append(w)
            all_deps.append([])

    with open(output_jsonl, 'w', encoding='utf-8') as out_f:
        for i, b_data in enumerate(blocks_data):
            d_words = all_words[i]
            deps = all_deps[i]
            n = len(d_words)
            
            matrix = [[0] * n for _ in range(n)]
            for j in range(n):
                matrix[j][j] = 1
                
            if n > 0:
                for w_idx, (head_idx, rel) in enumerate(deps):
                    head_idx = int(head_idx)
                    if head_idx > 0:
                        matrix[w_idx][head_idx - 1] = 1
                        matrix[head_idx - 1][w_idx] = 1
            
            segmented_snt = " ".join(d_words) if n > 0 else b_data["snt"]
            
            record = {
                "amr": b_data["graph_str"],
                "sent": segmented_snt,
                "dependency_matrix": matrix
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"Finished {output_jsonl}")

if __name__ == "__main__":
    data_dir = "../../data_amrbart"
    for split in ["train", "dev", "test"]:
        in_file = os.path.join(data_dir, f"{split}.amr")
        out_file = os.path.join(data_dir, f"{split}.jsonl")
        if split == "dev":
            out_file = os.path.join(data_dir, "val.jsonl")
        if os.path.exists(in_file):
            preprocess_amr_file(in_file, out_file)
