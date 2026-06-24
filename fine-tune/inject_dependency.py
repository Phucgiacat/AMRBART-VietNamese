import os
import sys
import json
import pyvi.ViTokenizer as ViTokenizer

try:
    import phonlp
    HAS_PHONLP = True
except ImportError:
    print("WARNING: phonlp is not installed. Dependency matrices will be empty.")
    HAS_PHONLP = False

phonlp_model = None

def extract_dependency_matrix(segmented_snt):
    global phonlp_model
    words = segmented_snt.split()
    n = len(words)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1
        
    if not HAS_PHONLP or n == 0:
        return matrix

    if phonlp_model is None:
        print("Loading PhoNLP model...")
        if not os.path.exists('./phonlp'):
            phonlp.download(save_dir='./phonlp')
        phonlp_model = phonlp.load(save_dir='./phonlp')

    try:
        annotations = phonlp_model.annotate(segmented_snt)
        if len(annotations[0]) > 0:
            words, pos, ner, deps = annotations[0][0], annotations[1][0], annotations[2][0], annotations[3][0]
            if len(words) == n:
                for i, (head_idx, rel) in enumerate(deps):
                    head_idx = int(head_idx)
                    if head_idx > 0:
                        matrix[i][head_idx - 1] = 1
                        matrix[head_idx - 1][i] = 1
    except Exception as e:
        print(f"PhoNLP error: {e}")
    return matrix

def main():
    if len(sys.argv) < 2:
        print("Usage: python inject_dependency.py <jsonl_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        sys.exit(1)

    print(f"Checking dependency matrices for {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    needs_update = False
    new_lines = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except:
            new_lines.append(line)
            continue
            
        if "dependency_matrix" not in record or not record["dependency_matrix"]:
            needs_update = True
            tgt = record.get("tgt", record.get("sent", ""))
            
            # Segment if not already segmented (pyvi handles this idempotently mostly, but let's be careful)
            # If it already has underscores, we can assume it's segmented
            if "_" not in tgt and " " in tgt:
                tgt = ViTokenizer.tokenize(tgt)
                record["tgt"] = tgt
                record["sent"] = tgt
            
            print(f"Injecting dependency matrix for line {idx+1}...")
            record["dependency_matrix"] = extract_dependency_matrix(tgt)
            new_lines.append(json.dumps(record, ensure_ascii=False))
        else:
            new_lines.append(line)
            
    if needs_update:
        print(f"Updating {file_path} with injected dependency matrices...")
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in new_lines:
                f.write(line + '\n')
    else:
        print("All records already have dependency matrices.")

if __name__ == "__main__":
    main()
