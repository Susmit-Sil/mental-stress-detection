import subprocess
import csv
import os

os.environ['KAGGLE_API_TOKEN'] = "KGAT_740122ecfc2faeeaf42dc26602f30f11"
KAGGLE_CMD = r"venv\Scripts\kaggle.exe"

queries = [
    "mental stress text",
    "emotion text",
    "depression text",
    "mental health text",
    "suicide text",
    "sentiment emotion",
]

downloaded = set()

# We already downloaded some in the previous step
already_have = [
    "payaldhokane/stress-analysis-in-social-media-dataset",
    "mohammedmoinuddin/emotion-classification-150k",
    "andrihjonior/mental-health-emotion-3000",
    "szegeelim/mental-health"
]
downloaded.update(already_have)

def search_kaggle(query):
    print(f"\nSearching for: {query}")
    result = subprocess.run(
        [KAGGLE_CMD, "datasets", "list", "-s", query, "--csv"],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"Error searching for {query}: {result.stderr}")
        return []
        
    datasets = []
    # Parse CSV
    lines = result.stdout.strip().split('\n')
    if len(lines) > 1:
        reader = csv.DictReader(lines)
        for row in reader:
            datasets.append(row['ref'])
    return datasets

def main():
    all_refs = []
    for q in queries:
        refs = search_kaggle(q)
        all_refs.extend(refs)
        
    # Remove duplicates
    unique_refs = list(set(all_refs))
    print(f"\nTotal unique datasets found across all queries: {len(unique_refs)}")
    
    # Filter keywords to avoid images/audio/unrelated
    skip_keywords = ['image', 'audio', 'video', 'vision', 'face', 'speech', 'voice', 'music', 'sound', 'eeg']
    
    to_download = []
    for ref in unique_refs:
        ref_lower = ref.lower()
        if ref in downloaded:
            continue
        if any(skip in ref_lower for skip in skip_keywords):
            print(f"Skipping (likely multimedia): {ref}")
            continue
        to_download.append(ref)
        
    print(f"\nWill download {len(to_download)} new datasets...")
    
    for i, ref in enumerate(to_download, 1):
        print(f"\n[{i}/{len(to_download)}] Downloading {ref}...")
        res = subprocess.run(
            [KAGGLE_CMD, "datasets", "download", "-d", ref, "-p", r"data\raw", "--unzip"]
        )
        if res.returncode == 0:
            print(f"Success: {ref}")
        else:
            print(f"Failed: {ref}")
            
if __name__ == "__main__":
    main()
