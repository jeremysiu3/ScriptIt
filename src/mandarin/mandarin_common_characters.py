import zipfile
import pandas as pd
from collections import Counter

zip_path = "data/raw/mandarin_news_dataset.zip"  

print("Analyzing character frequency across Chinese news...")

char_counter = Counter()
articles_processed = 0

with zipfile.ZipFile(zip_path, 'r') as zip_file:
    # Process first 3 CSVs
    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')][:3]
    
    for csv_name in csv_files:
        print(f"Processing {csv_name}...")
        
        with zip_file.open(csv_name) as csv_file:
            df = pd.read_csv(csv_file)
            
            for idx, row in df.iterrows():
                chinese_text = str(row['content'])
                
                # Count only Chinese characters
                for char in chinese_text:
                    if '\u4e00' <= char <= '\u9fff':  # Chinese character range
                        char_counter[char] += 1
                
                articles_processed += 1
                
                if articles_processed % 5000 == 0:
                    print(f"  Processed {articles_processed} articles...")

print(f"\n✓ Analyzed {articles_processed} articles")
print(f"✓ Found {len(char_counter)} unique Chinese characters")

# Get top 5000 most common
common_chars = [char for char, count in char_counter.most_common(2000)]

print(f"\nMost common 20 characters:")
for char, count in char_counter.most_common(20):
    print(f"  {char}: {count:,} occurrences")

# Save to file
import json
with open('data/processed/mandarin_common_2000_chars.json', 'w', encoding='utf-8') as f:
    json.dump(common_chars, f, ensure_ascii=False, indent=2)

print(f"\n✓ Saved 2,000 most common characters to mandarin_common_2000_chars.json")