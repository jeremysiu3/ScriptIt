import json

# Load common characters
with open('data/processed/mandarin_common_2000_chars.json', 'r', encoding='utf-8') as f:
    common_chars = set(json.load(f))

print(f"Loaded {len(common_chars)} common characters")

# Load all sentence data from 3 CSVs
import zipfile
import pandas as pd
from pypinyin import pinyin, Style

zip_path = "data/raw/mandarin_news_dataset.zip"  

print("Loading from 3 CSVs and filtering to common characters...")

training_data = []
total_processed = 0
max_articles = 50000  # Process 50k from the 3 CSVs

with zipfile.ZipFile(zip_path, 'r') as zip_file:
    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')][:3]
    
    for csv_name in csv_files:
        print(f"\nProcessing {csv_name}...")
        
        with zip_file.open(csv_name) as csv_file:
            df = pd.read_csv(csv_file)
            
            for idx, row in df.iterrows():
                if total_processed >= max_articles:
                    break
                
                chinese_text = str(row['content']).strip()
                
                if len(chinese_text) < 10:
                    continue
                
                # Filter: only keep text with mostly common characters
                chinese_chars_in_text = [c for c in chinese_text if '\u4e00' <= c <= '\u9fff']
                common_chars_in_text = [c for c in chinese_chars_in_text if c in common_chars]
                
                # REPLACE rare characters with common alternatives or remove them
                filtered_text = ''.join([c if (c in common_chars or not ('\u4e00' <= c <= '\u9fff')) else '' for c in chinese_text])

                # Skip if too short after filtering
                if len(filtered_text) < 10:
                    continue

                # Use filtered text instead of original
                chinese_text = filtered_text
                
                try:
                    # Convert to pinyin
                    pinyin_result = pinyin(chinese_text, style=Style.TONE3, v_to_u=False)
                    pinyin_text = ''.join([item[0] for item in pinyin_result])
                    
                    training_data.append({
                        'input': pinyin_text.lower(),
                        'label': pinyin_text,
                        'chinese': chinese_text
                    })
                    
                    total_processed += 1
                    
                    if total_processed % 5000 == 0:
                        print(f"  Processed {total_processed} articles...")
                        
                except Exception as e:
                    continue
        
        if total_processed >= max_articles:
            break

print(f"\n✓ Created {len(training_data)} filtered training examples")

# Show samples
print("\nSample data:")
for i in range(3):
    print(f"\nExample {i+1}:")
    print(f"Chinese: {training_data[i]['chinese'][:50]}...")
    print(f"Pinyin:  {training_data[i]['input'][:50]}...")

# Save
with open('data/processed/mandarin_news_data_filtered.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)

print(f"\n✓ Saved to mandarin_news_data_filtered.json")