import zipfile
import pandas as pd
from pypinyin import pinyin, Style
import json

zip_path = "data/raw/mandarin_news_dataset.zip"  

print("Loading Chinese news articles...")

training_data = []
total_processed = 0
max_articles = 20000 

with zipfile.ZipFile(zip_path, 'r') as zip_file:
    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_name in csv_files:
        print(f"\nProcessing {csv_name}...")
        
        with zip_file.open(csv_name) as csv_file:
            df = pd.read_csv(csv_file)
            
            print(f"  Articles in this file: {len(df)}")
            
            for idx, row in df.iterrows():
                if total_processed >= max_articles:
                    break
                
                # Get Chinese text from content column
                chinese_text = str(row['content']).strip()
                
                # Skip if empty or too short
                if len(chinese_text) < 10:
                    continue
                
                # Skip if mostly non-Chinese
                chinese_chars = sum(1 for c in chinese_text if '\u4e00' <= c <= '\u9fff')
                if chinese_chars < len(chinese_text) * 0.5:
                    continue
                
                try:
                    # Convert to pinyin with tone numbers
                    pinyin_result = pinyin(chinese_text, style=Style.TONE3, v_to_u=False)
                    pinyin_text = ''.join([item[0] for item in pinyin_result])
                    
                    # Create training pair
                    training_data.append({
                        'input': pinyin_text.lower(),  # Lowercase pinyin
                        'label': pinyin_text,          # Keep original (for reference)
                        'chinese': chinese_text
                    })
                    
                    total_processed += 1
                    
                    if total_processed % 1000 == 0:
                        print(f"  Processed {total_processed} articles...")
                        
                except Exception as e:
                    continue
        
        if total_processed >= max_articles:
            break

print(f"\n✓ Created {len(training_data)} training examples")

# Samples
print("\nSample training data:")
for i in range(3): 
    print(f"\nExample {i+1}:")
    print(f"Chinese: {training_data[i]['chinese'][:60]}...")
    print(f"Pinyin:  {training_data[i]['input'][:60]}...")

# Save    
with open('data/processed/mandarin_news_data.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)

print(f"\n✓ Saved to mandarin_news_data.json")