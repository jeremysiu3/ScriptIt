import os
from korean_romanizer import Romanizer
import json
import re

# Load text
file_path = "data/raw/korean_dataset.zst" 
print("Loading Korean Wikipedia and extracting words...")

import zstandard as zstd

korean_words = set() # Use set to avoid duplicates
line_count = 0
max_lines = 50000

try:
    with open(file_path, 'rb') as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            text_data = reader.read().decode('utf-8')
            
            for line in text_data.split('\n'):
                if line_count >= max_lines:
                    break
                
                korean_text = line.strip()
                
                # Skip empty
                if len(korean_text) < 5:
                    continue
                
                # Extract Korean words (split by spaces and punctuation)
                # Remove punctuation and split
                words = re.findall(r'[가-힣]+', korean_text)
                
                for word in words:
                    # Only keep words of reasonable length (2-10 characters)
                    if 2 <= len(word) <= 10:
                        korean_words.add(word)
                
                line_count += 1
                
                if line_count % 5000 == 0:
                    print(f"Processed {line_count} lines, found {len(korean_words)} unique words...")

except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

print(f"\n✓ Extracted {len(korean_words)} unique Korean words")

# Romanize each word
print("\nRomanizing words...")
training_data = []

for i, korean_word in enumerate(korean_words):
    try:
        r = Romanizer(korean_word)
        romanized = r.romanize()
        
        # Create training pair
        training_data.append({
            'input': romanized.lower(),
            'label': romanized,
            'korean': korean_word
        })
        
        if (i + 1) % 5000 == 0:
            print(f"Romanized {i + 1} words...")
            
    except Exception as e:
        continue

print(f"\n✓ Created {len(training_data)} word-level training pairs")

# Samples
print("\nSample word pairs:")
for i in range(min(10, len(training_data))):
    print(f"{training_data[i]['korean']:8} → {training_data[i]['label']:20} (input: {training_data[i]['input']})")

# Save
with open('data/processed/korean_word_training_data.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)

print(f"\n✓ Saved to korean_word_training_data.json")