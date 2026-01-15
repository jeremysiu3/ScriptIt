import json
from pypinyin import pinyin, Style

# Load conversational data
file_path = "data/raw/mandarin_convo_dataset.json" 

with open(file_path, 'r', encoding='utf-8') as f:
    conversations = json.load(f)

print(f"Loaded conversation data")
print(f"Type: {type(conversations)}")

# Check if it's nested (train/test split)
if isinstance(conversations, dict) and 'train' in conversations:
    conv_data = conversations['train']
elif isinstance(conversations, list):
    conv_data = conversations
else:
    print("Unexpected format!")
    exit()

print(f"Total conversation pairs: {len(conv_data)}")

# Extract individual sentences
training_data = []
processed = 0
max_sentences = 50000  # Process 50k sentences

for conversation in conv_data:
    if processed >= max_sentences:
        break
    
    # Each conversation is a list of dialogue turns
    if isinstance(conversation, list):
        for turn in conversation:
            # Remove spaces between characters
            chinese_text = turn.replace(' ', '').strip()
            
            # Skip if too short or empty
            if len(chinese_text) < 2:
                continue
            
            # Skip if mostly non-Chinese
            chinese_chars = sum(1 for c in chinese_text if '\u4e00' <= c <= '\u9fff')
            if chinese_chars < len(chinese_text) * 0.7:
                continue
            
            try:
                # Convert to pinyin
                pinyin_result = pinyin(chinese_text, style=Style.TONE3, v_to_u=False)
                pinyin_text = ''.join([item[0] for item in pinyin_result])
                
                training_data.append({
                    'input': pinyin_text.lower(),
                    'label': pinyin_text,
                    'chinese': chinese_text
                })
                
                processed += 1
                
                if processed % 5000 == 0:
                    print(f"Processed {processed} sentences...")
                    
            except Exception as e:
                continue

print(f"\n✓ Created {len(training_data)} conversational training examples")

# Show samples
print("\nSample conversational data:")
for i in range(10):
    print(f"\nExample {i+1}:")
    print(f"Chinese: {training_data[i]['chinese']}")
    print(f"Pinyin:  {training_data[i]['input'][:50]}...")

# Save
with open('data/processed/mandarin_convo_data.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)

print(f"\n✓ Saved to mandarin_convo_data.json")