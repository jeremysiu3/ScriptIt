import os
import json
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Load Bengali articles
folder_path = "data/raw/bengali_dataset"
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

print(f"Found {len(json_files)} Bengali articles")
print("Loading and converting to romanization...")

training_data = []

# Process each file
for i, filename in enumerate(json_files):
    if i >= 6000:  # Process 6000 articles
        break
    
    file_path = os.path.join(folder_path, filename)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        bengali_text = data.get('content', '').strip()
        
        # Skip if empty or too short
        if len(bengali_text) < 10:
            continue
        
        # Convert Bengali script to ITRANS romanization
        romanized = transliterate(bengali_text, sanscript.BENGALI, sanscript.ITRANS)
        
        # Create training pair
        input_text = romanized.lower()  # All lowercase (ambiguous)
        label_text = romanized          # With proper case (vowel lengths)
        
        training_data.append({
            'input': input_text,
            'label': label_text,
            'bengali': bengali_text
        })
        
        # Show progress
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} articles...")
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

print(f"\n✓ Created {len(training_data)} training examples")

# Show samples
print("\nSample training data:")
for i in range(3):
    print(f"\nExample {i+1}:")
    print(f"Bengali: {training_data[i]['bengali'][:80]}...")
    print(f"Input: {training_data[i]['input'][:80]}...")
    print(f"Label: {training_data[i]['label'][:80]}...")

# Save to file
import json
with open('data/processed/bengali_training_data.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)

print(f"\n✓ Saved to bengali_training_data.json")