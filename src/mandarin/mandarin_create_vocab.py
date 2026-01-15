import json

# Load sentence-level training data
with open('data/processed/mandarin_combined_training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} examples")

# Build vocabularies
input_chars = set()   # Pinyin characters (a-z, 0-5 for tones, v for ü)
output_chars = set()  # Chinese characters

for example in data:
    input_chars.update(example['input'])
    output_chars.update(example['chinese'])

# Create mappings
input_char_to_idx = {char: idx for idx, char in enumerate(sorted(input_chars))}
input_char_to_idx['<PAD>'] = len(input_char_to_idx)
input_char_to_idx['<SOS>'] = len(input_char_to_idx)
input_char_to_idx['<EOS>'] = len(input_char_to_idx)

output_char_to_idx = {char: idx for idx, char in enumerate(sorted(output_chars))}
output_char_to_idx['<PAD>'] = len(output_char_to_idx)
output_char_to_idx['<SOS>'] = len(output_char_to_idx)
output_char_to_idx['<EOS>'] = len(output_char_to_idx)

input_idx_to_char = {idx: char for char, idx in input_char_to_idx.items()}
output_idx_to_char = {idx: char for char, idx in output_char_to_idx.items()}

print(f"\nInput vocabulary size: {len(input_char_to_idx)}")
print(f"Output vocabulary size: {len(output_char_to_idx)}")

print(f"\nSample input chars: {sorted(list(input_chars))[:30]}")
print(f"Sample output chars: {list(output_chars)[:30]}")

# Save vocabularies
vocabs = {
    'input_char_to_idx': input_char_to_idx,
    'input_idx_to_char': input_idx_to_char,
    'output_char_to_idx': output_char_to_idx,
    'output_idx_to_char': output_idx_to_char,
    'input_vocab_size': len(input_char_to_idx),
    'output_vocab_size': len(output_char_to_idx)
}

with open('data/processed/mandarin_vocabularies.json', 'w', encoding='utf-8') as f:
    json.dump(vocabs, f, ensure_ascii=False, indent=2)

print("\n✓ Vocabularies saved to mandarin_vocabularies.json")