import json

# Load conversational data
with open('data/processed/mandarin_convo_data.json', 'r', encoding='utf-8') as f:
    conv_data = json.load(f)

# Load filtered news data (2000 chars)
with open('data/processed/mandarin_news_data_filtered.json', 'r', encoding='utf-8') as f:
    news_data = json.load(f)

print(f"Conversational: {len(conv_data)} examples")
print(f"News: {len(news_data)} examples")

# Combine them - put conversational FIRST so it's prioritized
combined_data = conv_data + news_data

print(f"\n✓ Combined total: {len(combined_data)} examples")

# Save
with open('data/processed/mandarin_combined_training_data.json', 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=2)

print("✓ Saved to mandarin_combined_training_data.json")