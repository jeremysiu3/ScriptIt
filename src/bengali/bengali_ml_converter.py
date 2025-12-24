import torch
import torch.nn as nn

# Define model architecture (must match training)
class VowelLengthPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super(VowelLengthPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        predictions = self.fc(lstm_out)
        return predictions

# Load model once (global)
_model = None
_char_to_idx = None
_max_length = None

def load_model():
    """Load the trained Bengali model (only once)"""
    global _model, _char_to_idx, _max_length
    
    if _model is None:
        checkpoint = torch.load('models/bengali_vowel_model.pth', map_location=torch.device('cpu'))
        _char_to_idx = checkpoint['char_to_idx']
        _max_length = checkpoint['max_length']
        vocab_size = checkpoint['vocab_size']
        
        _model = VowelLengthPredictor(vocab_size)
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model.eval()
    
    return _model, _char_to_idx, _max_length

def predict_with_ml(text):
    """Use ML model to predict vowel lengths"""
    model, char_to_idx, max_length = load_model()
    vowels = set('aeiou')
    
    # Convert to sequence
    sequence = [char_to_idx.get(c, char_to_idx.get('<UNK>', 0)) for c in text.lower()]
    
    # Pad
    if len(sequence) < max_length:
        sequence = sequence + [char_to_idx['<PAD>']] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    
    # Predict
    input_tensor = torch.tensor([sequence], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = outputs.argmax(dim=2)[0]
    
    # Build result
    result = []
    for i, char in enumerate(text[:max_length]):
        if char.lower() in vowels:
            result.append(char.upper() if predictions[i].item() == 1 else char.lower())
        else:
            result.append(char)
    
    return ''.join(result)

def predict_vowels_hybrid(text):
    """Hybrid: dictionary first, then ML"""
    from mappings.bengali_common_words import BENGALI_COMMON_WORDS
    
    text_lower = text.lower().strip()
    
    # Check dictionary
    if text_lower in BENGALI_COMMON_WORDS:
        return BENGALI_COMMON_WORDS[text_lower]
    
    # Handle multiple words
    words = text.split()
    if len(words) > 1:
        result_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower in BENGALI_COMMON_WORDS:
                result_words.append(BENGALI_COMMON_WORDS[word_lower])
            else:
                result_words.append(predict_with_ml(word))
        return ' '.join(result_words)
    
    # Use ML
    return predict_with_ml(text)