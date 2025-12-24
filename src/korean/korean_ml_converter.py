import torch
import torch.nn as nn
import json

# Define model architecture (must match training)
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, dropout=0.3):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, dropout=0.3):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        decoder_input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            decoder_input = output.argmax(2)
        
        return outputs

# Load model and vocabularies once (global)
_model = None
_input_char_to_idx = None
_output_idx_to_char = None
_device = None

def load_model():
    """Load the trained Korean seq2seq model (only once)"""
    global _model, _input_char_to_idx, _output_idx_to_char, _device
    
    if _model is None:
        # Load vocabularies
        with open('data/processed/korean_vocabularies.json', 'r', encoding='utf-8') as f:
            vocabs = json.load(f)
        
        _input_char_to_idx = vocabs['input_char_to_idx']
        _output_idx_to_char = vocabs['output_idx_to_char']
        input_vocab_size = vocabs['input_vocab_size']
        output_vocab_size = vocabs['output_vocab_size']
        
        # Load model
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('models/korean_seq2seq_model.pth', map_location=_device)
        
        embedding_dim = checkpoint['embedding_dim']
        hidden_dim = checkpoint['hidden_dim']
        
        encoder = Encoder(input_vocab_size, embedding_dim, hidden_dim)
        decoder = Decoder(output_vocab_size, embedding_dim, hidden_dim)
        _model = Seq2Seq(encoder, decoder, _device).to(_device)
        
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        _model.eval()
    
    return _model, _input_char_to_idx, _output_idx_to_char, _device

def translate_word(text, max_length=30):
    """Convert romanized Korean word to Hangul using ML"""
    model, input_char_to_idx, output_idx_to_char, device = load_model()
    output_char_to_idx = {}
    for k, v in output_idx_to_char.items():
        try:
            output_char_to_idx[v] = int(k)  # Convert string key to int
        except:
            output_char_to_idx[v] = k
    
    # Convert input to indices
    input_seq = [input_char_to_idx.get(c, input_char_to_idx.get('<PAD>', 0)) for c in text.lower()]
    
    # Pad
    while len(input_seq) < max_length:
        input_seq.append(input_char_to_idx['<PAD>'])
    input_seq = input_seq[:max_length]
    
    # Convert to tensor
    src = torch.tensor([input_seq], dtype=torch.long).to(device)
    
    # Decode
    with torch.no_grad():
        hidden, cell = model.encoder(src)
        
        trg_indices = [output_char_to_idx['<SOS>']]
        
        for _ in range(max_length):
            last_idx = int(trg_indices[-1]) if isinstance(trg_indices[-1], str) else trg_indices[-1]
            trg_tensor = torch.tensor([[last_idx]], dtype=torch.long).to(device)
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            pred_token = output.argmax(2).item()
            
            trg_indices.append(pred_token)
            
            # Check for EOS (make sure comparing ints)
            eos_idx = output_char_to_idx.get('<EOS>', -1)
            if pred_token == eos_idx:
                break
        
        # Convert to characters
        result = []
        for idx in trg_indices[1:]:
            if idx == output_char_to_idx['<EOS>']:
                break
            if idx == output_char_to_idx['<PAD>']:
                continue
            char = output_idx_to_char.get(str(idx), '')
            result.append(char)
        
        return ''.join(result)

def convert_korean_ml(text):
    """Convert Korean romanization to Hangul with hybrid ML + dictionary"""
    from mappings.korean_common_words import KOREAN_COMMON_WORDS
    
    # Process word by word
    words = text.split()
    result_words = []
    
    for word in words:
        # Remove punctuation for processing
        clean_word = ''.join(c for c in word if c.isalpha())
        
        if not clean_word:
            result_words.append(word)
            continue
        
        # Check dictionary first (case-insensitive)
        word_lower = clean_word.lower()
        if word_lower in KOREAN_COMMON_WORDS:
            result_words.append(KOREAN_COMMON_WORDS[word_lower])
        else:
            # Use ML for unknown words
            hangul = translate_word(clean_word)
            result_words.append(hangul)
    
    return ' '.join(result_words)