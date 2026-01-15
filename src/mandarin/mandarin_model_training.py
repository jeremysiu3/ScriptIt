import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load vocabularies
with open('data/processed/mandarin_vocabularies.json', 'r', encoding='utf-8') as f:
    vocabs = json.load(f)

input_char_to_idx = vocabs['input_char_to_idx']
output_char_to_idx = vocabs['output_char_to_idx']
input_vocab_size = vocabs['input_vocab_size']
output_vocab_size = vocabs['output_vocab_size']

print(f"Input vocab: {input_vocab_size}")
print(f"Output vocab: {output_vocab_size}")

# Load training data
with open('data/processed/mandarin_combined_training_data.json', 'r', encoding='utf-8') as f:
    training_data = json.load(f)

print(f"Training examples: {len(training_data)}")

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.dropout(self.embedding(x))  # [batch, seq_len, embedding_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell  # Return final hidden states

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x, hidden, cell):
        # x: [batch, 1] - one character at a time
        embedded = self.dropout(self.embedding(x))  # [batch, 1, embedding_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output)  # [batch, 1, output_size]
        return prediction, hidden, cell

# Define Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch, src_len] - romanization
        # trg: [batch, trg_len] - chinese
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        
        # Store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode input sequence
        hidden, cell = self.encoder(src)
        
        # First input to decoder is <SOS>
        decoder_input = trg[:, 0].unsqueeze(1)  # [batch, 1]
        
        for t in range(1, trg_len):
            # Decode one step
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            
            # Teacher forcing: use real target as next input sometimes
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                decoder_input = trg[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(2)  # Use prediction
        
        return outputs

# Create models
embedding_dim = 128
hidden_dim = 256

encoder = Encoder(input_vocab_size, embedding_dim, hidden_dim)
decoder = Decoder(output_vocab_size, embedding_dim, hidden_dim)
model = Seq2Seq(encoder, decoder, device).to(device)

print(f"\n✓ Seq2Seq model created!")
print(f"Model architecture:")
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Prepare dataset
class ChineseDataset(Dataset):
    def __init__(self, data, input_char_to_idx, output_char_to_idx, max_input_len=25, max_output_len=10):
        self.data = data
        self.input_char_to_idx = input_char_to_idx
        self.output_char_to_idx = output_char_to_idx
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Convert input (romanization) to indices
        input_text = example['input'][:self.max_input_len]
        input_seq = [self.input_char_to_idx.get(c, self.input_char_to_idx['<PAD>']) for c in input_text]
        
        # Pad input to max_input_len
        while len(input_seq) < self.max_input_len:
            input_seq.append(self.input_char_to_idx['<PAD>'])
        
        # Convert output (Chinese) to indices with <SOS> and <EOS>
        output_text = example['chinese'][:self.max_output_len - 2]  # Leave room for SOS/EOS
        output_seq = [self.output_char_to_idx['<SOS>']]  # Start with SOS
        output_seq.extend([self.output_char_to_idx.get(c, self.output_char_to_idx['<PAD>']) for c in output_text])
        output_seq.append(self.output_char_to_idx['<EOS>'])  # End with EOS
        
        # Pad output to max_output_len
        while len(output_seq) < self.max_output_len:
            output_seq.append(self.output_char_to_idx['<PAD>'])
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(output_seq, dtype=torch.long)

# Create dataset and split
print("\nPreparing dataset...")
full_dataset = ChineseDataset(training_data, input_char_to_idx, output_char_to_idx)

# Split 80/20
split_idx = int(0.8 * len(full_dataset))
train_data = [full_dataset[i] for i in range(split_idx)]
val_data = [full_dataset[i] for i in range(split_idx, len(full_dataset))]

print(f"Training: {len(train_data)} examples")
print(f"Validation: {len(val_data)} examples")

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

print(f"✓ Data loaders created (batch size: {batch_size})")

# Training function
def train_seq2seq(model, train_loader, val_loader, num_epochs):
    """Train the seq2seq model"""
    
    # Create class weights - make EOS 3x more important
    weights = torch.ones(output_vocab_size)
    eos_idx = output_char_to_idx['<EOS>']
    if isinstance(eos_idx, str):
        eos_idx = int(eos_idx)
    weights[eos_idx] = 3.0  # EOS errors cost 3x more

    # Move weights to device
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=output_char_to_idx['<PAD>'],
        weight=weights
    )
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # Reduce when loss stops decreasing
        factor=0.5,           # Cut LR in half
        patience=2,           # Wait 2 epochs before reducing
    )
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for src, trg in train_loader:
            src = src.to(device)
            trg = trg.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, trg, teacher_forcing_ratio=0.5)
            
            # Reshape for loss calculation
            # output: [batch, trg_len, vocab_size]
            # trg: [batch, trg_len]
            output = output[:, 1:].reshape(-1, output.shape[-1])  # Skip first position (SOS)
            trg = trg[:, 1:].reshape(-1)  # Skip first position
            
            # Calculate loss
            loss = criterion(output, trg)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for src, trg in val_loader:
                src = src.to(device)
                trg = trg.to(device)

                output = model(src, trg, teacher_forcing_ratio=0)  # No teacher forcing for validation
                
                output = output[:, 1:].reshape(-1, output.shape[-1])
                trg = trg[:, 1:].reshape(-1)
                
                loss = criterion(output, trg)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update scheduler
        scheduler.step(avg_val_loss)

        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print()
    
    print(f"{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    
    return model

# Start training
trained_model = train_seq2seq(model, train_loader, val_loader, 40)

# Save model
torch.save({
    'encoder_state_dict': trained_model.encoder.state_dict(),
    'decoder_state_dict': trained_model.decoder.state_dict(),
    'input_vocab': input_char_to_idx,
    'output_vocab': output_char_to_idx,
    'input_vocab_size': input_vocab_size,
    'output_vocab_size': output_vocab_size,
    'embedding_dim': embedding_dim,
    'hidden_dim': hidden_dim
}, 'models/chinese_seq2seq_model.pth')

print("\n✓ Model saved to 'chinese_seq2seq_model.pth'")