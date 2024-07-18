import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import math
from tqdm import tqdm
from ms_data_funcs import calculate_max_mz, tokenize_spectrum

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]
    
class MS_VIT(nn.Module):
    def __init__(self, num_classes, embed_depth=16, d_model=256, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # Initial embedding layer
        self.embedding = nn.Linear(embed_depth, d_model)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        # Final classification layer
        self.fc = nn.Linear(d_model, num_classes)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src shape: (batch_size, seq_length, 16)
        # Embed the input
        src = self.embedding(src) * math.sqrt(self.d_model)
        # Add positional encoding
        src = self.pos_encoder(src.transpose(0, 1))
        # Pass through transformer encoder
        output = self.transformer_encoder(src)
        # Global average pooling
        output = output.mean(dim=0)
        # Classification
        output = self.fc(output)
        
        return output

class SpectralDataset(Dataset):
    def __init__(self, df, labels, tokenization_method, max_mz):
        self.spectra = df['spectrum']
        self.labels = labels
        self.tokenization_method = tokenization_method
        self.max_mz = max_mz

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra.iloc[idx]
        label = self.labels[idx]
        tokenized = tokenize_spectrum(spectrum, self.tokenization_method, self.max_mz)
        return torch.tensor(tokenized, dtype=torch.float32).unsqueeze(0), label  # Add an extra dimension
    

def load_tokenized_data(X_train, y_train, X_test, y_test, method, max_mz=None, batch_size=32):
    if max_mz is None:
        max_mz = calculate_max_mz(X_train, 'spectrum')
    train_dataset = SpectralDataset(X_train, y_train, method, max_mz)
    test_dataset = SpectralDataset(X_test, y_test, method, max_mz)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def init_checkpoint_folder(path):

    return path

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=50, evaluate=True, verbose=1, from_checkpoint=False, checkpoint_path=None):
    device = next(model.parameters()).device
    if evaluate:
        history = {'loss':{}, 'accuracy':{}}
    else:
        history = {'loss':{}}
    checkpoint = False
    if checkpoint_path is not None:
        checkpoint_path = init_checkpoint_folder(checkpoint_path)
        checkpoint = True
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # progress bar if verbose
        if verbose == 1:
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
    
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Reshape input: (batch_size, 1, seq_length, 16) -> (batch_size, seq_length, 16)
            x_batch = x_batch.squeeze(1)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # Update progress bar description with current loss if verbose
            if verbose == 1:
                train_pbar.update(1)
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        history['loss'][epoch] = total_loss

        if evaluate:
            # Evaluate on test set
            model.eval()
            correct = 0
            total = 0

            # Create progress bar for evaluation if verbose
            if verbose == 1:
                test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]', leave=False)
            
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    x_batch = x_batch.squeeze(1)
                    outputs = model(x_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

                    # Update progress bar description with current accuracy if verbose
                    accuracy = correct / total
                    if verbose==1:
                        test_pbar.update(1)
                        test_pbar.set_postfix({'accuracy': f'{accuracy:.4f}'})
            
            accuracy = correct / total
            history['accuracy'][epoch] = accuracy
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')
        if checkpoint:
            torch.save(f"{checkpoint_path}_epoch_{epoch}.pth")
    return model, history

