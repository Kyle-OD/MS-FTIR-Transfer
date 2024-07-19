import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import math
import os
import json
from datetime import datetime
from tqdm.notebook import tqdm # swap with line below if not using jupyter notebook
#from tqdm import tqdm
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

def init_checkpoint_folder(base_path):
    """Initialize a new numbered folder for checkpoints."""
    i = 1
    while True:
        folder_path = os.path.join(base_path, f"checkpoint_{i}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path
        i += 1

def save_model_meta(folder_path, model, optimizer, criterion, num_epochs, train_loader, test_loader):
    """Save comprehensive model metadata to a JSON file."""
    meta = {
        "model": {
            "name": type(model).__name__,
            "num_classes": model.fc.out_features,
            "embed_depth": model.embedding.in_features,
            "d_model": model.d_model,
            "nhead": model.transformer_encoder.layers[0].self_attn.num_heads,
            "num_layers": len(model.transformer_encoder.layers),
            "dim_feedforward": model.transformer_encoder.layers[0].linear1.out_features,
            "dropout": model.transformer_encoder.layers[0].dropout.p
        },
        "optimizer": {
            "name": type(optimizer).__name__,
            "lr": optimizer.param_groups[0]['lr'],
            "weight_decay": optimizer.param_groups[0].get('weight_decay', 0)
        },
        "criterion": type(criterion).__name__,
        "training": {
            "num_epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "train_size": len(train_loader.dataset),
            "test_size": len(test_loader.dataset)
        },
        "data": {
            "input_shape": tuple(next(iter(train_loader))[0].shape),
            "tokenization_method": getattr(train_loader.dataset, 'tokenization_method', 'unknown'),
            "max_mz": getattr(train_loader.dataset, 'max_mz', 'unknown')
        },
        "device": str(torch.cuda.get_device_name(0)),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(folder_path, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

def load_model_from_meta(meta_path):
    """Load model and recreate training setup from metadata."""
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    model = MS_VIT(
        num_classes=meta['model']['num_classes'],
        embed_depth=meta['model']['embed_depth'],
        d_model=meta['model']['d_model'],
        nhead=meta['model']['nhead'],
        num_layers=meta['model']['num_layers'],
        dim_feedforward=meta['model']['dim_feedforward'],
        dropout=meta['model']['dropout']
    )
    
    optimizer_class = getattr(torch.optim, meta['optimizer']['name'])
    optimizer = optimizer_class(model.parameters(), lr=meta['optimizer']['lr'], weight_decay=meta['optimizer']['weight_decay'])
    
    criterion = getattr(torch.nn, meta['criterion'])()
    
    return model, optimizer, criterion, meta['training']['num_epochs']

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=50, evaluate=True, verbose=1, checkpoint_path=None, from_checkpoint=None):
    device = next(model.parameters()).device
    if evaluate:
        history = {'loss':{}, 'accuracy':{}}
    else:
        history = {'loss':{}}

    # Initialize checkpoint folder and load from checkpoint if specified
    if checkpoint_path is not None:
        if from_checkpoint:
            # Use the specified checkpoint folder
            checkpoint_folder = os.path.join(checkpoint_path, from_checkpoint)
            if os.path.exists(checkpoint_folder):
                meta_file = os.path.join(checkpoint_folder, "model_meta.json")
                if os.path.exists(meta_file):
                    model, optimizer, criterion, num_epochs = load_model_from_meta(meta_file)
                    model = model.to(device)
                
                checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith('.pth')]
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    checkpoint = torch.load(os.path.join(checkpoint_folder, latest_checkpoint))
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"Resuming from checkpoint: {latest_checkpoint}")
                else:
                    start_epoch = 0
                    print("No checkpoint file found in the specified folder. Starting from scratch.")
            else:
                raise ValueError(f"Specified checkpoint folder {from_checkpoint} does not exist.")
        else:
            # Create a new checkpoint folder
            checkpoint_folder = init_checkpoint_folder(checkpoint_path)
            start_epoch = 0
        
        if not from_checkpoint:
            save_model_meta(checkpoint_folder, model, optimizer, criterion, num_epochs, train_loader, test_loader)
    else:
        checkpoint_folder = None
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
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
        
        history['loss'][epoch] = total_loss / len(train_loader)

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
            #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')
        else:
            #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')
            pass
        
        if checkpoint_folder:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': history['loss'][epoch],
            }
            if evaluate:
                checkpoint['accuracy'] = history['accuracy'][epoch]
            torch.save(checkpoint, os.path.join(checkpoint_folder, f"checkpoint_epoch_{epoch+1}.pth"))

    return model, history
