import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import math
import os
import json
import rdkit
import pickle
from datetime import datetime
from tqdm.notebook import tqdm # swap with line below if not using jupyter notebook
#from tqdm import tqdm
from ms_data_funcs import *

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

class MS_VIT_Seq2Seq(nn.Module):
    def __init__(self, smiles_vocab_size, embed_depth=16, d_model=256, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, num_classes=None):
        super().__init__()
        self.d_model = d_model
        self.classification = num_classes is not None
        
        # Encoder
        self.embedding = nn.Linear(embed_depth, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Decoder
        self.smiles_embedding = nn.Embedding(smiles_vocab_size, d_model)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        
        # Output layers
        self.fc_smiles = nn.Linear(d_model, smiles_vocab_size)
        if self.classification:
            self.fc_classification = nn.Linear(d_model, num_classes)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.smiles_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_smiles.bias.data.zero_()
        self.fc_smiles.weight.data.uniform_(-initrange, initrange)
        if self.classification:
            self.fc_classification.bias.data.zero_()
            self.fc_classification.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt=None):
        # Encode input spectrum
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1))
        memory = self.transformer_encoder(src)
        
        # Decode SMILES
        tgt = self.smiles_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt.transpose(0, 1))
        output = self.transformer_decoder(tgt, memory)
        smiles_output = self.fc_smiles(output.transpose(0, 1))
        
        if self.classification:
            # Global average pooling and classification
            cls_output = memory.mean(dim=0)
            cls_output = self.fc_classification(cls_output)
            return smiles_output, cls_output
        else:
            return smiles_output

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
    
class SpectralSMILESDataset(Dataset):
    def __init__(self, df, tokenization_method, max_mz, smiles_tokenization='character'):
        self.spectra = df['spectrum']
        self.smiles = df['SMILES']
        self.tokenization_method = tokenization_method
        self.max_mz = max_mz
        self.smiles_tokenization = smiles_tokenization

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra.iloc[idx]
        smiles = self.smiles.iloc[idx]
        tokenized_spectrum = tokenize_spectrum(spectrum, self.tokenization_method, self.max_mz)
        
        if self.smiles_tokenization == 'character':
            tokenized_smiles = character_tokenization(smiles)
        elif self.smiles_tokenization == 'atom_wise':
            tokenized_smiles = atom_wise_tokenization(smiles)
        elif self.smiles_tokenization == 'substructure':
            tokenized_smiles = substructure_tokenization(smiles)
        
        return (torch.tensor(tokenized_spectrum, dtype=torch.float32).unsqueeze(0),
                torch.tensor([self.smiles_vocab[token] for token in tokenized_smiles], dtype=torch.long))

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

def save_model_meta(folder_path, model, optimizer, criterion, num_epochs, train_loader, test_loader, meta_tag):
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
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": meta_tag
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

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=50, evaluate=True, verbose=1, checkpoint_path=None, from_checkpoint=None, meta_tag=None):
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
            save_model_meta(checkpoint_folder, model, optimizer, criterion, num_epochs, train_loader, test_loader, meta_tag)
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

def train_model_seq2seq(model, train_loader, test_loader, optimizer, criterion_seq, num_epochs=50, criterion_cls=None, evaluate=True, verbose=1, checkpoint_path=None, from_checkpoint=None, meta_tag=None):
    device = next(model.parameters()).device
    history = {'seq_loss': {}, 'seq_accuracy': {}}
    if criterion_cls:
        history['cls_loss'] = {}
        history['cls_accuracy'] = {}

    # Initialize checkpoint folder and load from checkpoint if specified
    if checkpoint_path is not None:
        if from_checkpoint:
            # Use the specified checkpoint folder
            checkpoint_folder = os.path.join(checkpoint_path, from_checkpoint)
            if os.path.exists(checkpoint_folder):
                meta_file = os.path.join(checkpoint_folder, "model_meta.json")
                if os.path.exists(meta_file):
                    model, optimizer, criterion_cls, criterion_seq, num_epochs = load_model_from_meta(meta_file)
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
            save_model_meta(checkpoint_folder, model, optimizer, criterion_cls, criterion_seq, num_epochs, train_loader, test_loader, meta_tag)
    else:
        checkpoint_folder = None
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_seq_loss = 0
        if criterion_cls:
            total_cls_loss = 0
        
        if verbose == 1:
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
    
        for (x_batch, y_cls_batch), y_seq_batch in train_loader:
            x_batch, y_seq_batch = x_batch.to(device), y_seq_batch.to(device)
            if criterion_cls:
                y_cls_batch = y_cls_batch.to(device)
            x_batch = x_batch.squeeze(1)
            
            optimizer.zero_grad()
            
            if criterion_cls:
                smiles_output, cls_output = model(x_batch, y_seq_batch[:, :-1])
                cls_loss = criterion_cls(cls_output, y_cls_batch)
            else:
                smiles_output = model(x_batch, y_seq_batch[:, :-1])
            
            seq_loss = criterion_seq(smiles_output.reshape(-1, smiles_output.size(-1)), y_seq_batch[:, 1:].reshape(-1))
            
            if criterion_cls:
                loss = seq_loss + cls_loss
                total_cls_loss += cls_loss.item()
            else:
                loss = seq_loss
            
            loss.backward()
            optimizer.step()
            
            total_seq_loss += seq_loss.item()

            if verbose == 1:
                train_pbar.update(1)
                if criterion_cls:
                    train_pbar.set_postfix({'seq_loss': f'{seq_loss.item():.4f}', 'cls_loss': f'{cls_loss.item():.4f}'})
                else:
                    train_pbar.set_postfix({'seq_loss': f'{seq_loss.item():.4f}'})
        
        history['seq_loss'][epoch] = total_seq_loss / len(train_loader)
        if criterion_cls:
            history['cls_loss'][epoch] = total_cls_loss / len(train_loader)

        if evaluate:
            model.eval()
            seq_correct = 0
            seq_total = 0
            total_seq_loss = 0
            if criterion_cls:
                cls_correct = 0
                cls_total = 0
                total_cls_loss = 0
            
            if verbose == 1:
                test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]', leave=False)
            
            with torch.no_grad():
                for (x_batch, y_cls_batch), y_seq_batch in test_loader:
                    x_batch, y_seq_batch = x_batch.to(device), y_seq_batch.to(device)
                    if criterion_cls:
                        y_cls_batch = y_cls_batch.to(device)
                    x_batch = x_batch.squeeze(1)
                    
                    if criterion_cls:
                        seq_outputs, cls_outputs = model(x_batch, y_seq_batch[:, :-1])
                        cls_loss = criterion_cls(cls_outputs, y_cls_batch)
                        total_cls_loss += cls_loss.item()
                        _, cls_predicted = torch.max(cls_outputs.data, 1)
                        cls_total += y_cls_batch.size(0)
                        cls_correct += (cls_predicted == y_cls_batch).sum().item()
                    else:
                        seq_outputs = model(x_batch, y_seq_batch[:, :-1])
                    
                    seq_loss = criterion_seq(seq_outputs.reshape(-1, seq_outputs.size(-1)), y_seq_batch[:, 1:].reshape(-1))
                    total_seq_loss += seq_loss.item()
                    
                    _, seq_predicted = torch.max(seq_outputs.data, 2)
                    seq_total += y_seq_batch[:, 1:].numel()
                    seq_correct += (seq_predicted == y_seq_batch[:, 1:]).sum().item()
                    
                    if verbose == 1:
                        test_pbar.update(1)
                        if criterion_cls:
                            test_pbar.set_postfix({
                                'seq_acc': f'{seq_correct/seq_total:.4f}',
                                'cls_acc': f'{cls_correct/cls_total:.4f}'
                            })
                        else:
                            test_pbar.set_postfix({
                                'seq_acc': f'{seq_correct/seq_total:.4f}'
                            })
            
            seq_accuracy = seq_correct / seq_total
            history['seq_accuracy'][epoch] = seq_accuracy
            history['seq_loss'][epoch] = total_seq_loss / len(test_loader)
            
            if criterion_cls:
                cls_accuracy = cls_correct / cls_total
                history['cls_accuracy'][epoch] = cls_accuracy
                history['cls_loss'][epoch] = total_cls_loss / len(test_loader)
            
            if verbose == 2:
                print(f'Epoch {epoch+1}/{num_epochs}, '
                      f'Seq Loss: {history["seq_loss"][epoch]:.4f}, '
                      f'Seq Acc: {seq_accuracy:.4f}', end='')
                if criterion_cls:
                    print(f', Class Loss: {history["cls_loss"][epoch]:.4f}, '
                          f'Class Acc: {cls_accuracy:.4f}', end='')
                print()
        
        if checkpoint_folder:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'seq_loss': history['seq_loss'][epoch],
            }
            if evaluate:
                checkpoint['seq_accuracy'] = history['seq_accuracy'][epoch]
            if criterion_cls:
                checkpoint['cls_loss'] = history['cls_loss'][epoch]
                if evaluate:
                    checkpoint['cls_accuracy'] = history['cls_accuracy'][epoch]
            torch.save(checkpoint, os.path.join(checkpoint_folder, f"checkpoint_epoch_{epoch+1}.pth"))

    return model, history

def load_seq2seq_from_meta(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    model = MS_VIT_Seq2Seq(
        num_classes=meta['model']['num_classes'],
        smiles_vocab_size=meta['model']['smiles_vocab_size'],
        embed_depth=meta['model']['embed_depth'],
        d_model=meta['model']['d_model'],
        nhead=meta['model']['nhead'],
        num_layers=meta['model']['num_layers'],
        dim_feedforward=meta['model']['dim_feedforward'],
        dropout=meta['model']['dropout']
    )
    
    optimizer_class = getattr(torch.optim, meta['optimizer']['name'])
    optimizer = optimizer_class(model.parameters(), lr=meta['optimizer']['lr'], weight_decay=meta['optimizer']['weight_decay'])
    
    criterion_cls = getattr(torch.nn, meta['criterion_cls'])()
    criterion_seq = getattr(torch.nn, meta['criterion_seq'])()
    
    return model, optimizer, criterion_cls, criterion_seq, meta['training']['num_epochs']

def save_seq2seq_model_meta(folder_path, model, optimizer, criterion_cls, criterion_seq, num_epochs, train_loader, test_loader, meta_tag):
    meta = {
        "model": {
            "name": type(model).__name__,
            "num_classes": model.fc_classification.out_features,
            "smiles_vocab_size": model.fc_smiles.out_features,
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
        "criterion_cls": type(criterion_cls).__name__,
        "criterion_seq": type(criterion_seq).__name__,
        "training": {
            "num_epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "train_size": len(train_loader.dataset),
            "test_size": len(test_loader.dataset)
        },
        "data": {
            "input_shape": tuple(next(iter(train_loader))[0][0].shape),
            "tokenization_method": getattr(train_loader.dataset, 'tokenization_method', 'unknown'),
            "max_mz": getattr(train_loader.dataset, 'max_mz', 'unknown'),
            "smiles_tokenization": getattr(train_loader.dataset, 'smiles_tokenization', 'unknown')
        },
        "device": str(torch.cuda.get_device_name(0)),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": meta_tag
    }
    
    with open(os.path.join(folder_path, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

def create_smiles_vocab(smiles_list, tokenization='character'):
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    
    if len(smiles_list) == 0:
        return vocab  # Return basic vocabulary if list is empty
    
    unique_smiles = set(smiles_list)
    
    for smiles in unique_smiles:
        if tokenization == 'character':
            tokens = character_tokenization(smiles)
        elif tokenization == 'atom_wise':
            tokens = atom_wise_tokenization(smiles)
        elif tokenization == 'substructure':
            tokens = substructure_tokenization(smiles)
        else:
            raise ValueError(f"Unknown tokenization method: {tokenization}")
        
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    
    return vocab

def load_tokenized_data_with_smiles(X_train, y_train, X_test, y_test, method, smiles_vocab, max_mz=None, batch_size=32):
    if max_mz is None:
        max_mz = calculate_max_mz(X_train, 'spectrum')
    
    train_dataset = SpectralSMILESDataset(X_train, y_train, method, max_mz, smiles_vocab)
    test_dataset = SpectralSMILESDataset(X_test, y_test, method, max_mz, smiles_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader

def collate_fn(batch):
    spectra, cls_labels, smiles = zip(*batch)
    spectra = torch.stack(spectra)
    cls_labels = torch.tensor(cls_labels)
    
    # Pad SMILES sequences
    max_len = max(len(s) for s in smiles)
    padded_smiles = torch.zeros(len(smiles), max_len, dtype=torch.long)
    for i, s in enumerate(smiles):
        padded_smiles[i, :len(s)] = torch.tensor(s)
    
    return (spectra, cls_labels), padded_smiles

def evaluate_model_seq2seq(model, test_loader, smiles_vocab, criterion_cls=None):
    device = next(model.parameters()).device
    model.eval()
    seq_correct = 0
    seq_total = 0
    if criterion_cls:
        cls_correct = 0
        cls_total = 0
    
    inv_smiles_vocab = {v: k for k, v in smiles_vocab.items()}
    
    with torch.no_grad():
        for (x_batch, y_cls_batch), y_seq_batch in test_loader:
            x_batch, y_seq_batch = x_batch.to(device), y_seq_batch.to(device)
            if criterion_cls:
                y_cls_batch = y_cls_batch.to(device)
            x_batch = x_batch.squeeze(1)
            
            if criterion_cls:
                seq_outputs, cls_outputs = model(x_batch, y_seq_batch[:, :-1])
                _, cls_predicted = torch.max(cls_outputs.data, 1)
                cls_total += y_cls_batch.size(0)
                cls_correct += (cls_predicted == y_cls_batch).sum().item()
            else:
                seq_outputs = model(x_batch, y_seq_batch[:, :-1])
            
            _, seq_predicted = torch.max(seq_outputs.data, 2)
            seq_total += y_seq_batch[:, 1:].numel()
            seq_correct += (seq_predicted == y_seq_batch[:, 1:]).sum().item()
            
            # Print some example predictions
            for i in range(min(5, x_batch.size(0))):
                true_smiles = ''.join([inv_smiles_vocab[token.item()] for token in y_seq_batch[i] if token.item() != smiles_vocab['<pad>']])
                pred_smiles = ''.join([inv_smiles_vocab[token.item()] for token in seq_predicted[i] if token.item() != smiles_vocab['<pad>']])
                print(f"True SMILES: {true_smiles}")
                print(f"Pred SMILES: {pred_smiles}")
                print()
    
    seq_accuracy = seq_correct / seq_total
    print(f"Sequence Accuracy: {seq_accuracy:.4f}")
    
    if criterion_cls:
        cls_accuracy = cls_correct / cls_total
        print(f"Classification Accuracy: {cls_accuracy:.4f}")
        return seq_accuracy, cls_accuracy
    else:
        return seq_accuracy
    
def save_vocab(vocab, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def get_or_create_smiles_vocabs(df, vocab_dir='./vocabs', force_create=False):
    os.makedirs(vocab_dir, exist_ok=True)
    
    smiles_vocabs = {}
    tokenization_methods = ['character', 'atom_wise', 'substructure']
    
    for method in tokenization_methods:
        vocab_path = os.path.join(vocab_dir, f'smiles_vocab_{method}.pkl')
        
        if os.path.exists(vocab_path) and not force_create:
            print(f"Loading existing {method} vocabulary...")
            smiles_vocabs[method] = load_vocab(vocab_path)
        else:
            print(f"Creating new {method} vocabulary...")
            vocab = create_smiles_vocab(df['SMILES'].unique(), tokenization=method)
            save_vocab(vocab, vocab_path)
            smiles_vocabs[method] = vocab
        
        print(f"SMILES vocabulary size ({method}): {len(smiles_vocabs[method])}")
    
    return smiles_vocabs