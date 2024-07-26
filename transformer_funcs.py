import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import math
import os
import re
import json
import rdkit
import pickle
from datetime import datetime
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
import Levenshtein
from tqdm.notebook import tqdm # swap with line below if not using jupyter notebook
#from tqdm import tqdm
from ms_data_funcs import *

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

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
    def __init__(self, df, tokenization_method, max_mz, smiles_vocab):
        self.spectra = df['spectrum']
        self.smiles = df['SMILES']
        self.tokenization_method = tokenization_method
        self.max_mz = max_mz
        self.smiles_vocab = smiles_vocab

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra.iloc[idx]
        smiles = self.smiles.iloc[idx]
        tokenized_spectrum = tokenize_spectrum(spectrum, self.tokenization_method, self.max_mz)
        
        tokenized_smiles = [self.smiles_vocab['<sos>']]
        for token in smiles:
            tokenized_smiles.append(self.smiles_vocab.get(token, self.smiles_vocab['<unk>']))
        tokenized_smiles.append(self.smiles_vocab['<eos>'])
        
        return (torch.tensor(tokenized_spectrum, dtype=torch.float32).unsqueeze(0), 
                torch.tensor(tokenized_smiles, dtype=torch.long))

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
            print("Saving checkpoints to", folder_path)
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

def train_model_seq2seq(model, train_loader, test_loader, optimizer, criterion_seq, num_epochs=50, criterion_cls=None, evaluate=True, verbose=1, checkpoint_path=None, from_checkpoint=None, meta_tag=None, use_tensorboard=False):
    device = next(model.parameters()).device
    history = {
        'train_loss': {},
        'test_accuracy': {},
        'test_loss': {},
        'valid_smiles_percentage': {},
        'tanimoto_similarity': {},
        'dice_similarity': {},
        'avg_edit_distance': {}
    }

    # TensorBoard setup
    if use_tensorboard:
        tb_log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S'))
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard logs will be saved to {tb_log_dir}")

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
                
                # Load history
                    history_file = os.path.join(checkpoint_folder, "training_history.json")
                    if os.path.exists(history_file):
                        with open(history_file, 'r') as f:
                            history = json.load(f)
                
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
            save_seq2seq_model_meta(checkpoint_folder, model, optimizer, criterion_cls, criterion_seq, num_epochs, train_loader, test_loader, meta_tag)
    else:
        checkpoint_folder = None
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_seq_loss = 0
        
        if verbose == 1:
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
    
        for x_batch, y_seq_batch in train_loader:
            x_batch, y_seq_batch = x_batch.to(device), y_seq_batch.to(device)
            x_batch = x_batch.squeeze(1)
            
            optimizer.zero_grad()
            
            smiles_output = model(x_batch, y_seq_batch[:, :-1])
            
            seq_loss = criterion_seq(smiles_output.reshape(-1, smiles_output.size(-1)), y_seq_batch[:, 1:].reshape(-1))
            
            seq_loss.backward()
            optimizer.step()
            
            total_seq_loss += seq_loss.item()

            if verbose == 1:
                train_pbar.update(1)
                train_pbar.set_postfix({'train_loss': f'{seq_loss.item():.4f}'})
        
        avg_train_seq_loss = total_seq_loss / len(train_loader)
        history['train_loss'][epoch] = avg_train_seq_loss

        # Log training loss to TensorBoard
        if use_tensorboard:
            writer.add_scalar('train_loss', avg_train_seq_loss, epoch)

        if evaluate:
            '''if verbose == 1:
                print(f'Epoch {epoch+1}/{num_epochs} [Eval]')'''
            eval_results = evaluate_model_seq2seq(model, test_loader, train_loader.dataset.smiles_vocab, verbose=verbose)
            
            for metric, value in eval_results.items():
                history[metric][epoch] = value
                if use_tensorboard:
                    writer.add_scalar(f'{metric}', value, epoch)
            
            if verbose == 2:
                print(f'Epoch {epoch+1}/{num_epochs}, '
                      f'Train Loss: {avg_train_seq_loss:.4f}, '
                      f'Test Loss: {eval_results["mean_test_loss"]:.4f}, '
                      f'Test Acc: {eval_results["test_accuracy"]:.4f}, '
                      f'Valid SMILES: {eval_results["valid_smiles_percentage"]:.2f}%, '
                      f'Tanimoto Sim: {eval_results["tanimoto_similarity"]:.4f}, '
                      f'Dice Sim: {eval_results["dice_similarity"]:.4f}, '
                      f'Edit Dist: {eval_results["avg_edit_distance"]:.2f}')
        
        if checkpoint_folder:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': history['train_loss'][epoch],
            }
            if evaluate:
                checkpoint['test_loss'] = history['test_loss'][epoch]
            torch.save(checkpoint, os.path.join(checkpoint_folder, f"checkpoint_epoch_{epoch+1}.pth"))

            # Save training history
            history_file = os.path.join(checkpoint_folder, "training_history.json")
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=4)
    # Close the TensorBoard writer
    if use_tensorboard:
        writer.close()

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
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    
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

def load_tokenized_data_with_smiles(df_train, df_test, method, smiles_vocab, max_mz=None, batch_size=32):
    if max_mz is None:
        max_mz = calculate_max_mz(df_train, 'spectrum')
    
    train_dataset = SpectralSMILESDataset(df_train, method, max_mz, smiles_vocab)
    test_dataset = SpectralSMILESDataset(df_test, method, max_mz, smiles_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader

def collate_fn(batch):
    spectra, smiles = zip(*batch)
    spectra = torch.stack(spectra)
    
    # Pad SMILES sequences
    max_len = max(len(s) for s in smiles)
    padded_smiles = torch.zeros(len(smiles), max_len, dtype=torch.long)
    for i, s in enumerate(smiles):
        padded_smiles[i, :len(s)] = s.clone().detach()  # Changed this line
    
    return spectra, padded_smiles

def evaluate_model_seq2seq(model, test_loader, smiles_vocab, verbose=0, test=False):
    device = next(model.parameters()).device
    model.eval()
    seq_correct = 0
    seq_total = 0
    total_seq_loss = 0
    
    inv_smiles_vocab = {v: k for k, v in smiles_vocab.items()}
    
    criterion_seq = nn.CrossEntropyLoss(ignore_index=smiles_vocab['<pad>'])
    
    all_true_smiles = []
    all_pred_smiles = []

    if verbose == 1:
        pbar = tqdm(test_loader, desc='Evaluating', leave=False)
    
    with torch.no_grad():
        for x_batch, y_seq_batch in test_loader:
            x_batch, y_seq_batch = x_batch.to(device), y_seq_batch.to(device)
            x_batch = x_batch.squeeze(1)
            
            seq_outputs = model(x_batch, y_seq_batch[:, :-1])
            
            seq_loss = criterion_seq(seq_outputs.reshape(-1, seq_outputs.size(-1)), y_seq_batch[:, 1:].reshape(-1))
            total_seq_loss += seq_loss.item()
            
            _, seq_predicted = torch.max(seq_outputs.data, 2)
            seq_total += y_seq_batch[:, 1:].numel()
            seq_correct += (seq_predicted == y_seq_batch[:, 1:]).sum().item()
            
            for i in range(x_batch.size(0)):
                true_smiles = ''.join([inv_smiles_vocab[token.item()] for token in y_seq_batch[i] if token.item() not in [smiles_vocab['<pad>'], smiles_vocab['<sos>'], smiles_vocab['<eos>']]])
                pred_smiles = ''.join([inv_smiles_vocab[token.item()] for token in seq_predicted[i] if token.item() not in [smiles_vocab['<pad>'], smiles_vocab['<sos>'], smiles_vocab['<eos>']]])
                all_true_smiles.append(true_smiles)
                all_pred_smiles.append(pred_smiles)

            if verbose == 1:
                pbar.update(1)
                pbar.set_postfix({'Loss': f'{seq_loss.item():.4f}'})
    
    seq_accuracy = seq_correct / seq_total
    avg_seq_loss = total_seq_loss / len(test_loader)
    
    valid_smiles_percentage = calculate_valid_smiles_percentage(all_pred_smiles)
    tanimoto_similarity = calculate_tanimoto_similarity(all_true_smiles, all_pred_smiles)
    dice_similarity = calculate_dice_similarity(all_true_smiles, all_pred_smiles)
    avg_edit_distance = calculate_average_edit_distance(all_true_smiles, all_pred_smiles)
    
    if test:
        print(f"Sequence Accuracy: {seq_accuracy:.4f}")
        print(f"Average Sequence Loss: {avg_seq_loss:.4f}")
        print(f"Valid SMILES Percentage: {(100.0 * valid_smiles_percentage):.2f}%")
        print(f"Average Tanimoto Similarity: {tanimoto_similarity:.4f}")
        print(f"Average Dice Similarity: {dice_similarity:.4f}")
        print(f"Average Edit Distance: {avg_edit_distance:.2f}\n")
        # Print some example predictions
        for i in range(min(5, x_batch.size(0))):
            true_smiles = ''.join([inv_smiles_vocab[token.item()] for token in y_seq_batch[i] if token.item() not in [smiles_vocab['<pad>'], smiles_vocab['<sos>'], smiles_vocab['<eos>']]])
            pred_smiles = ''.join([inv_smiles_vocab[token.item()] for token in seq_predicted[i] if token.item() not in [smiles_vocab['<pad>'], smiles_vocab['<sos>'], smiles_vocab['<eos>']]])
            print(f"True SMILES: {true_smiles}")
            print(f"Pred SMILES: {pred_smiles}")
            print()
    
    return {
        'test_accuracy': seq_accuracy,
        'test_loss': avg_seq_loss,
        'valid_smiles_percentage': valid_smiles_percentage,
        'tanimoto_similarity': tanimoto_similarity,
        'dice_similarity': dice_similarity,
        'avg_edit_distance': avg_edit_distance
    }
    
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

def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def calculate_valid_smiles_percentage(predicted_smiles):
    valid_count = 0
    for smiles in predicted_smiles:
        if is_valid_smiles(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
            except:
                pass  # Ignore RDKit errors
    return (valid_count / len(predicted_smiles))

def calculate_tanimoto_similarity(true_smiles, pred_smiles):
    if len(true_smiles) != len(pred_smiles):
        return 0.0

    similarities = []
    for t, p in zip(true_smiles, pred_smiles):
        try:
            t_mol = Chem.MolFromSmiles(t)
            p_mol = Chem.MolFromSmiles(p)
            
            if t_mol is None:
                continue
            
            if p_mol is None:
                continue
            
            t_fp = AllChem.GetMorganFingerprintAsBitVect(t_mol, 2, nBits=2048)
            p_fp = AllChem.GetMorganFingerprintAsBitVect(p_mol, 2, nBits=2048)
            similarity = DataStructs.TanimotoSimilarity(t_fp, p_fp)
            similarities.append(similarity)
        except Exception as e:
            #print(f"Error processing SMILES pair: ({t}, {p}). Error: {str(e)}")
            continue

    return sum(similarities) / len(similarities) if similarities else 0.0

def calculate_dice_similarity(true_smiles, pred_smiles):
    if len(true_smiles) != len(pred_smiles):
        return 0.0

    similarities = []
    for t, p in zip(true_smiles, pred_smiles):
        try:
            t_mol = Chem.MolFromSmiles(t)
            p_mol = Chem.MolFromSmiles(p)
            
            if t_mol is None:
                continue
            
            if p_mol is None:
                continue
            
            t_fp = AllChem.GetMorganFingerprintAsBitVect(t_mol, 2, nBits=2048)
            p_fp = AllChem.GetMorganFingerprintAsBitVect(p_mol, 2, nBits=2048)
            similarity = DataStructs.DiceSimilarity(t_fp, p_fp)
            similarities.append(similarity)
        except Exception as e:
            #print(f"Error processing SMILES pair: ({t}, {p}). Error: {str(e)}")
            continue

    return sum(similarities) / len(similarities) if similarities else 0.0

def calculate_average_edit_distance(true_smiles, pred_smiles):
    distances = [Levenshtein.distance(t, p) for t, p in zip(true_smiles, pred_smiles)]
    return sum(distances) / len(distances)

def plot_training_history(history):
    metrics = ['train_loss', 'test_loss', 'test_accuracy', 'valid_smiles_percentage', 'tanimoto_similarity', 'dice_similarity', 'avg_edit_distance']
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
    fig.suptitle('Training History', fontsize=16)

    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        values = list(history[metric].values())
        epochs = list(history[metric].keys())
        
        ax.plot(epochs, values)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Fix y-axis scale
        if metric in ['test_accuracy', 'valid_smiles_percentage', 'tanimoto_similarity', 'dice_similarity']:
            ax.set_ylim(0, 1)
        elif metric == 'avg_edit_distance':
            ax.set_ylim(bottom=0)  # Start from 0, but let the upper limit be determined automatically

    # Remove the last empty subplot
    fig.delaxes(axes[3][1])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for the suptitle
    plt.show()