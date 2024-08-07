import torch
import torch.nn
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import math, os, json
from datetime import datetime
from rdkit import RDLogger
from tqdm.notebook import tqdm # swap with line below if not using jupyter notebook
#from tqdm import tqdm

from tokenizers import tokenize_spectrum, create_smiles_vocab
from ms_data_funcs import calculate_max_mz
from io_funcs import save_vocab, load_vocab

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

class PositionalEncoding(nn.Module):
    '''pytorch module for generating positional encoding values

    Args:
        d_model: dimensionality of the internal states of the model
        max_len: maximum number of tokens in context
    '''
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
    
class SpectralDataset(Dataset):
    '''pytorch dataset for loading mass spectrum data with class labels

    Args:
        df: pandas DataFrame to extract from
        labels: class labels
        tokenization_method: method used to tokenize ms spectrum data
        max_mz: the largest m/z value present in the dataset
    '''
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
    '''pytorch dataset for loading mass spectrum data with tokenized SMILES 

    Args:
        df: pandas DataFrame to extract from
        tokenization_method: method used to tokenize ms spectrum data
        max_mz: the largest m/z value present in the dataset
        smiles_vocab: vocab used with SMILES tokenization method
    '''
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
    '''create train and test DataLoaders 

    Args:
        X_train: training data spectrum
        y_train: training data classification
        X_test: test data spectrum
        y_test: test data classification
        method: tokenization method for spectral data
        max_mz: maximum m/z value if precalculated, else calculated within
        batch_size: chosen DataLoader batch size
    '''
    if max_mz is None:
        max_mz = calculate_max_mz(X_train, 'spectrum')
    train_dataset = SpectralDataset(X_train, y_train, method, max_mz)
    test_dataset = SpectralDataset(X_test, y_test, method, max_mz)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def load_tokenized_data_with_smiles(df_train, df_test, method, smiles_vocab, max_mz=None, batch_size=32):
    '''create train and test DataLoaders 

    Args:
        df_train: pandas DataFrame containing training data
        df_test: pandas DataFrame containing test data
        method: tokenization method for spectral data
        smiles_vocab: the SMILES vocabulary to use in dataset creation
        max_mz: maximum m/z value if precalculated, else calculated within
        batch_size: chosen DataLoader batch size
    '''
    if max_mz is None:
        max_mz = calculate_max_mz(df_train, 'spectrum')
    
    train_dataset = SpectralSMILESDataset(df_train, method, max_mz, smiles_vocab)
    test_dataset = SpectralSMILESDataset(df_test, method, max_mz, smiles_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader

def collate_fn(batch):
    '''collate and pad spectra and smiles from batch

    Args:
        batch: batch from pytorch DataLoader
    '''
    spectra, smiles = zip(*batch)
    spectra = torch.stack(spectra)
    
    # Pad SMILES sequences
    max_len = max(len(s) for s in smiles)
    padded_smiles = torch.zeros(len(smiles), max_len, dtype=torch.long)
    for i, s in enumerate(smiles):
        padded_smiles[i, :len(s)] = s.clone().detach()  # Changed this line
    
    return spectra, padded_smiles

def get_or_create_smiles_vocabs(df, vocab_dir='./vocabs', force_create=False):
    '''calculate the Dice similarity between the true and predicted SMILES values

    Args:
        df: pandas DataFrame with required 'SMILES' column
        vocab_dir: directory to save vocabularies
        force_create: boolean, force recreation rather than loading from a saved vocabulary
    '''
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