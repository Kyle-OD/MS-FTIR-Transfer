import torch
import torch.nn
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math, os
from rdkit import RDLogger
from tqdm.notebook import tqdm # swap with line below if not using jupyter notebook
#from tqdm import tqdm

from transformer.tokenizers import tokenize_spectrum
from transformer.vocabs import create_smiles_vocab
from transformer.data_funcs import calculate_max_mz
from transformer.evaluation import (
    calculate_average_edit_distance,
    calculate_dice_similarity,
    calculate_tanimoto_similarity,
    calculate_valid_smiles_percentage
)

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
    
class MultimodalSpectralSMILESDataset(torch.utils.data.Dataset):
    '''Dataset class for handling multiple spectral modalities
    Args:
        data: dict of dataframes for each modality 
        smiles: SMILES strings
        tokenization_methods: dict of tokenization methods for each modality
        max_values: dict of maximum values for each modality
        smiles_vocab: vocabulary for SMILES tokenization
        ms_peaks_only: boolean signifying whether MS data is full spectrum or only peaks
    '''
    def __init__(self, data, smiles, tokenization_methods, max_values, smiles_vocab):
        self.tokenization_methods = tokenization_methods
        self.max_values = max_values
        self.smiles_vocab = smiles_vocab

        # Verify all modalities have the same number of samples
        lengths = [len(df) for df in data.values()]
        if len(set(lengths)) > 1:
            min_length = min(lengths)
            print(f"Warning: Modalities have different lengths {lengths}. Truncating to shortest length: {min_length}")
            # Truncate all dataframes to the shortest length
            self.data = {modality: df.iloc[:min_length].reset_index(drop=True) 
                        for modality, df in data.items()}
            self.smiles = smiles.iloc[:min_length].reset_index(drop=True)
        else:
            self.data = data
            self.smiles = smiles
            
        # Double check lengths
        self.length = len(self.smiles)
        assert all(len(df) == self.length for df in self.data.values()), "Modality lengths don't match SMILES length"

    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.length}")
        # Process each modality
        processed_inputs = {}
        for modality, df in self.data.items():
            spectrum = df.iloc[idx]['spectrum']
            tokenized = tokenize_spectrum(
                spectrum,
                self.tokenization_methods[modality],
                self.max_values[modality],
            )
            processed_inputs[modality] = torch.tensor(
                tokenized,
                dtype=torch.float32
            ).unsqueeze(0)
        
        # Process SMILES
        smiles = self.smiles.iloc[idx]
        tokenized_smiles = [self.smiles_vocab['<sos>']]
        for token in smiles:
            tokenized_smiles.append(
                self.smiles_vocab.get(token, self.smiles_vocab['<unk>'])
            )
        tokenized_smiles.append(self.smiles_vocab['<eos>'])
        
        return processed_inputs, torch.tensor(tokenized_smiles, dtype=torch.long)
    
    
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

def load_tokenized_multimodal_data_with_smiles(
    data_dict,
    smiles_series,
    tokenization_methods,
    smiles_vocab,
    max_values=None,
    batch_size=32,
):
    '''Create train and test DataLoaders for multimodal spectral data with SMILES
    
    Args:
        data_dict: Dictionary of DataFrames for each modality, e.g.,
            {
                'MS': ms_df_train,  # DataFrame with 'spectrum' column
                'IR': ir_df_train   # DataFrame with 'spectrum' column
            }
        smiles_series: Pandas Series containing SMILES strings
        tokenization_methods: Dictionary of tokenization methods for each modality, e.g.,
            {
                'MS': 'direct',
                'IR': 'fourier2'
            }
        smiles_vocab: Vocabulary dictionary for SMILES tokenization
        max_values: Dictionary of maximum values for each modality (optional)
            If not provided, will be calculated from data
        batch_size: Batch size for DataLoader
        ms_peaks_only: boolean signifying whether MS data is full spectrum or only peaks
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    '''
    # Calculate max values for each modality if not provided
    if max_values is None:
        max_values = {}
        for modality, df in data_dict.items():
            if 'spectrum' in df.columns:
                max_values[modality] = calculate_max_mz(df, 'spectrum')
            else:
                raise ValueError(f"DataFrame for modality {modality} must contain 'spectrum' column")
    
    # Create dataset
    dataset = MultimodalSpectralSMILESDataset(
        data=data_dict,
        smiles=smiles_series,
        tokenization_methods=tokenization_methods,
        max_values=max_values,
        smiles_vocab=smiles_vocab,
    )
    
    # Create DataLoader with multimodal collate function
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multimodal_collate_fn
    )
    
    return loader

def split_and_load_tokenized_multimodal_data(
    train_data_dict,
    train_smiles,
    test_data_dict,
    test_smiles,
    tokenization_methods,
    smiles_vocab,
    max_values=None,
    batch_size=32,
):
    '''Create separate train and test DataLoaders for multimodal spectral data
    
    Args:
        train_data_dict: Dictionary of training DataFrames for each modality
        train_smiles: Pandas Series containing training SMILES strings
        test_data_dict: Dictionary of test DataFrames for each modality
        test_smiles: Pandas Series containing test SMILES strings
        tokenization_methods: Dictionary of tokenization methods for each modality
        smiles_vocab: Vocabulary dictionary for SMILES tokenization
        max_values: Dictionary of maximum values for each modality (optional)
        batch_size: Batch size for DataLoader
        ms_peaks_only: boolean signifying whether MS data is full spectrum or only peaks
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    '''
    # Calculate max values once if not provided
    if max_values is None:
        max_values = {}
        # Use training data to calculate max values
        for modality, df in train_data_dict.items():
            if 'spectrum' in df.columns:
                max_values[modality] = calculate_max_mz(df, 'spectrum')
            else:
                raise ValueError(f"DataFrame for modality {modality} must contain 'spectrum' column")
    
    # Create train loader
    train_loader = load_tokenized_multimodal_data_with_smiles(
        data_dict=train_data_dict,
        smiles_series=train_smiles,
        tokenization_methods=tokenization_methods,
        smiles_vocab=smiles_vocab,
        max_values=max_values,
        batch_size=batch_size,
    )
    
    # Create test loader
    test_loader = load_tokenized_multimodal_data_with_smiles(
        data_dict=test_data_dict,
        smiles_series=test_smiles,
        tokenization_methods=tokenization_methods,
        smiles_vocab=smiles_vocab,
        max_values=max_values,  # Use same max values as training
        batch_size=batch_size
    )
    
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

def multimodal_collate_fn(batch):
    '''Collate function for multimodal batches'''
    # Separate inputs and targets
    inputs_list, smiles_list = zip(*batch)
    
    # Process each modality
    processed_inputs = {}
    for modality in inputs_list[0].keys():
        modality_tensors = [sample[modality] for sample in inputs_list]
        processed_inputs[modality] = torch.stack(modality_tensors)
    
    # Pad SMILES sequences
    max_len = max(len(s) for s in smiles_list)
    padded_smiles = torch.zeros(len(smiles_list), max_len, dtype=torch.long)
    for i, s in enumerate(smiles_list):
        padded_smiles[i, :len(s)] = s
    
    return processed_inputs, padded_smiles
