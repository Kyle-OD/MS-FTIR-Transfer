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
    '''
    def __init__(self, data, smiles, tokenization_methods, max_values, smiles_vocab):
        self.data = data
        self.smiles = smiles
        self.tokenization_mehtods = tokenization_methods
        self.max_values = max_values
        self.smiles_vocab = smiles_vocab

    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        # Process each modality
        processed_inputs = {}
        for modality, df in self.data.items():
            spectrum = df.iloc[idx]['spectrum']
            tokenized = tokenize_spectrum(
                spectrum,
                self.tokenization_methods[modality],
                self.max_values[modality]
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

def decode_beam_search_results(beams, inv_smiles_vocab):
    '''Convert beam search results to SMILES strings
    
    Args:
        beams: beam search results from MS_VIT_Seq2Seq_Beam
        inv_smiles_vocab: inverse vocabulary mapping
    '''
    all_predictions = []
    
    for beam in beams:
        predictions = []
        for i in range(len(beam['sequences'])):
            tokens = beam['sequences'][i].tolist()
            smiles = ''.join([inv_smiles_vocab[token] for token in tokens 
                            if token not in [0, 1, 2]])  # Exclude pad, sos, eos
            score = beam['scores'][i].item()
            predictions.append((smiles, score))
        all_predictions.append(predictions)
    
    return all_predictions

def evaluate_model_seq2seq_beam(model, test_loader, smiles_vocab, beam_width=5, verbose=0):
    '''Evaluate sequence to sequence model with beam search
    
    Args:
        model: MS_VIT_Seq2Seq_Beam model
        test_loader: pytorch DataLoader for test data
        smiles_vocab: vocabulary used in encoding SMILES values
        beam_width: number of beams to maintain
        verbose: verbosity level (0-1)
    '''
    device = next(model.parameters()).device
    model.eval()
    
    inv_smiles_vocab = {v: k for k, v in smiles_vocab.items()}
    all_predictions = []
    all_true_smiles = []
    
    if verbose == 1:
        pbar = tqdm(test_loader, desc='Evaluating', leave=False)
    
    with torch.no_grad():
        for x_batch, y_seq_batch in test_loader:
            x_batch = x_batch.to(device)
            x_batch = x_batch.squeeze(1)
            
            # Get beam search predictions
            beams = model.beam_search(x_batch, beam_width=beam_width)
            batch_predictions = decode_beam_search_results(beams, inv_smiles_vocab)
            all_predictions.extend(batch_predictions)
            
            # Get true SMILES
            for seq in y_seq_batch:
                true_smiles = ''.join([inv_smiles_vocab[token.item()] 
                                     for token in seq 
                                     if token.item() not in [0, 1, 2]])
                all_true_smiles.append(true_smiles)
            
            if verbose == 1:
                pbar.update(1)
    
    return all_predictions, all_true_smiles