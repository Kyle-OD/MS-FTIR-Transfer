import numpy as np
import ast, pywt
import matplotlib.pyplot as plt
from rdkit import Chem

from ms_data_funcs import bin_spectrum

def direct_tokenization(binned_spectrum, window_size=16):
    # Pad the spectrum if necessary
    if len(binned_spectrum) % window_size != 0:
        pad_length = window_size - (len(binned_spectrum) % window_size)
        binned_spectrum = np.pad(binned_spectrum, (0, pad_length), mode='constant')
    
    # Reshape into 2D
    return binned_spectrum.reshape(-1, window_size)

def fourier_tokenization_2d(binned_spectrum, window_size=16):
    fft = np.fft.fft(binned_spectrum)
    magnitude_spectrum = np.abs(fft[:len(fft)//2])
    
    # Pad if necessary
    if len(magnitude_spectrum) % window_size != 0:
        pad_length = window_size - (len(magnitude_spectrum) % window_size)
        magnitude_spectrum = np.pad(magnitude_spectrum, (0, pad_length), mode='constant')
    
    # Reshape into 2D
    return magnitude_spectrum.reshape(-1, window_size)

def wavelet_tokenization_2d(binned_spectrum, window_size=16, wavelet='db1'):
    coeffs = pywt.wavedec(binned_spectrum, wavelet)
    flat_coeffs = np.concatenate(coeffs)
    
    # Pad if necessary
    if len(flat_coeffs) % window_size != 0:
        pad_length = window_size - (len(flat_coeffs) % window_size)
        flat_coeffs = np.pad(flat_coeffs, (0, pad_length), mode='constant')
    
    # Reshape into 2D
    return flat_coeffs.reshape(-1, window_size)

def peak_tokenization(spectrum_string, top_n=50, pad_to=50):
    spectrum = ast.literal_eval(spectrum_string)
    spectrum.sort(key=lambda x: x[1], reverse=True)
    top_peaks = spectrum[:top_n]
    flattened = [val for peak in top_peaks for val in peak]
    
    # Pad if necessary
    if len(flattened) < pad_to * 2:
        flattened.extend([0] * (pad_to * 2 - len(flattened)))
    
    return np.array(flattened).reshape(-1, 2)

def tokenize_spectrum(spectrum, method, max_mz, window_size=16):
    if isinstance(spectrum, str):
        binned_spectrum = bin_spectrum(spectrum, max_mz)
    else:
        binned_spectrum = spectrum

    if method == 'direct':
        return direct_tokenization(binned_spectrum, window_size)
    elif method == 'peak':
        return peak_tokenization(spectrum)
    elif method == 'fourier2':
        return fourier_tokenization_2d(binned_spectrum, window_size)
    elif method == 'wavelet2':
        return wavelet_tokenization_2d(binned_spectrum, window_size)
    else:
        raise ValueError(f"Unknown tokenization method: {method}")
    
def visualize_tokenization(spectrum, method, max_mz, window_size=16, path='./figures/tokenization/'):
    tokenized = tokenize_spectrum(spectrum, method, max_mz, window_size)
    
    plt.figure(figsize=(12, 6))
    
    if method == 'peak':
        plt.scatter(tokenized[:, 0], tokenized[:, 1])
        plt.xlabel('m/z')
        plt.ylabel('Intensity')
        plt.title(f'Peak Tokenization (Top {tokenized.shape[0]} peaks)')
    else:
        plt.imshow(tokenized, aspect='auto', interpolation='nearest')
        plt.colorbar(label='Intensity')
        plt.xlabel('Feature')
        plt.ylabel('Token')
        plt.title(f'{method.capitalize()} Tokenization')
    
    plt.tight_layout()
    plt.savefig(path+method+".png")
    plt.show()

def character_tokenization(smiles):
    return list(smiles)

def atom_wise_tokenization(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [atom.GetSymbol() for atom in mol.GetAtoms()]

def substructure_tokenization(smiles, max_length=10):
    tokens = []
    for i in range(len(smiles)):
        for j in range(1, max_length + 1):
            if i + j <= len(smiles):
                tokens.append(smiles[i:i+j])
    return list(set(tokens))

def create_smiles_vocab(smiles_list, tokenization='character'):
    '''create SMILES vocabulary object based on tokenization method and SMILES list 

    Args:
        smiles_list: List of SMILES strings
        tokenization: method for tokenizations.  currently implemented are:
            character
            atom_wise
            substructure
    '''
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