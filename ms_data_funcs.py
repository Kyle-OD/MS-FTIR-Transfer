import numpy as np
import matplotlib.pyplot as plt
import pywt
import ast
from rdkit import Chem

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

def calculate_max_mz(df, spectrum_column='spectrum'):
    def get_max_mz(spectrum_string):
        spectrum = ast.literal_eval(spectrum_string)
        return max(peak[0] for peak in spectrum)

    max_mz_series = df[spectrum_column].apply(get_max_mz)
    return int(np.ceil(max_mz_series.max()))

def bin_spectrum(spectrum_string, max_mz):
    spectrum = ast.literal_eval(spectrum_string)
    binned = np.zeros(max_mz + 1)  # +1 to include the max_mz value
    
    for mz, intensity in spectrum:
        mz_int = int(np.round(mz))
        if mz_int <= max_mz:
            binned[mz_int] += intensity
    
    return binned

def variable_density_bin_spectrum(spectrum_string, max_mz):
    spectrum = ast.literal_eval(spectrum_string)
    binned = np.zeros(max_mz+1)
    for mz, intensity in spectrum:
        continue
    pass

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

# Function to get a sample spectrum
def get_sample_spectrum(df):
    return df['spectrum'].iloc[0]

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

def is_valid_smiles(smiles):
    if not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def remove_invalid_smiles(df):
    # Drop rows with invalid SMILES
    df['valid_smiles'] = df['SMILES'].apply(is_valid_smiles)
    df = df[df['valid_smiles']]
    df = df.drop('valid_smiles', axis=1)

    print(f"Shape after dropping invalid SMILES: {df.shape}")
    return df
