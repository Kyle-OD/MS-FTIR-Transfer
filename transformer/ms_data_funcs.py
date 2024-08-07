import numpy as np
import ast

from evaluation import is_valid_smiles

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
    ### TODO
    spectrum = ast.literal_eval(spectrum_string)
    binned = np.zeros(max_mz+1)
    for mz, intensity in spectrum:
        continue
    pass

# Function to get a sample spectrum
def get_sample_spectrum(df):
    return df['spectrum'].iloc[0]

def remove_invalid_smiles(df):
    # Drop rows with invalid SMILES
    df['valid_smiles'] = df['SMILES'].apply(is_valid_smiles)
    df = df[df['valid_smiles']]
    df = df.drop('valid_smiles', axis=1)

    print(f"Shape after dropping invalid SMILES: {df.shape}")
    return df