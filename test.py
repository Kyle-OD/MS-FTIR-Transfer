import pandas as pd
import glob
import os
from pathlib import Path

input_directory = "data/multimodal_spectroscopic_dataset_v2"

modalities = {
    'IR': 'ir_spectra',
    'MS/MS Positive 10ev': 'msms_cfmid_positive_10ev',
    'MS/MS Positive 20ev': 'msms_cfmid_positive_20ev',
    'MS/MS Positive 40ev': 'msms_cfmid_positive_40ev',
    'MS/MS Negative 10ev': 'msms_cfmid_negative_10ev',
    'MS/MS Negative 20ev': 'msms_cfmid_negative_20ev',
    'MS/MS Negative 40ev': 'msms_cfmid_negative_40ev',
}
modalities_2 = {
    'HSQC NMR': 'hsqc_nmr_spectrum',
    'H NMR': 'h_nmr_spectra',
    'C NMR': 'c_nmr_spectra',
    'MS/MS Iceberg Positive': 'msms_iceberg_positive',
    'MS/MS Scarf Positive': 'msms_scarf_positive'
}

parquet_files = sorted(glob.glob(os.path.join(input_directory, "aligned_chunk_*.parquet")))
total_files = len(parquet_files)

print("Total files:", total_files)

for (key, value) in modalities.items():
    print(key)
    df = pd.DataFrame(columns=['smiles', value])
    for file in parquet_files:
        temp = pd.read_parquet(file, columns=['smiles', value])
        print('.', end='')
        df = pd.concat([df, temp], axis=0)
    df.reset_index(drop=True).to_feather('data/'+value+'_v2.feather')
    print('')

for (key, value) in modalities_2.items():
    print(key)
    df = pd.DataFrame(columns=['smiles', value])
    for file in parquet_files:
        temp = pd.read_parquet(file, columns=['smiles', value])
        print('.', end='')
        pd.concat([df, temp], axis=0)
    df.reset_index(drop=True).to_feather('data/'+value+'_v2.feather')
    print('')