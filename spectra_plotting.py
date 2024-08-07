import numpy as np
import ast

import matplotlib.pyplot as plt

def count_spectrum_values(spectrum_string):
    try:
        spectrum_list = ast.literal_eval(spectrum_string)
        return len(spectrum_list)
    except:
        return 0
    
def plot_spectrum_length_histogram(df, spectrum_column, bin_size=2, yscale='linear'):
    # Count spectrum values for each row
    spectrum_counts = df[spectrum_column].apply(count_spectrum_values)

    # Create histogram
    plt.figure(figsize=(12, 6))
    plt.hist(spectrum_counts, bins=range(0, max(spectrum_counts) + bin_size, bin_size), 
             edgecolor='black')
    plt.xlabel('Number of Data Points')
    if yscale == 'log':
        plt.ylabel('Frequency (log10)')
    else:
        plt.ylabel('Frequency')
    plt.yscale(yscale)
    plt.title('Number of Data Points per Spectrum')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Print summary statistics
    print(f"Total spectra: {len(spectrum_counts)}")
    print(f"Min values: {min(spectrum_counts)}")
    print(f"Max values: {max(spectrum_counts)}")
    print(f"Mean values: {spectrum_counts.mean():.2f}")
    print(f"Median values: {spectrum_counts.median():.2f}")

    return spectrum_counts

def extract_mz_values(spectrum_string):
    try:
        spectrum_list = ast.literal_eval(spectrum_string)
        return [peak[0] for peak in spectrum_list]  # Extract m/z values
    except:
        return []  # Return empty list if there's an error parsing the string

def plot_mz_histogram(df, spectrum_column='spectrum', bin_size=2, min_mz=0, max_mz=None):
    # Extract all m/z values
    all_mz_values = []
    for spectrum in df[spectrum_column]:
        all_mz_values.extend(extract_mz_values(spectrum))

    # Determine max_mz if not provided
    if max_mz is None:
        max_mz = np.ceil(max(all_mz_values))

    # Create bins
    bins = np.arange(min_mz, max_mz + bin_size, bin_size)

    # Create histogram
    plt.figure(figsize=(15, 6))
    plt.hist(all_mz_values, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('m/z')
    plt.ylabel('Frequency')
    plt.title(f'Count of m/z Values Across Dataset (Bin Size: {bin_size})')
    plt.xlim(min_mz, max_mz)
    plt.xticks(np.arange(min_mz, max_mz + 1, 200))#max(20, bin_size * 10)))  # Adjust tick frequency
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Print summary statistics
    print(f"Total m/z values: {len(all_mz_values)}")
    print(f"Min m/z: {min(all_mz_values):.2f}")
    print(f"Max m/z: {max(all_mz_values):.2f}")
    print(f"Mean m/z: {np.mean(all_mz_values):.2f}")
    print(f"Median m/z: {np.median(all_mz_values):.2f}")

    return all_mz_values

def extract_mz_abundance_values(spectrum_string):
    try:
        spectrum_list = ast.literal_eval(spectrum_string)
        mz = [peak[0] for peak in spectrum_list]
        abundance = [peak[1] for peak in spectrum_list]
        return mz, abundance
    except:
        return []  # Return empty list if there's an error parsing the string

def plot_all_abundance(df, spectrum_column='spectrum', min_mz=0, max_mz=None):
    # Extract all m/z values
    all_mz = []
    all_abundance = []
    for spectrum in df[spectrum_column]:
        mz, abundance = extract_mz_abundance_values(spectrum)
        all_mz.extend(mz)
        all_abundance.extend(abundance)

    # Determine max_mz if not provided
    if max_mz is None:
        max_mz = np.ceil(max(all_mz))

    # Plotting abundance by m/z
    plt.figure(figsize=(15, 6))
    plt.scatter( all_mz, all_abundance)
    plt.xlabel('m/z')
    plt.ylabel('Relative Abundance')
    plt.title(f'')
    plt.xlim(min_mz, max_mz)
    #plt.xticks(np.arange(min_mz, max_mz + 1, max(20, bin_size * 10)))  # Adjust tick frequency
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Print summary statistics
    print(f"Total data points: {len(all_mz)}")
    print(f"Min m/z: {min(all_mz):.2f}")
    print(f"Max m/z: {max(all_mz):.2f}")
    print(f"Mean m/z: {np.mean(all_mz):.2f}")
    print(f"Median m/z: {np.median(all_mz):.2f}")

    print(f"Min abundance: {min(all_abundance):.2f}")
    print(f"Max abundance: {max(all_abundance):.2f}")
    print(f"Mean abundance: {np.mean(all_abundance):.2f}")
    print(f"Median abundance: {np.median(all_abundance):.2f}")

    return 