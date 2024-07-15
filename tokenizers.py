from scipy.signal import find_peaks, cwt, ricker
import pywt
import numpy as np

import ast
import numpy as np
import matplotlib.pyplot as plt

# Direct 1D tokenization
def tokenize_spectrum(spectrum, token_width=50, num_levels=256):
    tokens = []
    for i in range(0, len(spectrum), token_width):
        segment = spectrum[i:i+token_width]
        token = np.mean(segment)  # or max, sum, etc.
        tokens.append(int(token * (num_levels - 1) / 100))
    return tokens

# Wavelet transform
def wavelet_tokenize(spectrum):
    coeffs = pywt.wavedec(spectrum, 'db1', level=5)
    return np.concatenate(coeffs)

# Continuous Wavelet transform
def cwt_tokenize(spectrum):
    widths = np.arange(1, 31)
    cwt_matrix = cwt(spectrum, ricker, widths)
    return cwt_matrix.flatten()



def peak_wavelet_tokenize(spectrum, bins, prominence=5, wavelet='db1', level=5):
    # Find peaks
    peaks, _ = find_peaks(spectrum, prominence=prominence)
    
    # Extract peak information
    peak_tokens = [(bins[p], spectrum[p]) for p in peaks]
    
    # Perform wavelet transform
    coeffs = pywt.wavedec(spectrum, wavelet, level=level)
    wavelet_tokens = np.concatenate(coeffs)
    
    # Combine peak and wavelet information
    combined_tokens = np.concatenate([np.array(peak_tokens).flatten(), wavelet_tokens])
    
    return combined_tokens