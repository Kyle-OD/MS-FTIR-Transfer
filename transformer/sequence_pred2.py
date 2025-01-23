import torch
import numpy as np
from rdkit import Chem
from typing import List, Dict, Union, Tuple

def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid."""
    if not smiles:  # Handle empty strings
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def normalize_scores(scores: List[float]) -> List[float]:
    """Convert log probabilities to normalized probabilities."""
    scores = np.array(scores)
    scores = np.exp(scores)  # Convert from log space
    return scores / np.sum(scores)

def predict_smiles_from_spectra(
    model: torch.nn.Module,
    spectra: List[Dict[str, Union[str, np.ndarray]]],
    tokenization_methods: Dict[str, str],
    max_values: Dict[str, int],
    smiles_vocab: Dict[str, int],
    beam_width: int = 5,
    max_length: int = 100,
    min_length: int = 3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> List[List[Tuple[str, float]]]:
    """
    Predict SMILES strings from spectral data using beam search with improved filtering.
    
    Args:
        model: Trained MultimodalVITSeq2SeqBeam model
        spectra: List of dictionaries containing spectral data
        tokenization_methods: Dict mapping spectrum type to tokenization method
        max_values: Dict mapping spectrum type to maximum m/z value
        smiles_vocab: SMILES vocabulary dictionary
        beam_width: Number of top predictions to return
        max_length: Maximum length of generated SMILES
        min_length: Minimum length of generated SMILES
        device: Device to run model on
        
    Returns:
        List of lists, where each inner list contains (SMILES, score) tuples for the top predictions
    """
    from transformer.tokenizers import tokenize_spectrum
    
    # Prepare model
    model = model.to(device)
    model.eval()
    
    # Create inverse vocabulary mapping
    inv_smiles_vocab = {v: k for k, v in smiles_vocab.items()}
    
    # Process input spectra
    inputs = prepare_spectra_for_prediction(spectra, tokenization_methods, max_values)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions using beam search
    with torch.no_grad():
        beams = model.beam_search(inputs, beam_width=beam_width * 2)  # Get more candidates for filtering
    
    # Decode and filter predictions
    all_predictions = []
    for beam in beams:
        predictions = []
        unfiltered_predictions = []
        
        for i in range(len(beam['sequences'])):
            tokens = beam['sequences'][i].tolist()
            
            # Convert tokens to SMILES
            smiles = ''.join([
                inv_smiles_vocab[token]
                for token in tokens
                if token not in [0, 1, 2]  # Exclude pad, sos, eos tokens
            ])
            
            score = beam['scores'][i].item()
            
            # Filter predictions
            if (len(smiles) >= min_length and 
                len(smiles) <= max_length and 
                not any(c * 5 in smiles for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ/\\') and  # Avoid repetitive patterns
                is_valid_smiles(smiles)):
                predictions.append((smiles, score))
            else:
                unfiltered_predictions.append((smiles, score))
        
        # If we don't have enough valid predictions, add some unfiltered ones
        if len(predictions) < beam_width and unfiltered_predictions:
            remaining = beam_width - len(predictions)
            predictions.extend(unfiltered_predictions[:remaining])
        
        # Sort by score and take top beam_width
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Normalize scores to probabilities
        if predictions:
            scores = [score for _, score in predictions]
            normalized_scores = normalize_scores(scores)
            predictions = [(smiles, score) for (smiles, _), score in zip(predictions, normalized_scores)]
        
        all_predictions.append(predictions)
    
    return all_predictions

def prepare_spectra_for_prediction(
    spectra: List[Dict[str, Union[str, np.ndarray]]],
    tokenization_methods: Dict[str, str],
    max_values: Dict[str, int]
) -> Dict[str, torch.Tensor]:
    """Prepare pre-binned spectra for prediction."""
    from transformer.tokenizers import tokenize_spectrum
    
    organized_spectra = {'MS': [], 'IR': []}
    for spectrum in spectra:
        spec_type = spectrum['type']
        if spec_type not in ['MS', 'IR']:
            raise ValueError(f"Invalid spectrum type: {spec_type}")
        organized_spectra[spec_type].append(spectrum['data'])
    
    processed_inputs = {}
    for modality, spec_list in organized_spectra.items():
        if not spec_list:
            continue
        
        tokenized_spectra = []
        for spec in spec_list:
            tokenized = tokenize_spectrum(
                spec,
                tokenization_methods[modality],
                max_values[modality],
                peaks_only=False
            )
            tokenized_spectra.append(tokenized)
        
        batch = torch.tensor(np.stack(tokenized_spectra), dtype=torch.float32).unsqueeze(1)
        processed_inputs[modality] = batch
    
    return processed_inputs

# Example usage:
'''
predictions = predict_smiles_from_spectra(
    model,
    spectra,
    tokenization_methods,
    max_values,
    smiles_vocab,
    beam_width=5,
    max_length=100,
    min_length=3
)

# Print results with probabilities
for i, preds in enumerate(predictions):
    print(f"\nPredictions for spectrum {i+1}:")
    for j, (smiles, prob) in enumerate(preds, 1):
        print(f"{j}. {smiles} (probability: {prob:.2%})")
'''