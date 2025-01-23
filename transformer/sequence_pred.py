import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os, json
import pandas as pd
from typing import List, Dict, Tuple, Union, Any
from datetime import datetime
from tqdm.notebook import tqdm # swap with line below if not using jupyter notebook
#from tqdm import tqdm

from transformer.io_funcs import load_seq2seq_from_meta, save_seq2seq_model_meta, init_checkpoint_folder
from transformer.evaluation import evaluate_model_seq2seq, evaluate_multimodal_beam_model

def train_model_seq2seq(model, train_loader, test_loader, optimizer, criterion_seq, num_epochs=50, evaluate=True, verbose=1, checkpoint_path=None, from_checkpoint=None, meta_tag=None, use_tensorboard=False):
    '''train a transformer encoder/decoder model for spectra to SMILES

    Args:
        model: Sequence to sequence transformer object.  Will be overwritten if continuing training from checkpoint
        train_loader: pytorch DataLoader for train object
        test_loader: pytorch DataLoader for test data
        optimizer: pytorch optimizer object
        criterion_seq: pytorch criterion or loss object
        num_epochs: number of epochs to train
        evaluate: boolean, whether or not to run evaluation on specified test_loader data
        verbose: [0,1,2] training verbosity field
            0: no metric reporting
            1: training and evaluation progress bars created for each epoch
            2: progress bars and evaluation metrics printed per epoch
        checkpoint_path: location to save checkpoint files.  If not specified, no checkpoints saved
        from_checkpoint: whether to restart training from latest checkpoint in checkpoint path
        meta_tag: text tag to include in meta.json file for user reference
        use_tensorboard: flag to initialize tensorboard metric tracking
    '''    
    device = next(model.parameters()).device
    history = {
        'train_loss': {},
        'test_accuracy': {},
        'test_loss': {},
        'valid_smiles_percentage': {},
        'tanimoto_similarity': {},
        'dice_similarity': {},
        'avg_edit_distance': {}
    }

    # TensorBoard setup
    if use_tensorboard:
        tb_log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S'))
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard logs will be saved to {tb_log_dir}")

    # Initialize checkpoint folder and load from checkpoint if specified
    if checkpoint_path is not None:
        if from_checkpoint:
            # Use the specified checkpoint folder
            checkpoint_folder = os.path.join(checkpoint_path, from_checkpoint)
            if os.path.exists(checkpoint_folder):
                meta_file = os.path.join(checkpoint_folder, "model_meta.json")
                if os.path.exists(meta_file):
                    model, optimizer, criterion_cls, criterion_seq, num_epochs = load_seq2seq_from_meta(meta_file)
                    model = model.to(device)
                
                # Load history
                    history_file = os.path.join(checkpoint_folder, "training_history.json")
                    if os.path.exists(history_file):
                        with open(history_file, 'r') as f:
                            history = json.load(f)
                
                checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith('.pth')]
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    checkpoint = torch.load(os.path.join(checkpoint_folder, latest_checkpoint))
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"Resuming from checkpoint: {latest_checkpoint}")
                else:
                    start_epoch = 0
                    print("No checkpoint file found in the specified folder. Starting from scratch.")
            else:
                raise ValueError(f"Specified checkpoint folder {from_checkpoint} does not exist.")
        else:
            # Create a new checkpoint folder
            checkpoint_folder = init_checkpoint_folder(checkpoint_path)
            start_epoch = 0
        
        if not from_checkpoint:
            save_seq2seq_model_meta(checkpoint_folder, model, optimizer, criterion_cls, criterion_seq, num_epochs, train_loader, test_loader, meta_tag)
    else:
        checkpoint_folder = None
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_seq_loss = 0
        
        if verbose >= 1:
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
    
        for x_batch, y_seq_batch in train_loader:
            x_batch, y_seq_batch = x_batch.to(device), y_seq_batch.to(device)
            x_batch = x_batch.squeeze(1)
            
            optimizer.zero_grad()
            
            smiles_output = model(x_batch, y_seq_batch[:, :-1])
            
            seq_loss = criterion_seq(smiles_output.reshape(-1, smiles_output.size(-1)), y_seq_batch[:, 1:].reshape(-1))
            
            seq_loss.backward()
            optimizer.step()
            
            total_seq_loss += seq_loss.item()

            if verbose >= 1:
                train_pbar.update(1)
                train_pbar.set_postfix({'train_loss': f'{seq_loss.item():.4f}'})
        
        avg_train_seq_loss = total_seq_loss / len(train_loader)
        history['train_loss'][epoch] = avg_train_seq_loss

        # Log training loss to TensorBoard
        if use_tensorboard:
            writer.add_scalar('train_loss', avg_train_seq_loss, epoch)

        if evaluate:
            '''if verbose >= 1:
                print(f'Epoch {epoch+1}/{num_epochs} [Eval]')'''
            eval_results = evaluate_model_seq2seq(model, test_loader, train_loader.dataset.smiles_vocab, verbose=verbose)
            
            for metric, value in eval_results.items():
                history[metric][epoch] = value
                if use_tensorboard:
                    writer.add_scalar(f'{metric}', value, epoch)
            
            if verbose == 2:
                print(f'Epoch {epoch+1}/{num_epochs}, '
                      f'Train Loss: {avg_train_seq_loss:.4f}, '
                      f'Test Loss: {eval_results["mean_test_loss"]:.4f}, '
                      f'Test Acc: {eval_results["test_accuracy"]:.4f}, '
                      f'Valid SMILES: {eval_results["valid_smiles_percentage"]:.2f}%, '
                      f'Tanimoto Sim: {eval_results["tanimoto_similarity"]:.4f}, '
                      f'Dice Sim: {eval_results["dice_similarity"]:.4f}, '
                      f'Edit Dist: {eval_results["avg_edit_distance"]:.2f}')
        
        if checkpoint_folder:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': history['train_loss'][epoch],
            }
            if evaluate:
                checkpoint['test_loss'] = history['test_loss'][epoch]
            torch.save(checkpoint, os.path.join(checkpoint_folder, f"checkpoint_epoch_{epoch+1}.pth"))

            # Save training history
            history_file = os.path.join(checkpoint_folder, "training_history.json")
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=4)
    # Close the TensorBoard writer
    if use_tensorboard:
        writer.close()

    return model, history

def train_multimodal_beam_model(
    model, 
    train_loader, 
    test_loader, 
    optimizer, 
    criterion, 
    num_epochs=50, 
    evaluate=True, 
    verbose=1, 
    checkpoint_path=None, 
    from_checkpoint=None, 
    meta_tag=None, 
    use_tensorboard=False
):
    '''Train a multimodal transformer model with beam search capabilities
    
    Args:
        model: MultimodalVITSeq2SeqBeam model instance
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer: PyTorch optimizer
        criterion: Loss criterion (typically CrossEntropyLoss)
        num_epochs: Number of training epochs
        evaluate: Whether to evaluate during training
        verbose: Verbosity level (0-2)
        checkpoint_path: Path to save checkpoints
        from_checkpoint: Resume training from checkpoint
        meta_tag: Metadata tag for logging
        use_tensorboard: Whether to use TensorBoard logging
    '''
    device = next(model.parameters()).device
    history = {
        'train_loss': {},
        'test_accuracy': {},
        'test_loss': {},
        'valid_smiles_percentage': {},
        'tanimoto_similarity': {},
        'dice_similarity': {},
        'avg_edit_distance': {}
    }

    # TensorBoard setup
    if use_tensorboard:
        tb_log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S'))
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard logs will be saved to {tb_log_dir}")

    # Initialize checkpoint folder and load from checkpoint if specified
    if checkpoint_path is not None:
        if from_checkpoint:
            checkpoint_folder = os.path.join(checkpoint_path, from_checkpoint)
            if os.path.exists(checkpoint_folder):
                # Load checkpoint
                checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith('.pth')]
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    checkpoint = torch.load(os.path.join(checkpoint_folder, latest_checkpoint))
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"Resuming from checkpoint: {latest_checkpoint}")
                else:
                    start_epoch = 0
            else:
                raise ValueError(f"Checkpoint folder {from_checkpoint} does not exist")
        else:
            checkpoint_folder = init_checkpoint_folder(checkpoint_path)
            start_epoch = 0
    else:
        checkpoint_folder = None
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        
        if verbose >= 1:
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        
        for inputs, y_seq_batch in train_loader:
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            y_seq_batch = y_seq_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, y_seq_batch[:, :-1])
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), y_seq_batch[:, 1:].reshape(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if verbose >= 1:
                train_pbar.update(1)
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'][epoch] = avg_train_loss
        
        if use_tensorboard:
            writer.add_scalar('train_loss', avg_train_loss, epoch)
        
        if evaluate:
            eval_results = evaluate_multimodal_beam_model(
                model, 
                test_loader, 
                train_loader.dataset.smiles_vocab,
                verbose=verbose
            )
            
            for metric, value in eval_results.items():
                history[metric][epoch] = value
                if use_tensorboard:
                    writer.add_scalar(metric, value, epoch)
            
            if verbose >= 2:
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'  Train Loss: {avg_train_loss:.4f}')
                print(f'  Test Loss: {eval_results["test_loss"]:.4f}')
                print(f'  Valid SMILES: {eval_results["valid_smiles_percentage"]:.2f}%')
                print(f'  Tanimoto Similarity: {eval_results["tanimoto_similarity"]:.4f}')
                print(f'  Dice Similarity: {eval_results["dice_similarity"]:.4f}')
                print(f'  Average Edit Distance: {eval_results["avg_edit_distance"]:.2f}')
        
        # Save checkpoint
        if checkpoint_folder:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': history['train_loss'][epoch],
            }
            if evaluate:
                checkpoint.update({k: history[k][epoch] for k in eval_results.keys()})
            
            torch.save(
                checkpoint,
                os.path.join(checkpoint_folder, f"checkpoint_epoch_{epoch+1}.pth")
            )
            
            # Save training history
            with open(os.path.join(checkpoint_folder, "training_history.json"), "w") as f:
                json.dump(history, f, indent=4)

    if use_tensorboard:
        writer.close()
    
    return model, history

def prepare_dataframes_for_prediction(
    data_dict: Dict[str, pd.DataFrame],
    spectrum_column: str = 'spectrum'
    ) -> List[Dict[str, Any]]:
    """
    Convert dictionary of dataframes to format required for prediction.
    
    Args:
        data_dict: Dictionary mapping modality type ('MS' or 'IR') to corresponding DataFrame
        spectrum_column: Name of column containing spectrum data
    
    Returns:
        List of dictionaries in format required by prepare_spectra_for_prediction
    """
    formatted_spectra = []
    
    # Process each modality's dataframe
    for modality, df in data_dict.items():
        if modality not in ['MS', 'IR']:
            raise ValueError(f"Invalid modality type: {modality}")
            
        # Convert each row to required format
        for _, row in df.iterrows():
            formatted_spectra.append({
                'type': modality,
                'data': row[spectrum_column]
            })
    
    return formatted_spectra

def prepare_spectra_for_prediction(
    spectra: List[Dict[str, Union[str, np.ndarray]]],
    tokenization_methods: Dict[str, str],
    max_values: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
    """
    Prepare a list of pre-binned spectra for prediction.
    
    Args:
        spectra: List of dictionaries containing spectral data
                Each dict should have keys 'type' ('MS' or 'IR') and 'data' (pre-binned numpy array)
        tokenization_methods: Dict mapping spectrum type to tokenization method
        max_values: Dict mapping spectrum type to maximum m/z value
    
    Returns:
        Dict mapping modality to batched tensor ready for model input
    """
    from transformer.tokenizers import tokenize_spectrum
    
    # Organize spectra by type
    organized_spectra = {'MS': [], 'IR': []}
    for spectrum in spectra:
        spec_type = spectrum['type']
        if spec_type not in ['MS', 'IR']:
            raise ValueError(f"Invalid spectrum type: {spec_type}")
        organized_spectra[spec_type].append(spectrum['data'])
    
    # Process each modality
    processed_inputs = {}
    for modality, spec_list in organized_spectra.items():
        if not spec_list:  # Skip if no spectra of this type
            continue
            
        # Convert to tokens (for pre-binned data, this mainly handles reshaping)
        tokenized_spectra = []
        for spec in spec_list:
            tokenized = tokenize_spectrum(
                spec,
                tokenization_methods[modality],
                max_values[modality],
                peaks_only=False  # Since data is already binned
            )
            tokenized_spectra.append(tokenized)
        
        # Stack into batch
        batch = torch.tensor(np.stack(tokenized_spectra), dtype=torch.float32).unsqueeze(1)
        processed_inputs[modality] = batch
    
    return processed_inputs

def predict_smiles_from_spectra(
    model: torch.nn.Module,
    spectra: List[Dict[str, Union[str, np.ndarray]]],
    tokenization_methods: Dict[str, str],
    max_values: Dict[str, int],
    smiles_vocab: Dict[str, int],
    beam_width: int = 5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> List[List[Tuple[str, float]]]:
    """
    Predict SMILES strings from spectral data using beam search.
    
    Args:
        model: Trained MultimodalVITSeq2SeqBeam model
        spectra: List of dictionaries containing spectral data
                Each dict should have keys 'type' ('MS' or 'IR') and 'data' (pre-binned numpy array)
        tokenization_methods: Dict mapping spectrum type to tokenization method
        max_values: Dict mapping spectrum type to maximum m/z value
        smiles_vocab: SMILES vocabulary dictionary
        beam_width: Number of top predictions to return
        device: Device to run model on
        
    Returns:
        List of lists, where each inner list contains (SMILES, score) tuples for the top predictions
    """
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
        beams = model.beam_search(inputs, beam_width=beam_width)
    
    # Decode predictions
    all_predictions = []
    for beam in beams:
        predictions = []
        for i in range(len(beam['sequences'])):
            tokens = beam['sequences'][i].tolist()
            smiles = ''.join([
                inv_smiles_vocab[token]
                for token in tokens
                if token not in [0, 1, 2]  # Exclude pad, sos, eos tokens
            ])
            score = beam['scores'][i].item()
            predictions.append((smiles, score))
        
        # Sort by score and take top beam_width predictions
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:beam_width]
        all_predictions.append(predictions)
    
    return all_predictions