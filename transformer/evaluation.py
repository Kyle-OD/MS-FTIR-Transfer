import torch
from torch import nn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import Levenshtein
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm # swap with line below if not using jupyter notebook
#from tqdm import tqdm

def evaluate_model_seq2seq(model, test_loader, smiles_vocab, verbose=0, test=False):
    '''code to run evaluation of a seq2seq transformer model

    Args:
        model: seq2seq model for evaluation
        test_loader: pytorch DataLoader containing test data
        smiles_vocab: vocabulary used in encoding SMILES values
        verbose: [0,1] if 1, use tqdm ptogress bars
        test: boolean, use during testing to print results
    '''
    device = next(model.parameters()).device
    model.eval()
    seq_correct = 0
    seq_total = 0
    total_seq_loss = 0
    
    inv_smiles_vocab = {v: k for k, v in smiles_vocab.items()}
    
    criterion_seq = nn.CrossEntropyLoss(ignore_index=smiles_vocab['<pad>'])
    
    all_true_smiles = []
    all_pred_smiles = []

    if verbose == 1:
        pbar = tqdm(test_loader, desc='Evaluating', leave=False)
    
    with torch.no_grad():
        for x_batch, y_seq_batch in test_loader:
            x_batch, y_seq_batch = x_batch.to(device), y_seq_batch.to(device)
            x_batch = x_batch.squeeze(1)
            
            seq_outputs = model(x_batch, y_seq_batch[:, :-1])
            
            seq_loss = criterion_seq(seq_outputs.reshape(-1, seq_outputs.size(-1)), y_seq_batch[:, 1:].reshape(-1))
            total_seq_loss += seq_loss.item()
            
            _, seq_predicted = torch.max(seq_outputs.data, 2)
            seq_total += y_seq_batch[:, 1:].numel()
            seq_correct += (seq_predicted == y_seq_batch[:, 1:]).sum().item()
            
            for i in range(x_batch.size(0)):
                true_smiles = ''.join([inv_smiles_vocab[token.item()] for token in y_seq_batch[i] if token.item() not in [smiles_vocab['<pad>'], smiles_vocab['<sos>'], smiles_vocab['<eos>']]])
                pred_smiles = ''.join([inv_smiles_vocab[token.item()] for token in seq_predicted[i] if token.item() not in [smiles_vocab['<pad>'], smiles_vocab['<sos>'], smiles_vocab['<eos>']]])
                all_true_smiles.append(true_smiles)
                all_pred_smiles.append(pred_smiles)

            if verbose == 1:
                pbar.update(1)
                pbar.set_postfix({'Loss': f'{seq_loss.item():.4f}'})
    
    seq_accuracy = seq_correct / seq_total
    avg_seq_loss = total_seq_loss / len(test_loader)
    
    valid_smiles_percentage = calculate_valid_smiles_percentage(all_pred_smiles)
    tanimoto_similarity = calculate_tanimoto_similarity(all_true_smiles, all_pred_smiles)
    dice_similarity = calculate_dice_similarity(all_true_smiles, all_pred_smiles)
    avg_edit_distance = calculate_average_edit_distance(all_true_smiles, all_pred_smiles)
    
    if test:
        print(f"Sequence Accuracy: {seq_accuracy:.4f}")
        print(f"Average Sequence Loss: {avg_seq_loss:.4f}")
        print(f"Valid SMILES Percentage: {(100.0 * valid_smiles_percentage):.2f}%")
        print(f"Average Tanimoto Similarity: {tanimoto_similarity:.4f}")
        print(f"Average Dice Similarity: {dice_similarity:.4f}")
        print(f"Average Edit Distance: {avg_edit_distance:.2f}\n")
        # Print some example predictions
        for i in range(min(5, x_batch.size(0))):
            true_smiles = ''.join([inv_smiles_vocab[token.item()] for token in y_seq_batch[i] if token.item() not in [smiles_vocab['<pad>'], smiles_vocab['<sos>'], smiles_vocab['<eos>']]])
            pred_smiles = ''.join([inv_smiles_vocab[token.item()] for token in seq_predicted[i] if token.item() not in [smiles_vocab['<pad>'], smiles_vocab['<sos>'], smiles_vocab['<eos>']]])
            print(f"True SMILES: {true_smiles}")
            print(f"Pred SMILES: {pred_smiles}")
            print()
    
    return {
        'test_accuracy': seq_accuracy,
        'test_loss': avg_seq_loss,
        'valid_smiles_percentage': valid_smiles_percentage,
        'tanimoto_similarity': tanimoto_similarity,
        'dice_similarity': dice_similarity,
        'avg_edit_distance': avg_edit_distance
    }

def is_valid_smiles(smiles):
    '''check if SMILES string corresponds to valid molecule

    Args:
        smiles: a SMILES string
    '''
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def calculate_valid_smiles_percentage(predicted_smiles):
    '''calculate the percentage of valid SMILES strings

    Args:
        predicted_smiles: a list of SMILES strings
    '''
    valid_count = 0
    for smiles in predicted_smiles:
        if is_valid_smiles(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
            except:
                pass  # Ignore RDKit errors
    return (valid_count / len(predicted_smiles))

def calculate_tanimoto_similarity(true_smiles, pred_smiles):
    '''calculate the Tanimoto similarity between the true and predicted SMILES values

    Args:
        true_smiles: either one or a list of true SMILES strings
        pred_smiles: either one or a list of predicted SMILES strings
    '''
    if len(true_smiles) != len(pred_smiles):
        return 0.0

    similarities = []
    for t, p in zip(true_smiles, pred_smiles):
        try:
            t_mol = Chem.MolFromSmiles(t)
            p_mol = Chem.MolFromSmiles(p)
            
            if t_mol is None:
                continue
            
            if p_mol is None:
                continue
            
            t_fp = AllChem.GetMorganFingerprintAsBitVect(t_mol, 2, nBits=2048)
            p_fp = AllChem.GetMorganFingerprintAsBitVect(p_mol, 2, nBits=2048)
            similarity = DataStructs.TanimotoSimilarity(t_fp, p_fp)
            similarities.append(similarity)
        except Exception as e:
            #print(f"Error processing SMILES pair: ({t}, {p}). Error: {str(e)}")
            continue

    return sum(similarities) / len(similarities) if similarities else 0.0

def calculate_dice_similarity(true_smiles, pred_smiles):
    '''calculate the Dice similarity between the true and predicted SMILES values

    Args:
        true_smiles: either one or a list of true SMILES strings
        pred_smiles: either one or a list of predicted SMILES strings
    '''
    if len(true_smiles) != len(pred_smiles):
        return 0.0

    similarities = []
    for t, p in zip(true_smiles, pred_smiles):
        try:
            t_mol = Chem.MolFromSmiles(t)
            p_mol = Chem.MolFromSmiles(p)
            
            if t_mol is None:
                continue
            
            if p_mol is None:
                continue
            
            t_fp = AllChem.GetMorganFingerprintAsBitVect(t_mol, 2, nBits=2048)
            p_fp = AllChem.GetMorganFingerprintAsBitVect(p_mol, 2, nBits=2048)
            similarity = DataStructs.DiceSimilarity(t_fp, p_fp)
            similarities.append(similarity)
        except Exception as e:
            #print(f"Error processing SMILES pair: ({t}, {p}). Error: {str(e)}")
            continue

    return sum(similarities) / len(similarities) if similarities else 0.0

def calculate_average_edit_distance(true_smiles, pred_smiles):
    distances = [Levenshtein.distance(t, p) for t, p in zip(true_smiles, pred_smiles)]
    return sum(distances) / len(distances)

def plot_training_history(history):
    '''plot training history using relevant metrics

    Args:
        history: json object containing training history of model
    '''
    metrics = ['train_loss', 'test_loss', 'test_accuracy', 'valid_smiles_percentage', 'tanimoto_similarity', 'dice_similarity', 'avg_edit_distance']
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
    fig.suptitle('Training History', fontsize=16)

    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        values = list(history[metric].values())
        epochs = list(history[metric].keys())
        
        ax.plot(epochs, values)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Fix y-axis scale
        if metric in ['test_accuracy', 'valid_smiles_percentage', 'tanimoto_similarity', 'dice_similarity']:
            ax.set_ylim(0, 1)
        elif metric == 'avg_edit_distance':
            ax.set_ylim(bottom=0)  # Start from 0, but let the upper limit be determined automatically

    # Remove the last empty subplot
    fig.delaxes(axes[3][1])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for the suptitle
    plt.show()