import torch
from torch import nn
import math, os
from tqdm.notebook import tqdm # swap with line below if not using jupyter notebook
#from tqdm import tqdm

from io_funcs import save_model_meta, load_model_from_meta, init_checkpoint_folder
from transformer_utils import PositionalEncoding

class MS_VIT(nn.Module):
    '''pytorch module classifying from mass spectral input

    Args:
        num_classes: number of classes present in classification set
        embed_depth: token depth
        d_model: dimensionality of the internal states of the model
        n_head: number of attention heads
        num_layers: number of transformer encoder layers
        dim_feedforward: dimensionality of feedforward classifier network
        dropout: dropout percentage applied to entire model
    '''
    def __init__(self, num_classes, embed_depth=16, d_model=256, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # Initial embedding layer
        self.embedding = nn.Linear(embed_depth, d_model)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        # Final classification layer
        self.fc = nn.Linear(d_model, num_classes)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src shape: (batch_size, seq_length, 16)
        # Embed the input
        src = self.embedding(src) * math.sqrt(self.d_model)
        # Add positional encoding
        src = self.pos_encoder(src.transpose(0, 1))
        # Pass through transformer encoder
        output = self.transformer_encoder(src)
        # Global average pooling
        output = output.mean(dim=0)
        # Classification
        output = self.fc(output)
        
        return output
    
def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=50, evaluate=True, verbose=1, checkpoint_path=None, from_checkpoint=None, meta_tag=None):
    '''train a transformer encoder/decoder model for spectra to classification

    Args:
        model: Sequence to sequence transformer object.  Will be overwritten if continuing training from checkpoint
        train_loader: pytorch DataLoader for train object
        test_loader: pytorch DataLoader for test data
        optimizer: pytorch optimizer object
        criterion: pytorch criterion or loss object
        evaluate: boolean, whether or not to run evaluation on specified test_loader data
        verbose: [0,1] training verbosity field
            0: no metric reporting
            1: training and evaluation progress bars created for each epoch
        checkpoint_path: location to save checkpoint files.  If not specified, no checkpoints saved
        from_checkpoint: whether to restart training from latest checkpoint in checkpoint path
        meta_tag: text tag to include in meta.json file for user reference
    '''   
    device = next(model.parameters()).device
    if evaluate:
        history = {'loss':{}, 'accuracy':{}}
    else:
        history = {'loss':{}}

    # Initialize checkpoint folder and load from checkpoint if specified
    if checkpoint_path is not None:
        if from_checkpoint:
            # Use the specified checkpoint folder
            checkpoint_folder = os.path.join(checkpoint_path, from_checkpoint)
            if os.path.exists(checkpoint_folder):
                meta_file = os.path.join(checkpoint_folder, "model_meta.json")
                if os.path.exists(meta_file):
                    model, optimizer, criterion, num_epochs = load_model_from_meta(meta_file)
                    model = model.to(device)
                
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
            save_model_meta(checkpoint_folder, model, optimizer, criterion, num_epochs, train_loader, test_loader, meta_tag)
    else:
        checkpoint_folder = None
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        
        # progress bar if verbose
        if verbose == 1:
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
    
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Reshape input: (batch_size, 1, seq_length, 16) -> (batch_size, seq_length, 16)
            x_batch = x_batch.squeeze(1)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # Update progress bar description with current loss if verbose
            if verbose == 1:
                train_pbar.update(1)
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        history['loss'][epoch] = total_loss / len(train_loader)

        if evaluate:
            # Evaluate on test set
            model.eval()
            correct = 0
            total = 0

            # Create progress bar for evaluation if verbose
            if verbose == 1:
                test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]', leave=False)
            
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    x_batch = x_batch.squeeze(1)
                    outputs = model(x_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

                    # Update progress bar description with current accuracy if verbose
                    accuracy = correct / total
                    if verbose==1:
                        test_pbar.update(1)
                        test_pbar.set_postfix({'accuracy': f'{accuracy:.4f}'})
            
            accuracy = correct / total
            history['accuracy'][epoch] = accuracy
            #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')
        else:
            #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')
            pass
        
        if checkpoint_folder:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': history['loss'][epoch],
            }
            if evaluate:
                checkpoint['accuracy'] = history['accuracy'][epoch]
            torch.save(checkpoint, os.path.join(checkpoint_folder, f"checkpoint_epoch_{epoch+1}.pth"))

    return model, history