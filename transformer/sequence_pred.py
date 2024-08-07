import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import math, os, json
from datetime import datetime
from tqdm.notebook import tqdm # swap with line below if not using jupyter notebook
#from tqdm import tqdm

from transformer_utils import PositionalEncoding
from io_funcs import load_seq2seq_from_meta, save_seq2seq_model_meta, init_checkpoint_folder
from evaluation import evaluate_model_seq2seq


class MS_VIT_Seq2Seq(nn.Module):
    '''pytorch module predicting sequence (usually SMILES) from from mass spectral input

    Args:
        smiles_vocab_size: size of vocabulary of SMILES encoding method 
        embed_depth: token depth
        d_model: dimensionality of the internal states of the model
        n_head: number of attention heads
        num_layers: number of transformer encoder layers
        dim_feedforward: dimensionality of feedforward classifier network
        dropout: dropout percentage applied to entire model
    '''
    def __init__(self, smiles_vocab_size, embed_depth=16, d_model=256, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Encoder
        self.embedding = nn.Linear(embed_depth, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Decoder
        self.smiles_embedding = nn.Embedding(smiles_vocab_size, d_model)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        
        # Output layers
        self.fc_smiles = nn.Linear(d_model, smiles_vocab_size)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.smiles_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_smiles.bias.data.zero_()
        self.fc_smiles.weight.data.uniform_(-initrange, initrange)
        if self.classification:
            self.fc_classification.bias.data.zero_()
            self.fc_classification.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt=None):
        # Encode input spectrum
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1))
        memory = self.transformer_encoder(src)
        
        # Decode SMILES
        tgt = self.smiles_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt.transpose(0, 1))
        output = self.transformer_decoder(tgt, memory)
        smiles_output = self.fc_smiles(output.transpose(0, 1))
        
        if self.classification:
            # Global average pooling and classification
            cls_output = memory.mean(dim=0)
            cls_output = self.fc_classification(cls_output)
            return smiles_output, cls_output
        else:
            return smiles_output

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