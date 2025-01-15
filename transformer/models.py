import torch
from torch import nn
import math

from transformer.transformer_utils import PositionalEncoding

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


class MultimodalVITSeq2Seq(nn.Module):
    '''pytorch module predicting sequence from multiple spectral input modalities

    Args:
        smiles_vocab_size: size of vocabulary of SMILES encoding method
        modality_configs: dict of modality configurations, e.g.,
            {
                'MS': {'embed_depth': 16},
                'IR': {'embed_depth': 32}
            }
        d_model: dimensionality of the internal states of the model
        nhead: number of attention heads
        num_layers: number of transformer encoder layers
        dim_feedforward: dimensionality of feedforward classifier network
        dropout: dropout percentage applied to entire model
    '''
    def __init__(self, smiles_vocab_size, modality_configs, d_model=256, 
                 nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Modality-specific embeddings
        self.modality_embeddings = nn.ModuleDict({
            modality: nn.Linear(config['embed_depth'], d_model)
            for modality, config in modality_configs.items()
        })
        
        # Modality type embeddings (learned embeddings for each modality type)
        self.modality_type_embeddings = nn.Embedding(
            num_embeddings=len(modality_configs),
            embedding_dim=d_model
        )
        
        # Create a mapping from modality names to indices
        self.modality_to_idx = {
            modality: idx for idx, modality in enumerate(modality_configs.keys())
        }
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Decoder components
        self.smiles_embedding = nn.Embedding(smiles_vocab_size, d_model)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        
        # Output layer
        self.fc_smiles = nn.Linear(d_model, smiles_vocab_size)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # Initialize modality embeddings
        for embedding in self.modality_embeddings.values():
            embedding.weight.data.uniform_(-initrange, initrange)
        
        # Initialize other components
        self.modality_type_embeddings.weight.data.uniform_(-initrange, initrange)
        self.smiles_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_smiles.bias.data.zero_()
        self.fc_smiles.weight.data.uniform_(-initrange, initrange)

    def encode_modality(self, x, modality):
        '''Encode a single modality input
        
        Args:
            x: input tensor for the specific modality
            modality: string identifier for the modality (e.g., 'MS', 'IR')
        '''
        # Get modality embedding
        x = self.modality_embeddings[modality](x) * math.sqrt(self.d_model)
        
        # Add modality type embedding
        modality_idx = torch.tensor([self.modality_to_idx[modality]], 
                                  device=x.device).expand(x.size(0))
        modality_embedding = self.modality_type_embeddings(modality_idx).unsqueeze(1)
        x = x + modality_embedding
        
        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1))
        return x

    def forward(self, inputs, tgt=None):
        '''
        Args:
            inputs: dict of modality inputs, e.g.,
                {
                    'MS': ms_tensor,
                    'IR': ir_tensor
                }
            tgt: target sequence (for training)
        '''
        # Process each modality
        encoded_features = []
        for modality, x in inputs.items():
            encoded = self.encode_modality(x, modality)
            encoded_features.append(encoded)
        
        # Concatenate encoded features along the sequence dimension
        combined_features = torch.cat(encoded_features, dim=0)
        
        # Pass through transformer encoder
        memory = self.transformer_encoder(combined_features)
        
        # Decode SMILES if target is provided
        if tgt is not None:
            tgt = self.smiles_embedding(tgt) * math.sqrt(self.d_model)
            tgt = self.pos_encoder(tgt.transpose(0, 1))
            output = self.transformer_decoder(tgt, memory)
            return self.fc_smiles(output.transpose(0, 1))
        
        return memory
        

class MS_VIT_Seq2Seq_Beam(nn.Module):
    '''pytorch module predicting sequence (usually SMILES) from mass spectral input with beam search
    
    Args:
        smiles_vocab_size: size of vocabulary of SMILES encoding method 
        embed_depth: token depth
        d_model: dimensionality of the internal states of the model
        n_head: number of attention heads
        num_layers: number of transformer encoder layers
        dim_feedforward: dimensionality of feedforward classifier network
        dropout: dropout percentage applied to entire model
    '''
    def __init__(self, smiles_vocab_size, embed_depth=16, d_model=256, nhead=8, 
                 num_layers=6, dim_feedforward=2048, dropout=0.1):
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
        
        # Output layer
        self.fc_smiles = nn.Linear(d_model, smiles_vocab_size)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.smiles_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_smiles.bias.data.zero_()
        self.fc_smiles.weight.data.uniform_(-initrange, initrange)

    def encode(self, src):
        # Encode input spectrum
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1))
        return self.transformer_encoder(src)

    def decode_step(self, tgt, memory):
        # Decode single step
        tgt = self.smiles_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        return self.fc_smiles(output)

    def forward(self, src, tgt=None):
        # Standard forward pass for training
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1))
        memory = self.transformer_encoder(src)
        
        if tgt is not None:
            tgt = self.smiles_embedding(tgt) * math.sqrt(self.d_model)
            tgt = self.pos_encoder(tgt.transpose(0, 1))
            output = self.transformer_decoder(tgt, memory)
            return self.fc_smiles(output.transpose(0, 1))
        
        return memory

    def beam_search(self, src, beam_width=5, max_len=100, sos_idx=1, eos_idx=2):
        '''Perform beam search to get top 5 most likely sequences
        
        Args:
            src: input spectrum
            beam_width: number of beams to maintain (default: 5)
            max_len: maximum sequence length (default: 100)
            sos_idx: start of sequence token index
            eos_idx: end of sequence token index
        '''
        device = next(self.parameters()).device
        batch_size = src.size(0)
        
        # Encode input
        memory = self.encode(src)
        
        # Initialize beams for each batch item
        beams = []
        for b in range(batch_size):
            # Start with single beam containing SOS token
            beam = {
                'sequences': torch.tensor([[sos_idx]], device=device),
                'scores': torch.zeros(1, device=device),
                'finished': [False]
            }
            beams.append(beam)
        
        # Generate sequences
        for _ in range(max_len - 1):
            all_candidates = []
            
            # Process each batch item separately
            for b, beam in enumerate(beams):
                if all(beam['finished']):
                    all_candidates.append(beam)
                    continue
                
                current_sequences = beam['sequences']
                current_scores = beam['scores']
                
                # Get predictions for all current sequences
                decoder_output = self.decode_step(
                    current_sequences.transpose(0, 1),
                    memory.narrow(1, b, 1)
                )
                
                # Get probabilities for last token
                log_probs = torch.log_softmax(decoder_output[-1], dim=1)
                
                # Get top k probabilities and tokens for each sequence
                top_probs, top_tokens = log_probs.topk(beam_width)
                
                # Create candidates for each sequence
                candidates = {
                    'sequences': [],
                    'scores': [],
                    'finished': []
                }
                
                for i in range(len(current_sequences)):
                    if beam['finished'][i]:
                        # Keep finished sequences as is
                        candidates['sequences'].append(current_sequences[i])
                        candidates['scores'].append(current_scores[i])
                        candidates['finished'].append(True)
                        continue
                    
                    # Add top k candidates for this sequence
                    for j in range(beam_width):
                        new_seq = torch.cat([current_sequences[i], top_tokens[i, j].unsqueeze(0)])
                        new_score = current_scores[i] + top_probs[i, j]
                        
                        candidates['sequences'].append(new_seq)
                        candidates['scores'].append(new_score)
                        candidates['finished'].append(top_tokens[i, j].item() == eos_idx)
                
                all_candidates.append(candidates)
            
            # Select top beam_width candidates for each batch item
            for b in range(batch_size):
                candidates = all_candidates[b]
                if all(candidates['finished']):
                    beams[b] = candidates
                    continue
                
                # Convert lists to tensors for sorting
                scores = torch.tensor(candidates['scores'], device=device)
                
                # Get top beam_width indices
                top_indices = scores.argsort(descending=True)[:beam_width]
                
                # Update beam
                beams[b] = {
                    'sequences': torch.stack([candidates['sequences'][i] for i in top_indices]),
                    'scores': scores[top_indices],
                    'finished': [candidates['finished'][i] for i in top_indices]
                }
        
        return beams
