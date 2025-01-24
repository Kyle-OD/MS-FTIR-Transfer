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


class MultimodalVITSeq2SeqBeam(nn.Module):
    '''PyTorch module for multimodal spectrum to SMILES prediction with beam search
    
    Args:
        smiles_vocab_size: size of vocabulary for SMILES encoding
        modality_configs: dict of modality configurations, e.g.,
            {
                'MS': {'embed_depth': 16},
                'IR': {'embed_depth': 32}
            }
        d_model: dimensionality of the internal states of the model
        nhead: number of attention heads
        num_layers: number of transformer encoder/decoder layers
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
        
        # Modality type embeddings
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

        # Add length adapter for each modality
        self.length_adapters = nn.ModuleDict({
            modality: nn.Linear(config.get('max_length', 512), d_model) for modality, config in modality_configs.items()
        })
        
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
        
        # Initialize length adapters
        for adapter in self.length_adapters.values():
            nn.init.uniform_(adapter.weight.data, -initrange, initrange)
            nn.init.zeros_(adapter.bias.data)

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
        # # Get modality embedding
        # x = self.modality_embeddings[modality](x) * math.sqrt(self.d_model)
        
        # # Add modality type embedding
        # modality_idx = torch.tensor([self.modality_to_idx[modality]], 
        #                           device=x.device).expand(x.size(0))
        # modality_embedding = self.modality_type_embeddings(modality_idx).unsqueeze(1)
        # x = x + modality_embedding
        
        # # Add positional encoding
        # x = self.pos_encoder(x.transpose(0, 1))
        # return x

        # Get modality embedding (batch_size, 1, seq_length, embed_depth)
        batch_size = x.size(0)
        x = x.squeeze(1)  # Remove the extra dimension
        
        # Apply modality-specific embedding
        x = self.modality_embeddings[modality](x)  # (batch_size, seq_length, d_model)
        
        # Add modality type embedding
        modality_idx = torch.tensor([self.modality_to_idx[modality]], 
                                  device=x.device).expand(batch_size)
        modality_embedding = self.modality_type_embeddings(modality_idx).unsqueeze(1)
        x = x + modality_embedding
        
        # Adapt sequence length using length adapter
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_length)
        x = self.length_adapters[modality](x)  # (batch_size, d_model, d_model)
        x = x.transpose(1, 2)  # (batch_size, d_model, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1))  # (d_model, batch_size, d_model)
        
        return x

    def encode(self, inputs):
        '''Encode all modality inputs
        
        Args:
            inputs: dict of modality inputs, e.g.,
                {
                    'MS': ms_tensor,
                    'IR': ir_tensor
                }
        '''
        # # Process each modality
        # encoded_features = []
        # for modality, x in inputs.items():
        #     encoded = self.encode_modality(x, modality)
        #     encoded_features.append(encoded)
        
        # # Concatenate encoded features along the sequence dimension
        # combined_features = torch.cat(encoded_features, dim=0)
        
        # # Pass through transformer encoder
        # return self.transformer_encoder(combined_features)
        # Process each modality
        encoded_features = []
        for modality, x in inputs.items():
            encoded = self.encode_modality(x, modality)  # Shape: (1, batch_size, d_model)
            encoded_features.append(encoded)
        
        # Concatenate encoded features along sequence dimension
        combined_features = torch.cat(encoded_features, dim=0)  # Shape: (num_modalities, batch_size, d_model)
        
        # Pass through transformer encoder
        return self.transformer_encoder(combined_features)

    def decode_step(self, tgt, memory):
        '''Decode a single step for beam search
        
        Args:
            tgt: target sequence tensor
            memory: encoded memory from encoder
        '''
        batch_size = tgt.size(0)
    
        # Reshape and embed target sequence
        tgt = tgt.transpose(0, 1)  # [seq_len, batch_size]
        tgt = self.smiles_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)  # [seq_len, batch_size, d_model]
        
        # Reshape memory if needed
        if len(memory.shape) == 3 and memory.size(1) != batch_size:
            # Reshape memory to have correct batch dimension
            memory = memory.repeat(1, batch_size, 1)
        
        # Pass through decoder
        output = self.transformer_decoder(tgt, memory)  # [seq_len, batch_size, d_model]
        
        # Transform output
        output = output.transpose(0, 1)  # [batch_size, seq_len, d_model]
        output = self.fc_smiles(output)  # [batch_size, seq_len, vocab_size]
        
        return output

    def forward(self, inputs, tgt=None):
        '''Forward pass for training
        
        Args:
            inputs: dict of modality inputs
            tgt: target sequence (optional, for training)
        '''
        memory = self.encode(inputs)
        
        if tgt is not None:
            tgt = self.smiles_embedding(tgt) * math.sqrt(self.d_model)
            tgt = self.pos_encoder(tgt.transpose(0, 1))
            output = self.transformer_decoder(tgt, memory)
            return self.fc_smiles(output.transpose(0, 1))
        
        return memory

    def beam_search(self, src, beam_width=5, max_len=100, sos_idx=1, eos_idx=2):
        '''Perform beam search with improved sequence generation and scoring
        
        Args:
            src: input spectrum
            beam_width: number of beams to maintain
            max_len: maximum sequence length
            sos_idx: start of sequence token index
            eos_idx: end of sequence token index
        '''
        device = next(self.parameters()).device
        batch_size = next(iter(src.values())).size(0)
        
        # Encode input
        memory = self.encode(src)
        
        # Initialize beams for each batch item
        beams = []
        for b in range(batch_size):
            # Initialize with start token
            beam = {
                'sequences': torch.tensor([[sos_idx]], device=device),
                'scores': torch.zeros(1, device=device),
                'finished': [False],
                'length_normalized_scores': torch.zeros(1, device=device)
            }
            beams.append(beam)
        
        # Generate sequences
        for step in range(max_len - 1):
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
                
                # Length normalization factor
                length_penalty = ((5 + step + 1) ** 0.6) / (5 ** 0.6)
                
                # Get top k probabilities and tokens for each sequence
                top_probs, top_tokens = log_probs.topk(beam_width)
                
                candidates = {
                    'sequences': [],
                    'scores': [],
                    'finished': [],
                    'length_normalized_scores': []
                }
                
                for i in range(len(current_sequences)):
                    if beam['finished'][i]:
                        # Keep finished sequences as is
                        candidates['sequences'].append(current_sequences[i])
                        candidates['scores'].append(current_scores[i])
                        candidates['finished'].append(True)
                        candidates['length_normalized_scores'].append(
                            current_scores[i] / length_penalty
                        )
                        continue
                    
                    # Add top k candidates for this sequence
                    for j in range(beam_width):
                        new_token = top_tokens[i, j].unsqueeze(0)
                        new_seq = torch.cat([current_sequences[i], new_token])
                        new_score = current_scores[i] + top_probs[i, j]
                        
                        # Check for special tokens and repeated sequences
                        token_val = new_token.item()
                        prev_tokens = current_sequences[i][-3:] if len(current_sequences[i]) >= 3 else current_sequences[i]
                        
                        # Skip if generating a repetitive sequence
                        if len(prev_tokens) >= 3 and all(t == token_val for t in prev_tokens):
                            continue
                            
                        is_finished = (token_val == eos_idx or step == max_len - 2)
                        
                        candidates['sequences'].append(new_seq)
                        candidates['scores'].append(new_score)
                        candidates['finished'].append(is_finished)
                        candidates['length_normalized_scores'].append(new_score / length_penalty)
                
                all_candidates.append(candidates)
            
            # Select top beam_width candidates for each batch item
            for b in range(batch_size):
                candidates = all_candidates[b]
                if all(candidates['finished']):
                    beams[b] = candidates
                    continue
                
                # Use length-normalized scores for selection
                scores = torch.tensor(candidates['length_normalized_scores'], device=device)
                top_indices = scores.argsort(descending=True)[:beam_width]
                
                beams[b] = {
                    #'sequences': torch.stack([candidates['sequences'][i] for i in top_indices]),
                    'sequences': torch.stack([candidates['sequences'][i] for i in top_indices]),
                    'scores': torch.tensor([candidates['scores'][i] for i in top_indices], device=device),
                    'finished': [candidates['finished'][i] for i in top_indices],
                    'length_normalized_scores': scores[top_indices]
                }
                print(beams)
        
        return beams
    
    # def beam_search(self, inputs, beam_width=5, max_len=100, sos_idx=1, eos_idx=2):
    #     '''Perform beam search to get top k most likely sequences
        
    #     Args:
    #         inputs: dict of modality inputs
    #         beam_width: number of beams to maintain
    #         max_len: maximum sequence length
    #         sos_idx: start of sequence token index
    #         eos_idx: end of sequence token index
    #     '''
    #     device = next(self.parameters()).device
    #     batch_size = next(iter(inputs.values())).size(0)
        
    #     # Encode input and prepare memory
    #     memory = self.encode(inputs)  # [seq_len, batch_size, d_model]
        
    #     # Initialize beams for each batch item
    #     beams = []
    #     for b in range(batch_size):
    #         # Select memory for this batch item and ensure shape
    #         batch_memory = memory.clone()  # Keep full memory tensor
    #         # Adjust dimensions for individual batch processing
    #         batch_memory = batch_memory[:, b:b+1, :].contiguous()
            
    #         beam = {
    #             'sequences': torch.tensor([[sos_idx]], device=device),
    #             'scores': torch.zeros(1, device=device),
    #             'finished': [False],
    #             'memory': batch_memory
    #         }
    #         beams.append(beam)
        
    #     # Generate sequences
    #     for step in range(max_len - 1):
    #         all_candidates = []
            
    #         # Process each batch item separately
    #         for b, beam in enumerate(beams):
    #             if all(beam['finished']):
    #                 all_candidates.append(beam)
    #                 continue
                
    #             current_sequences = beam['sequences']
    #             current_scores = beam['scores']
    #             current_memory = beam['memory']
                
    #             # Get predictions for all current sequences
    #             decoder_output = self.decode_step(current_sequences, current_memory)
                
    #             # Get probabilities for last token
    #             log_probs = torch.log_softmax(decoder_output[:, -1], dim=1)
    #             top_probs, top_tokens = log_probs.topk(beam_width)
                
    #             # Create candidates for each sequence
    #             candidates = {
    #                 'sequences': [],
    #                 'scores': [],
    #                 'finished': [],
    #                 'memory': []
    #             }
                
    #             for i in range(len(current_sequences)):
    #                 if beam['finished'][i]:
    #                     candidates['sequences'].append(current_sequences[i])
    #                     candidates['scores'].append(current_scores[i])
    #                     candidates['finished'].append(True)
    #                     candidates['memory'].append(current_memory)
    #                     continue
                    
    #                 # Add top k candidates for this sequence
    #                 for j in range(beam_width):
    #                     if beam['finished'][i]:
    #                         # If sequence is finished, keep it as is
    #                         candidates['sequences'].append(current_sequences[i])
    #                         candidates['scores'].append(current_scores[i])
    #                         candidates['finished'].append(True)
    #                     else:
    #                         new_token = top_tokens[i, j].unsqueeze(0)
    #                         is_eos = (new_token.item() == eos_idx)
                            
    #                         # For finished sequences, add EOS token if not present
    #                         if is_eos:
    #                             new_seq = torch.cat([current_sequences[i], new_token])
    #                         else:
    #                             new_seq = torch.cat([current_sequences[i], new_token])
                            
    #                         new_score = current_scores[i] + top_probs[i, j]
                            
    #                         candidates['sequences'].append(new_seq)
    #                         candidates['scores'].append(new_score)
    #                         candidates['finished'].append(is_eos)
                        
    #                     candidates['memory'].append(current_memory)
                
    #             all_candidates.append(candidates)
            
    #         # Select top beam_width candidates for each batch item
    #         for b in range(batch_size):
    #             candidates = all_candidates[b]
    #             if all(candidates['finished']):
    #                 beams[b] = candidates
    #                 continue
                
    #             scores = torch.tensor(candidates['scores'], device=device)
    #             top_indices = scores.argsort(descending=True)[:beam_width]
                
    #             # Get max sequence length among selected candidates
    #             selected_sequences = [candidates['sequences'][i] for i in top_indices]
    #             max_len = max(seq.size(0) for seq in selected_sequences)
                
    #             # Pad sequences to same length
    #             padded_sequences = []
    #             for seq in selected_sequences:
    #                 if seq.size(0) < max_len:
    #                     padding = torch.full((max_len - seq.size(0),), 0, 
    #                                     dtype=seq.dtype, device=seq.device)
    #                     seq = torch.cat([seq, padding])
    #                 padded_sequences.append(seq)
                
    #             beams[b] = {
    #                 'sequences': torch.stack(padded_sequences),
    #                 'scores': scores[top_indices],
    #                 'finished': [candidates['finished'][i] for i in top_indices],
    #                 'memory': candidates['memory'][0]  # All candidates share the same memory
    #             }
        
    #     return beams