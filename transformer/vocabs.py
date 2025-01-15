import os, pickle

from transformer.tokenizers import character_tokenization, atom_wise_tokenization, substructure_tokenization

def save_vocab(vocab, file_path):
    '''save vocabulary file

    Args:
        file_path: path to save specified vocabulary
    '''
    with open(file_path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(file_path):
    '''load vocabulary file

    Args:
        file_path: path to specified vocabulary
    '''
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_smiles_vocab(smiles_list, tokenization='character'):
    '''create SMILES vocabulary object based on tokenization method and SMILES list 

    Args:
        smiles_list: List of SMILES strings
        tokenization: method for tokenizations.  currently implemented are:
            character
            atom_wise
            substructure
    '''
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    
    unique_smiles = set(smiles_list)
    
    for smiles in unique_smiles:
        if tokenization == 'character':
            tokens = character_tokenization(smiles)
        elif tokenization == 'atom_wise':
            tokens = atom_wise_tokenization(smiles)
        elif tokenization == 'substructure':
            tokens = substructure_tokenization(smiles)
        else:
            raise ValueError(f"Unknown tokenization method: {tokenization}")
        
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    
    return vocab

def get_or_create_smiles_vocabs(df, smiles_col='SMILES', vocab_dir='./vocabs', force_create=False, source='MoNA'):
    '''calculate the Dice similarity between the true and predicted SMILES values

    Args:
        df: pandas DataFrame containing SMILES data
        smiles_col: name of dataframe column to use 
        vocab_dir: directory to save vocabularies
        force_create: boolean, force recreation rather than loading from a saved vocabulary
    '''
    os.makedirs(vocab_dir, exist_ok=True)
    
    smiles_vocabs = {}
    tokenization_methods = ['character', 'atom_wise', 'substructure']
    
    for method in tokenization_methods:
        vocab_path = os.path.join(vocab_dir, f'smiles_vocab_{source}_{method}.pkl')
        
        if os.path.exists(vocab_path) and not force_create:
            print(f"Loading existing {method} vocabulary...")
            smiles_vocabs[method] = load_vocab(vocab_path)
        else:
            print(f"Creating new {method} vocabulary...")
            vocab = create_smiles_vocab(df[smiles_col].unique(), tokenization=method)
            save_vocab(vocab, vocab_path)
            smiles_vocabs[method] = vocab
        
        print(f"SMILES vocabulary size ({method}): {len(smiles_vocabs[method])}")
    
    return smiles_vocabs