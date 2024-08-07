import os, json, pickle
import torch
from datetime import datetime

from classifier import MS_VIT
from sequence_pred import MS_VIT_Seq2Seq

def init_checkpoint_folder(base_path):
    '''initialize a new checkpoint folder, named checkpoint_*, where * increments by 1 with each new checkpoint path

    Args:
        base_path: the location to initialize a new checkpoint file
    '''
    i = 1
    while True:
        folder_path = os.path.join(base_path, f"checkpoint_{i}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print("Saving checkpoints to", folder_path)
            return folder_path
        i += 1

def save_model_meta(folder_path, model, optimizer, criterion, num_epochs, train_loader, test_loader, meta_tag):
    '''save a classification transformer model given a saved meta file

    Args:
        folder_path: folder to save meta.json file
        model: model object to parse parameters for saving
        optimizer: pytorch optimizer object to parse parameters for saving
        criterion: pytorch criterion or loss object to parse parameters for saving
        num_epochs: number of epochs the model is to be trained to
        train_loader: pytorch DataLoader object containing train data
        test_loader: pytorch DataLoader object containing test data
        meta_tag: text tag to include in meta.json file for user reference
    '''
    meta = {
        "model": {
            "name": type(model).__name__,
            "num_classes": model.fc.out_features,
            "embed_depth": model.embedding.in_features,
            "d_model": model.d_model,
            "nhead": model.transformer_encoder.layers[0].self_attn.num_heads,
            "num_layers": len(model.transformer_encoder.layers),
            "dim_feedforward": model.transformer_encoder.layers[0].linear1.out_features,
            "dropout": model.transformer_encoder.layers[0].dropout.p
        },
        "optimizer": {
            "name": type(optimizer).__name__,
            "lr": optimizer.param_groups[0]['lr'],
            "weight_decay": optimizer.param_groups[0].get('weight_decay', 0)
        },
        "criterion": type(criterion).__name__,
        "training": {
            "num_epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "train_size": len(train_loader.dataset),
            "test_size": len(test_loader.dataset)
        },
        "data": {
            "input_shape": tuple(next(iter(train_loader))[0].shape),
            "tokenization_method": getattr(train_loader.dataset, 'tokenization_method', 'unknown'),
            "max_mz": getattr(train_loader.dataset, 'max_mz', 'unknown')
        },
        "device": str(torch.cuda.get_device_name(0)),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": meta_tag
    }
    
    with open(os.path.join(folder_path, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

def load_model_from_meta(meta_path):
    '''load a classification transformer model given a saved meta file

    Args:
        meta_path: filepath of specified meta.json file
    '''
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    model = MS_VIT(
        num_classes=meta['model']['num_classes'],
        embed_depth=meta['model']['embed_depth'],
        d_model=meta['model']['d_model'],
        nhead=meta['model']['nhead'],
        num_layers=meta['model']['num_layers'],
        dim_feedforward=meta['model']['dim_feedforward'],
        dropout=meta['model']['dropout']
    )
    
    optimizer_class = getattr(torch.optim, meta['optimizer']['name'])
    optimizer = optimizer_class(model.parameters(), lr=meta['optimizer']['lr'], weight_decay=meta['optimizer']['weight_decay'])
    
    criterion = getattr(torch.nn, meta['criterion'])()
    
    return model, optimizer, criterion, meta['training']['num_epochs']

def load_seq2seq_from_meta(meta_path):
    '''load a sequence to sequence transformer model given a saved meta file

    Args:
        meta_path: filepath of specified meta.json file
    '''
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    model = MS_VIT_Seq2Seq(
        num_classes=meta['model']['num_classes'],
        smiles_vocab_size=meta['model']['smiles_vocab_size'],
        embed_depth=meta['model']['embed_depth'],
        d_model=meta['model']['d_model'],
        nhead=meta['model']['nhead'],
        num_layers=meta['model']['num_layers'],
        dim_feedforward=meta['model']['dim_feedforward'],
        dropout=meta['model']['dropout']
    )
    
    optimizer_class = getattr(torch.optim, meta['optimizer']['name'])
    optimizer = optimizer_class(model.parameters(), lr=meta['optimizer']['lr'], weight_decay=meta['optimizer']['weight_decay'])
    
    criterion_cls = getattr(torch.nn, meta['criterion_cls'])()
    criterion_seq = getattr(torch.nn, meta['criterion_seq'])()
    
    return model, optimizer, criterion_cls, criterion_seq, meta['training']['num_epochs']

def save_seq2seq_model_meta(folder_path, model, optimizer, criterion_seq, num_epochs, train_loader, test_loader, meta_tag):
    '''save a sequence to sequence transformer model given a saved meta file

    Args:
        folder_path: folder to save meta.json file
        model: model object to parse parameters for saving
        optimizer: pytorch optimizer object to parse parameters for saving
        criterion_seq: pytorch criterion or loss object to parse parameters for saving
        num_epochs: number of epochs the model is to be trained to
        train_loader: pytorch DataLoader object containing train data
        test_loader: pytorch DataLoader object containing test data
        meta_tag: text tag to include in meta.json file for user reference
    '''
    meta = {
        "model": {
            "name": type(model).__name__,
            "smiles_vocab_size": model.fc_smiles.out_features,
            "embed_depth": model.embedding.in_features,
            "d_model": model.d_model,
            "nhead": model.transformer_encoder.layers[0].self_attn.num_heads,
            "num_layers": len(model.transformer_encoder.layers),
            "dim_feedforward": model.transformer_encoder.layers[0].linear1.out_features,
            "dropout": model.transformer_encoder.layers[0].dropout.p
        },
        "optimizer": {
            "name": type(optimizer).__name__,
            "lr": optimizer.param_groups[0]['lr'],
            "weight_decay": optimizer.param_groups[0].get('weight_decay', 0)
        },
        "criterion_seq": type(criterion_seq).__name__,
        "training": {
            "num_epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "train_size": len(train_loader.dataset),
            "test_size": len(test_loader.dataset)
        },
        "data": {
            "input_shape": tuple(next(iter(train_loader))[0][0].shape),
            "tokenization_method": getattr(train_loader.dataset, 'tokenization_method', 'unknown'),
            "max_mz": getattr(train_loader.dataset, 'max_mz', 'unknown'),
            "smiles_tokenization": getattr(train_loader.dataset, 'smiles_tokenization', 'unknown')
        },
        "device": str(torch.cuda.get_device_name(0)),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": meta_tag
    }
    
    with open(os.path.join(folder_path, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

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