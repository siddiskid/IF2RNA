#!/usr/bin/env python3

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from if2rna.model import IF2RNA, fit, evaluate, predict
from if2rna.data import create_synthetic_data, IF2RNADataset
from if2rna.config import DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG


def test_model_creation():
    print("Testing model creation...")
    
    config = DEFAULT_MODEL_CONFIG.copy()
    config['device'] = 'cpu'
    
    model = IF2RNA(**config)
    print(f"Model created: {model.input_dim} -> {model.output_dim}")
    return model


def test_forward_pass():
    print("\nTesting forward pass...")
    
    # Create model
    config = DEFAULT_MODEL_CONFIG.copy()
    config['device'] = 'cpu'
    config['output_dim'] = 50
    model = IF2RNA(**config)
    
    # Create synthetic input
    batch_size = 4
    n_tiles = 100
    input_tensor = torch.randn(batch_size, config['input_dim'], n_tiles)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Forward pass: {input_tensor.shape} -> {output.shape}")
    
    return True


def test_synthetic_training():
    print("\nTesting training...")
    n_samples = 50
    n_genes = 20
    X, y, patients, projects = create_synthetic_data(
        n_samples=n_samples, 
        n_tiles=500, 
        n_genes=n_genes
    )
    
    print(f"Data: {n_samples} samples, {X.shape} features, {y.shape} labels")
    
    # Create dataset
    genes = [f"gene_{i:03d}" for i in range(n_genes)]
    dataset = IF2RNADataset(genes, patients, projects, X, y)
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create model
    config = DEFAULT_MODEL_CONFIG.copy()
    config['device'] = 'cpu'
    config['output_dim'] = n_genes
    model = IF2RNA(**config)
    
    # Training config
    train_config = DEFAULT_TRAINING_CONFIG.copy()
    train_config['max_epochs'] = 3
    train_config['batch_size'] = 8
    
    # Get validation projects
    val_projects = np.array([projects[i] for i in val_set.indices])
    
    print("✓ Starting mini training...")
    preds, labels = fit(
        model=model,
        train_set=train_set,
        valid_set=val_set,
        valid_projects=val_projects,
        params=train_config,
        logdir='./test_logs'
    )
    
    corr = np.corrcoef(preds.flatten(), labels.flatten())[0,1]
    print(f"Training done. Correlation: {corr:.3f}")
    
    return True


def main():
    print("IF2RNA Baseline Validation")
    print("=" * 50)
    
    try:
        model = test_model_creation()
        test_forward_pass()
        test_synthetic_training()
        
        print("\n" + "=" * 50)
        print("All tests passed")
        
        return 0
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
