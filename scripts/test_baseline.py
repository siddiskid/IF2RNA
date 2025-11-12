#!/usr/bin/env python3
"""
IF2RNA Baseline Test: Validate model architecture with synthetic data
"""

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
    """Test IF2RNA model instantiation."""
    print("Testing model creation...")
    
    config = DEFAULT_MODEL_CONFIG.copy()
    config['device'] = 'cpu'  # Force CPU for testing
    
    model = IF2RNA(**config)
    print(f"✓ Model created: {model.__class__.__name__}")
    print(f"  - Input dim: {model.input_dim}")
    print(f"  - Output dim: {model.output_dim}")
    print(f"  - Device: {model.device}")
    print(f"  - Top-k values: {model.ks}")
    return model


def test_forward_pass():
    """Test model forward pass with synthetic data."""
    print("\nTesting forward pass...")
    
    # Create model
    config = DEFAULT_MODEL_CONFIG.copy()
    config['device'] = 'cpu'
    config['output_dim'] = 50  # Smaller for testing
    model = IF2RNA(**config)
    
    # Create synthetic input
    batch_size = 4
    n_tiles = 100
    input_tensor = torch.randn(batch_size, config['input_dim'], n_tiles)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {input_tensor.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    return True


def test_synthetic_training():
    """Test training with synthetic data."""
    print("\nTesting synthetic data training...")
    
    # Create synthetic data
    n_samples = 50
    n_genes = 20
    X, y, patients, projects = create_synthetic_data(
        n_samples=n_samples, 
        n_tiles=500, 
        n_genes=n_genes
    )
    
    print(f"✓ Synthetic data created")
    print(f"  - Samples: {n_samples}")
    print(f"  - Features: {X.shape}")
    print(f"  - Labels: {y.shape}")
    
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
    train_config['max_epochs'] = 3  # Quick test
    train_config['batch_size'] = 8
    
    # Get validation projects
    val_projects = np.array([projects[i] for i in val_set.indices])
    
    print("✓ Starting mini training...")
    
    # Train model
    preds, labels = fit(
        model=model,
        train_set=train_set,
        valid_set=val_set,
        valid_projects=val_projects,
        params=train_config,
        logdir='./test_logs'
    )
    
    print(f"✓ Training completed")
    print(f"  - Predictions shape: {preds.shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Correlation range: [{np.corrcoef(preds.flatten(), labels.flatten())[0,1]:.3f}]")
    
    return True


def main():
    """Run all baseline tests."""
    print("IF2RNA Baseline Validation")
    print("=" * 50)
    
    try:
        # Test 1: Model creation
        model = test_model_creation()
        
        # Test 2: Forward pass
        test_forward_pass()
        
        # Test 3: Synthetic training
        test_synthetic_training()
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("IF2RNA baseline architecture is working correctly.")
        print("Ready to proceed with real data integration.")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
