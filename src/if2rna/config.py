"""
IF2RNA: Configuration constants and paths
"""

import os

# Data paths
PATH_TO_SLIDE = 'data/slides'
PATH_TO_TILES = 'data/tiles'
PATH_TO_TRANSCRIPTOME = 'data/transcriptome'
PATH_TO_IF_DATA = 'data/immunofluorescence'
PATH_TO_GEOMX_DATA = 'data/geomx'

# Model configuration
DEFAULT_MODEL_CONFIG = {
    'input_dim': 2048,  # ResNet-50 feature dimension
    'output_dim': 100,  # Number of genes to predict (will be updated)
    'layers': [1],  # Hidden layer dimensions
    'ks': [10, 25, 50],  # Top-k values for attention
    'dropout': 0.5,
    'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
}

# Training configuration
DEFAULT_TRAINING_CONFIG = {
    'max_epochs': 200,
    'patience': 20,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'weight_decay': 0.0,
    'num_workers': 0  # Set to 0 for hdf5 files
}

# Data processing configuration
DEFAULT_DATA_CONFIG = {
    'n_tiles': 8000,
    'tile_size': 224,
    'overlap': 0,
    'magnification': '0.50_mpp',
    'feature_extractor': 'resnet50'
}

# Experiment configuration
DEFAULT_EXPERIMENT_CONFIG = {
    'random_seed': 42,
    'cv_folds': 5,
    'test_size': 0.2,
    'val_size': 0.2
}
