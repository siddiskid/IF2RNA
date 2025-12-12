import os
PATH_TO_SLIDE = 'data/slides'
PATH_TO_TILES = 'data/tiles'
PATH_TO_TRANSCRIPTOME = 'data/transcriptome'
PATH_TO_IF_DATA = 'data/immunofluorescence'
PATH_TO_GEOMX_DATA = 'data/geomx'

DEFAULT_MODEL_CONFIG = {
    'input_dim': 2048,
    'output_dim': 100,
    'layers': [1],
    'ks': [10, 25, 50]
    'dropout': 0.5,
    'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
}

DEFAULT_TRAINING_CONFIG = {
    'max_epochs': 200,
    'patience': 20,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'weight_decay': 0.0,
    'num_workers': 0
}

DEFAULT_DATA_CONFIG = {
    'n_tiles': 8000,
    'tile_size': 224,
    'overlap': 0,
    'magnification': '0.50_mpp',
    'feature_extractor': 'resnet50'
}

DEFAULT_EXPERIMENT_CONFIG = {
    'random_seed': 42,
    'cv_folds': 5,
    'test_size': 0.2,
    'val_size': 0.2
}
