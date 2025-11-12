"""
IF2RNA: Predicting Spatial Gene Expression from Immunofluorescence Imaging using Deep Learning
"""

__version__ = "0.1.0"

# Import main components
from .model import IF2RNA, fit, evaluate, predict, training_epoch, MultiChannelResNet50
from .data import (
    IF2RNADataset, IF2RNATileDataset, create_synthetic_data, 
    IFDataset, create_synthetic_if_data, normalize_if_channels,
    load_multichannel_image, IFTileTransform
)
from .experiment import IF2RNAExperiment, create_experiment_config
from .config import (
    DEFAULT_MODEL_CONFIG, 
    DEFAULT_TRAINING_CONFIG, 
    DEFAULT_DATA_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG
)

__all__ = [
    'IF2RNA',
    'fit', 
    'evaluate', 
    'predict',
    'training_epoch',
    'MultiChannelResNet50',
    'IF2RNADataset',
    'IF2RNATileDataset', 
    'IFDataset',
    'create_synthetic_data',
    'create_synthetic_if_data',
    'normalize_if_channels',
    'load_multichannel_image',
    'IFTileTransform',
    'IF2RNAExperiment',
    'create_experiment_config',
    'DEFAULT_MODEL_CONFIG',
    'DEFAULT_TRAINING_CONFIG',
    'DEFAULT_DATA_CONFIG',
    'DEFAULT_EXPERIMENT_CONFIG'
]
