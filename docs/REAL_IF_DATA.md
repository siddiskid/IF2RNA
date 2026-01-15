# Real IF Data Integration

This document describes the changes made to support real immunofluorescence (IF) image data instead of synthetic data.

## Overview

The codebase has been updated to work with real IF data from GeoMx DCC (Digital Count Conversion) files. The system now supports both synthetic data (for testing) and real IF data (for production use).

## New Components

### 1. `RealIFImageLoader` (src/if2rna/real_if_loader.py)

A new class that loads and processes real IF data from DCC files:

- Reads DCC files containing probe count data
- Parses PKC (probe kit configuration) files for channel names
- Generates spatial IF images from count data
- Compatible with the existing `SimulatedIFGenerator` interface

**Key methods:**
- `load_all_dcc_data()`: Load all DCC files in directory
- `generate_for_roi(roi_index, tissue_type)`: Generate IF image for specific ROI
- `generate_batch(roi_indices, tissue_types)`: Generate batch of IF images

### 2. `create_real_if_data()` (src/if2rna/data.py)

New function to load real IF data:

```python
from if2rna.data import create_real_if_data

if_images, gene_expression, patients, projects = create_real_if_data(
    data_dir='data/real_geomx',
    n_genes=100,
    use_synthetic_fallback=True
)
```

## Updated Components

### 1. `HybridIF2RNADataset` (src/if2rna/hybrid_dataset.py)

Updated to support both synthetic and real IF data:

```python
from if2rna.hybrid_dataset import HybridIF2RNADataset
from if2rna.real_if_loader import RealIFImageLoader

if_loader = RealIFImageLoader('data/real_geomx')

dataset = HybridIF2RNADataset(
    integrated_data=integrated,
    if_generator=if_loader,
    n_tiles_per_roi=16,
    use_real_if=True  # Set to True for real data
)
```

### 2. `IF2RNAExperiment` (src/if2rna/experiment.py)

New method for real IF experiments:

```python
from if2rna.experiment import IF2RNAExperiment

experiment = IF2RNAExperiment(experiment_name="real_if_exp")

results = experiment.run_real_if_experiment(
    data_dir='data/real_geomx',
    n_genes=100
)
```

### 3. `config.py`

New configuration parameters:

```python
PATH_TO_GEOMX_DATA = 'data/real_geomx'

DEFAULT_DATA_CONFIG = {
    ...
    'use_real_if': True,  # Enable real IF data
    'real_if_data_dir': PATH_TO_GEOMX_DATA
}

DEFAULT_EXPERIMENT_CONFIG = {
    ...
    'use_real_data': True  # Use real data by default
}
```

## Data Format

### Expected Directory Structure

```
data/real_geomx/
├── Sample_01_ROI_001.dcc
├── Sample_01_ROI_002.dcc
├── Sample_02_ROI_001.dcc
├── Sample_02_ROI_002.dcc
├── Hs_R_NGS_WTA_v1.0.pkc
└── spatial_annotations.xml (optional)
```

### DCC File Format

DCC files are XML-based files containing:
- Probe count data for each gene/marker
- ROI metadata
- Quality control metrics

### PKC File Format

PKC files contain probe kit configuration:
- Target gene/marker names
- Probe sequences
- Panel information

## Usage Examples

### Example 1: Load Real IF Data

```python
from if2rna.real_if_loader import RealIFImageLoader
from if2rna.real_geomx_parser import RealGeoMxDataParser

# Load GeoMx expression data
parser = RealGeoMxDataParser('data/real_geomx')
parser.load_raw_counts()
integrated = parser.get_integrated_data(n_genes=100)

# Load IF images
if_loader = RealIFImageLoader('data/real_geomx')
if_image = if_loader.generate_for_roi(0, tissue_type='Tumor')
```

### Example 2: Create Dataset with Real IF

```python
from if2rna.hybrid_dataset import HybridIF2RNADataset
from if2rna.real_if_loader import RealIFImageLoader
from if2rna.real_geomx_parser import RealGeoMxDataParser

# Load data
parser = RealGeoMxDataParser('data/real_geomx')
parser.load_raw_counts()
integrated = parser.get_integrated_data(n_genes=100)

# Create IF loader
if_loader = RealIFImageLoader('data/real_geomx')

# Create dataset
dataset = HybridIF2RNADataset(
    integrated_data=integrated,
    if_generator=if_loader,
    n_tiles_per_roi=16,
    use_real_if=True
)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### Example 3: Run Experiment with Real Data

```python
from if2rna.experiment import IF2RNAExperiment

experiment = IF2RNAExperiment(
    experiment_name="real_if_experiment",
    save_dir="experiments"
)

# Configure for real data
experiment.data_config['use_real_if'] = True
experiment.data_config['real_if_data_dir'] = 'data/real_geomx'

# Run experiment
results = experiment.run_real_if_experiment(
    data_dir='data/real_geomx',
    n_genes=100
)

print(f"Overall correlation: {results['overall_correlation_mean']:.4f}")
```

## Testing

Run the integration test to verify real data loading:

```bash
python scripts/test_real_if_integration.py
```

This will test:
1. Real IF image loading from DCC files
2. GeoMx data integration
3. Hybrid dataset creation with real IF
4. Complete experiment with real data

## Migration Guide

### From Synthetic to Real Data

**Before:**
```python
from if2rna.simulated_if_generator import SimulatedIFGenerator

generator = SimulatedIFGenerator(image_size=224)
dataset = HybridIF2RNADataset(
    integrated_data=integrated,
    if_generator=generator
)
```

**After:**
```python
from if2rna.real_if_loader import RealIFImageLoader

loader = RealIFImageLoader('data/real_geomx', image_size=224)
dataset = HybridIF2RNADataset(
    integrated_data=integrated,
    if_generator=loader,
    use_real_if=True
)
```

## Backward Compatibility

All existing code using synthetic data continues to work:

- `SimulatedIFGenerator` is still available
- `create_synthetic_if_data()` and `create_synthetic_data()` still work
- Default behavior can be controlled via config flags

## Notes

- Real IF data loading includes automatic fallback to synthetic data if real data is unavailable
- The `RealIFImageLoader` generates spatial patterns from count data since actual IF images are not yet available
- Once actual IF images are available, the loader can be updated to read TIFF/image files directly
- Channel count is limited to 50 to match the ROSIE model expectations

## Future Enhancements

When actual IF image files become available:

1. Update `RealIFImageLoader` to read TIFF files directly
2. Add image preprocessing pipeline (normalization, registration)
3. Support for multiple image formats (TIFF, OME-TIFF, etc.)
4. Spatial alignment with GeoMx ROI coordinates
5. Quality control and validation metrics
