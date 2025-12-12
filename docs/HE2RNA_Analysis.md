# HE2RNA Technical Analysis & Architecture Documentation

## Repository Overview

**Source**: https://github.com/owkin/HE2RNA_code  
**License**: GPL v3.0  
**Paper**: "HE2RNA: Learning Representations of Histology Images to Predict Gene Expression"  
**Status**: Archived (read-only since Feb 2023)

## Core Architecture Analysis

### 1. Model Architecture (`model.py`)

#### HE2RNA Class Structure
```python
class HE2RNA(nn.Module):
    def __init__(self, input_dim, output_dim, layers=[1], nonlin=nn.ReLU(), 
                 ks=[10], dropout=0.5, device='cpu', bias_init=None)
```

**Key Components:**
- **Architecture**: Sequential 1D convolutions (equivalent to shared MLP across tiles)
- **Input**: `n_tiles × 2048` ResNet features (n_tiles = 100 for supertiles, 8000 for full)
- **Layers**: Configurable depth with 1D conv layers
- **Attention Mechanism**: Top-k aggregation with dynamic k sampling during training
- **Output**: Direct prediction of gene expression values (regression)

#### Forward Pass Logic
1. **Masking**: Remove zero-padded tiles
2. **Feature Transform**: Apply 1D convolutions with dropout + nonlinearity
3. **Attention**: Top-k aggregation (different k values during train/test)
4. **Aggregation**: Mean pooling over selected tiles

### 2. Data Pipeline (`wsi_data.py`, `transcriptome_data.py`)

#### Data Flow Architecture
```
WSI Images → Tiles → ResNet Features → Supertiles → Gene Expression
```

**Components:**
- **TCGAFolder**: PyTorch Dataset for tile features
- **TranscriptomeDataset**: Gene expression data loading
- **H5Dataset**: Efficient storage for large datasets
- **Patient-level splits**: Prevents data leakage

#### Data Preprocessing
1. **Tile Extraction**: 224×224 patches from WSI
2. **Feature Extraction**: ResNet-50 (ImageNet pretrained) → 2048D features
3. **Supertile Clustering**: K-means to reduce from 8K→100 tiles per slide
4. **Gene Expression**: Log10(1 + FPKM) normalization

### 3. Training Pipeline (`main.py`)

#### Experiment Class
- **Configuration-driven**: INI files for hyperparameters
- **Cross-validation**: Patient-level splits (5-fold default)
- **Evaluation**: Pearson correlation per gene, per cancer type

#### Training Components
- **Loss**: MSE regression loss
- **Optimizer**: Adam (lr=3e-4) or SGD with momentum
- **Regularization**: Dropout (0.25), early stopping (patience=50)
- **Monitoring**: TensorboardX logging

### 4. Feature Extraction (`extract_tile_features*.py`)

#### Two Pipelines
1. **From WSI**: Direct extraction from whole slide images
2. **From Tiles**: Pre-extracted tile images

**ResNet Feature Extraction:**
- **Model**: ResNet-50 pretrained on ImageNet
- **Layer**: Average pooling layer (2048D output)
- **Preprocessing**: Color correction, normalization

## Key Dependencies Analysis

### Critical Dependencies (HE2RNA vs IF2RNA)
| Component | HE2RNA Version | IF2RNA Status | Notes |
|-----------|---------------|---------------|-------|
| **PyTorch** | 1.4.0 | 2.9.0 ✅ | Compatible, newer version |
| **TensorFlow** | 1.14.0 | Not installed | Only needed for ResNet features |
| **Keras** | 2.2.4 | Not installed | For ResNet feature extraction |
| **OpenSlide** | 1.1.1 | Need to install | Critical for WSI processing |
| **libKMCUDA** | Custom build | Optional | For supertile preprocessing |
| **colorcorrect** | 0.9 | Need to install | For image preprocessing |

### Missing Dependencies to Add
```bash
# Essential for IF2RNA
openslide-python>=1.1.2
tensorflow>=2.4.0  # For ResNet features
keras>=2.4.0
colorcorrect>=0.3.0

# Optional but recommended
# libKMCUDA>=6.2.2  # For supertile clustering
```

## Adaptation Strategy for IF2RNA

### 1. Direct Reuse Components ✅
- **Model Architecture**: HE2RNA class (minimal changes)
- **Training Loop**: fit() function and experiment management
- **Evaluation Metrics**: Correlation computation utilities
- **Configuration System**: INI-based parameter management

### 2. Adaptation Required 🔄
- **Data Loaders**: Modify for GeoMx file formats
- **Feature Extraction**: Adapt for immunofluorescence images
- **Path Constants**: Update for IF2RNA directory structure
- **Preprocessing**: IF-specific normalization and color handling

### 3. New Implementation Needed ⭐
- **GeoMx Data Parser**: Handle ROI-based spatial data
- **IF Image Processing**: Multi-channel immunofluorescence handling
- **Spatial Context**: Full-slide prediction vs ROI-based
- **Cross-organ Training**: Handle multiple organ types simultaneously

## File-by-File Analysis

### Core Files for IF2RNA Adaptation

#### Must Adapt
- **`model.py`** → `src/if2rna/model.py` (minimal changes)
- **`main.py`** → `src/if2rna/experiment.py` (moderate adaptation)
- **`wsi_data.py`** → `src/if2rna/data_loaders.py` (significant changes)
- **`utils.py`** → `src/if2rna/utils.py` (direct reuse)

#### Reference Only
- **`extract_tile_features*.py`** → Understand feature extraction approach
- **`transcriptome_data.py`** → Understand gene expression handling
- **`supertile_preprocessing.py`** → Understand clustering approach

#### Skip for Initial Implementation
- **`msi_prediction.py`** → MSI-specific, not needed for IF2RNA
- **`spatialization.py`** → Visualization, implement later

## Configuration Adaptation

### HE2RNA Config Example
```ini
[data]
path_to_transcriptome: data/TCGA_transcriptome/all_transcriptomes.csv
path_to_data: data/TCGA_100_supertiles.h5

[architecture]  
layers: 1024,1024
ks: 1,2,5,10,20,50,100
```

### IF2RNA Config Target
```ini
[data]
path_to_geomx_data: data/geomx_datasets/
organs: liver,kidney,colon,pancreas,lymphnode

[architecture]
layers: 1024,1024  # Keep same architecture
ks: 1,2,5,10,20,50,100  # Keep attention mechanism
```

## Next Steps for Implementation

### Phase 1: Core Adaptation
1. Copy and adapt model.py (preserve architecture)
2. Create IF2RNA-specific data loaders
3. Update constants and paths
4. Basic training pipeline adaptation

### Phase 2: Testing & Validation  
1. Test with minimal synthetic data
2. Validate against HE2RNA behavior
3. Debug integration issues

### Phase 3: GeoMx Integration
1. Implement GeoMx data parsers
2. Add immunofluorescence preprocessing
3. Multi-organ training logic

This analysis provides the foundation for successful HE2RNA→IF2RNA adaptation while preserving the core deep learning architecture that made HE2RNA successful.
