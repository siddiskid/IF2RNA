# IF2RNA

Predicting Spatial Gene Expression from Immunofluorescence Imaging using Deep Learning

## Overview

IF2RNA is a deep learning framework for predicting whole-slide gene expression from immunofluorescence (IF) images using paired GeoMx Digital Spatial Profiler data. The model extends the HE2RNA approach to work with multi-channel IF imaging and spatial transcriptomics data across multiple organ types.

### 🎉 Recent Update: Real Data Integration

**Now using real GeoMx gene expression data!**
- ✅ Downloaded GSE289483 (pulmonary cancer, 114 ROIs, 18,815 genes)
- ✅ Created parser for GeoMx CSV/PKC file formats
- ✅ Hybrid approach: Real expression + biologically realistic simulated IF images
- ✅ Training-ready dataset with 1,824 samples (16 tiles per ROI)

See [Real Data Integration Summary](docs/Real_Data_Integration_Summary.md) for details.

## Quick Start

### Option 1: Interactive Notebook (Recommended)
```bash
cd analysis
jupyter notebook real_data_integration.ipynb
```

### Option 2: Python Script
```python
from if2rna.real_geomx_parser import RealGeoMxDataParser
from if2rna.simulated_if_generator import SimulatedIFGenerator
from if2rna.hybrid_dataset import HybridIF2RNADataset

# Load real GeoMx data
parser = RealGeoMxDataParser('data/geomx_datasets/GSE289483')
parser.load_raw_counts()
integrated = parser.get_integrated_data(use_processed=True, n_genes=1000)

# Generate simulated IF images
if_generator = SimulatedIFGenerator(image_size=224)

# Create training dataset
dataset = HybridIF2RNADataset(integrated, if_generator, n_tiles_per_roi=16)
```

See [Quick Start Guide](docs/Quick_Start_Real_Data.md) for complete training example.

## Project Structure

```
IF2RNA/
├── src/if2rna/                    # Core implementation
│   ├── model.py                   # IF2RNA architecture (MultiChannelResNet50)
│   ├── real_geomx_parser.py      # Parse real GeoMx data ✨ NEW
│   ├── simulated_if_generator.py  # Generate tissue-specific IF ✨ NEW
│   ├── hybrid_dataset.py          # Real expression + simulated IF ✨ NEW
│   ├── data.py                    # Data utilities
│   ├── experiment.py              # Training framework
│   └── config.py                  # Configurations
├── data/
│   └── geomx_datasets/
│       └── GSE289483/             # Real GeoMx data ✨ NEW
│           ├── GSE289483_raw_counts.csv
│           ├── GSE289483_processed.csv
│           └── GSE289483_pkc
├── analysis/
│   ├── real_data_integration.ipynb  # Complete pipeline ✨ NEW
│   ├── if_adaptation_test.ipynb     # IF2RNA testing
│   └── he2rna_validation_test.ipynb # HE2RNA baseline
├── docs/
│   ├── Real_Data_Integration_Summary.md  # Detailed docs ✨ NEW
│   ├── Quick_Start_Real_Data.md          # Training guide ✨ NEW
│   ├── Real_Data_Acquisition_Guide.md    # Dataset info
│   └── ROSIE_Analysis_and_Relevance.md   # Future H&E→IF
├── external/HE2RNA_code/          # Reference implementation
└── requirements.txt
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/IF2RNA.git
cd IF2RNA

# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data

### Real GeoMx Gene Expression
- **Source:** NCBI GEO (https://www.ncbi.nlm.nih.gov/geo/)
- **Current Dataset:** GSE289483 (pulmonary pleomorphic carcinoma)
- **Format:** CSV with genes (rows) × ROI samples (columns)
- **Genes:** 18,815 total → 1,000 most variable selected for training
- **ROIs:** 114 samples across multiple tissue regions

### Simulated IF Images
- **Channels:** 6 (DAPI, CD3, CD20, CD45, CD68, Pan-Cytokeratin)
- **Size:** 224×224 pixels per image
- **Tissue Types:** Tumor, Immune Aggregate, Stroma, Normal
- **Biological Patterns:** Cell density and marker expression match tissue types

**Why Simulated IF?**
- Real multi-channel IF images rarely available in public datasets
- Simulated patterns based on known biology (T cells in immune regions, epithelial markers in tumor, etc.)
- Future: Replace with ROSIE-generated IF or contact authors for real images

## Model Architecture

- **Feature Extractor:** MultiChannelResNet50 (adapted for 6-channel IF input)
- **Aggregation:** IF2RNA with top-k attention mechanism
- **Input:** (6, 224, 224) multi-channel IF images
- **Output:** (n_genes,) gene expression vector
- **Training:** MSE loss between predicted and real GeoMx expression

## Performance

*Training in progress...*

Expected correlations (based on HE2RNA with real H&E):
- With simulated IF: r ~ 0.2-0.3 (proof of concept)
- With real IF: r ~ 0.4-0.6 (projected)
- With multi-organ training: r ~ 0.5-0.7 (goal)

## Documentation

- [Real Data Integration Summary](docs/Real_Data_Integration_Summary.md) - Complete overview of data pipeline
- [Quick Start Guide](docs/Quick_Start_Real_Data.md) - Training walkthrough
- [Real Data Acquisition Guide](docs/Real_Data_Acquisition_Guide.md) - GeoMx dataset details
- [ROSIE Analysis](docs/ROSIE_Analysis_and_Relevance.md) - Future H&E→IF generation

## Development

**Directed Studies Project**  
University of British Columbia, Winter 2025-2026

**Supervisors:**
- Dr. Amrit Singh (Centre for Heart Lung Innovation)
- Dr. Jiarui Ding (Department of Biochemistry and Molecular Biology)

**Student:** Siddarth Chilukuri

## Roadmap

- [x] HE2RNA baseline validation
- [x] IF2RNA architecture implementation
- [x] Real GeoMx data integration ✨ **COMPLETED**
- [x] Simulated IF generator with tissue-specific patterns ✨ **COMPLETED**
- [x] Hybrid dataset infrastructure ✨ **COMPLETED**
- [ ] Model training on real expression data (in progress)
- [ ] Multi-organ training (5+ tissue types)
- [ ] ROSIE integration for H&E→IF generation
- [ ] Contact authors for real IF images
- [ ] Cross-organ generalization evaluation

## Citation

Coming soon...

## License

MIT License - see LICENSE file for details