# IF2RNA

Predicting gene expression from immunofluorescence images using deep learning.

## Recent Updates

**Real IF Data Support**: The codebase now supports real immunofluorescence data from GeoMx DCC files instead of only synthetic data. See [docs/REAL_IF_DATA.md](docs/REAL_IF_DATA.md) for details.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train with Synthetic Data (Testing)
```bash
python scripts/test_baseline.py
```

### Test with Real IF Data
```bash
python scripts/test_real_if_integration.py
```

### Run Experiment with Real Data
```python
from if2rna.experiment import IF2RNAExperiment

experiment = IF2RNAExperiment(experiment_name="real_if_exp")
results = experiment.run_real_if_experiment(
    data_dir='data/real_geomx',
    n_genes=100
)
```

## Data

### Real IF Data (Production)
Place GeoMx DCC files in `data/real_geomx/`:
- `*.dcc` - Digital Count Conversion files
- `*.pkc` - Probe kit configuration files

### GeoMx Datasets
Place processed GeoMx datasets in `data/geomx_datasets/`

### Synthetic Data (Testing)
Synthetic data is automatically generated for testing when real data is unavailable.

## Key Features

- **Real IF Data Loading**: Load and process GeoMx DCC files
- **Hybrid Dataset**: Combine IF images with gene expression data
- **Flexible Architecture**: Support for both synthetic and real IF data
- **Backward Compatible**: Existing synthetic data pipelines still work

See [docs/REAL_IF_DATA.md](docs/REAL_IF_DATA.md) for comprehensive documentation on real data integration.

