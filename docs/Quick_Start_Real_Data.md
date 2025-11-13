# Quick Start: Using Real GeoMx Data with IF2RNA

This guide shows how to quickly start training IF2RNA with the newly integrated real data.

---

## Prerequisites

```bash
# Required packages (should already be installed)
pip install torch torchvision numpy pandas scipy matplotlib seaborn
```

---

## 5-Minute Quick Start

### Option 1: Use the Jupyter Notebook (Recommended)

```bash
cd analysis
jupyter notebook real_data_integration.ipynb
```

Then run all cells to see the complete pipeline!

### Option 2: Python Script

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from if2rna.real_geomx_parser import RealGeoMxDataParser
from if2rna.simulated_if_generator import SimulatedIFGenerator
from if2rna.hybrid_dataset import HybridIF2RNADataset, create_train_val_split
from torch.utils.data import DataLoader

# 1. Load real GeoMx data (takes ~5 seconds)
print("Loading real GeoMx data...")
parser = RealGeoMxDataParser('data/geomx_datasets/GSE289483')
parser.load_raw_counts()
parser.load_processed_expression()
parser.create_metadata()
integrated = parser.get_integrated_data(use_processed=True, n_genes=1000)
print(f"✅ Loaded {integrated['metadata']['n_rois']} ROIs, "
      f"{integrated['metadata']['n_genes']} genes")

# 2. Create IF generator
print("\nCreating IF generator...")
if_generator = SimulatedIFGenerator(image_size=224, seed=42)
print(f"✅ Generator ready with {if_generator.n_channels} channels")

# 3. Create dataset
print("\nCreating hybrid dataset...")
dataset = HybridIF2RNADataset(
    integrated_data=integrated,
    if_generator=if_generator,
    n_tiles_per_roi=16
)
print(f"✅ Dataset created with {len(dataset)} samples")

# 4. Split train/val
print("\nSplitting train/val...")
train_ds, val_ds = create_train_val_split(dataset, val_fraction=0.2, seed=42)
print(f"✅ Train: {len(train_ds)}, Val: {len(val_ds)}")

# 5. Create DataLoaders
print("\nCreating DataLoaders...")
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# 6. Get a batch
print("\nTesting batch loading...")
batch = next(iter(train_loader))
print(f"✅ Batch shapes:")
print(f"   Images: {batch['image'].shape}")
print(f"   Expression: {batch['expression'].shape}")

print("\n" + "="*60)
print("🎉 SUCCESS! Ready for model training!")
print("="*60)
```

---

## Training IF2RNA Model

### Load Model

```python
from if2rna.model import MultiChannelResNet50, IF2RNA
import torch
import torch.nn as nn
import torch.optim as optim

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
n_channels = 6  # DAPI, CD3, CD20, CD45, CD68, CK
n_genes = 1000  # Number of genes to predict

# Feature extractor (ResNet50 adapted for IF)
feature_extractor = MultiChannelResNet50(
    n_input_channels=n_channels,
    pretrained=False  # Don't use ImageNet weights for IF data
).to(device)

# IF2RNA model (attention-based aggregation)
model = IF2RNA(
    n_genes=n_genes,
    n_tiles=1,  # For tile-level dataset
    top_k=512,   # Top-k attention
    feature_extractor=feature_extractor
).to(device)

print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Training Loop

```python
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        images = batch['image'].to(device)
        expression = batch['expression'].to(device)
        
        # Forward pass
        predictions = model(images.unsqueeze(2))  # Add tile dimension
        
        # Compute loss
        loss = criterion(predictions, expression)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            expression = batch['expression'].to(device)
            
            predictions = model(images.unsqueeze(2))
            loss = criterion(predictions, expression)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

# Train for a few epochs
print("\nStarting training...")
n_epochs = 10

for epoch in range(n_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

print("✅ Training complete!")
```

### Evaluate Performance

```python
import numpy as np
from scipy.stats import pearsonr

def compute_correlations(model, loader, device):
    """Compute per-gene correlations between predicted and true expression"""
    model.eval()
    
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            expression = batch['expression'].to(device)
            
            predictions = model(images.unsqueeze(2))
            
            all_preds.append(predictions.cpu().numpy())
            all_true.append(expression.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_true = np.vstack(all_true)
    
    # Per-gene correlation
    gene_corrs = []
    for i in range(all_preds.shape[1]):
        r, _ = pearsonr(all_true[:, i], all_preds[:, i])
        gene_corrs.append(r)
    
    return np.array(gene_corrs), all_preds, all_true

# Compute correlations
print("\nComputing performance metrics...")
gene_corrs, preds, true = compute_correlations(model, val_loader, device)

print(f"\n📊 Performance on Validation Set:")
print(f"   Mean correlation: {gene_corrs.mean():.3f}")
print(f"   Median correlation: {np.median(gene_corrs):.3f}")
print(f"   % genes with r > 0.3: {(gene_corrs > 0.3).mean()*100:.1f}%")
print(f"   % genes with r > 0.5: {(gene_corrs > 0.5).mean()*100:.1f}%")
```

---

## What to Expect

### Training Time
- **GPU:** ~5-10 minutes per epoch (114 ROIs, 1824 samples)
- **CPU:** ~30-60 minutes per epoch

### Expected Performance

Based on HE2RNA paper (with H&E images), typical correlations:
- Small datasets (~100 samples): **r ~ 0.2-0.3**
- Medium datasets (~500 samples): **r ~ 0.3-0.4**
- Large datasets (~1000+ samples): **r ~ 0.4-0.6**

With our **simulated IF** (not real), expect:
- Initial epochs: **r ~ 0.1-0.2** (learning basic patterns)
- After convergence: **r ~ 0.2-0.3** (IF simulation quality limit)

This is **lower than ideal** but proves the pipeline works! Performance will improve dramatically when using:
1. Real IF images (future)
2. ROSIE-generated IF (intermediate step)
3. More training data (additional GEO datasets)

### Baseline Comparison

Always compare to baseline (predicting mean expression):
```python
# Baseline: predict mean for each gene
mean_expr = all_true.mean(axis=0, keepdims=True)
baseline_preds = np.repeat(mean_expr, len(all_true), axis=0)

baseline_corrs = []
for i in range(all_true.shape[1]):
    r, _ = pearsonr(all_true[:, i], baseline_preds[:, i])
    baseline_corrs.append(r)

print(f"\nBaseline (mean prediction): {np.array(baseline_corrs).mean():.3f}")
print(f"IF2RNA improvement: {(gene_corrs.mean() - np.array(baseline_corrs).mean()):.3f}")
```

---

## Troubleshooting

### PyTorch Import Error
If you get `ImportError: ... libtorch_cpu.dylib`, reinstall PyTorch:
```bash
pip uninstall torch torchvision
pip install torch torchvision
```

### Out of Memory
Reduce batch size:
```python
train_loader = DataLoader(train_ds, batch_size=8, ...)  # Instead of 32
```

### Slow Training
- Use GPU if available
- Increase `num_workers` in DataLoader (e.g., `num_workers=4`)
- Reduce image size: `SimulatedIFGenerator(image_size=112, ...)`

### Poor Correlations
This is expected with simulated IF! Try:
1. Train for more epochs (50-100)
2. Use more training data (download additional GEO datasets)
3. Experiment with different model architectures
4. **Future:** Replace with ROSIE-generated or real IF images

---

## Next Steps

1. ✅ **Run the notebook** to visualize data and understand pipeline
2. ✅ **Train model** using script above
3. ✅ **Analyze results** - which genes predict well? Which tissue types?
4. ✅ **Download more data** - GSE279942, GSE243408, GSE306381
5. ✅ **Optimize hyperparameters** - learning rate, model size, etc.

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/if2rna/real_geomx_parser.py` | Parse real GeoMx CSV files |
| `src/if2rna/simulated_if_generator.py` | Generate tissue-specific IF images |
| `src/if2rna/hybrid_dataset.py` | Dataset combining real + simulated |
| `src/if2rna/model.py` | IF2RNA model architectures |
| `analysis/real_data_integration.ipynb` | Complete pipeline walkthrough |
| `docs/Real_Data_Integration_Summary.md` | Detailed documentation |

---

**Questions?** Check the comprehensive summary in `docs/Real_Data_Integration_Summary.md`

**Last Updated:** January 2025
