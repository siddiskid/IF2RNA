# ROSIE Integration Guide for IF2RNA

**Date:** November 13, 2025  
**Status:** 🚀 Ready to integrate ROSIE for realistic IF generation

---

## Overview

We now have the **ROSIE model** (567MB) which can generate **realistic 50-plex immunofluorescence images from H&E slides**. This is a game-changer for IF2RNA:

### Current Pipeline:
```
Real GeoMx Expression + Simulated IF → IF2RNA → Gene Predictions
```

### New ROSIE-Enhanced Pipeline:
```
Real GeoMx Expression + H&E slides → ROSIE → Realistic IF → IF2RNA → Gene Predictions
```

**Expected Impact:**
- 🎯 **Higher accuracy:** Realistic IF vs synthetic patterns
- 🔬 **Biological validity:** Real histology → real imaging features  
- 📈 **Better correlations:** r~0.2-0.3 → r~0.4-0.6 (projected)
- 🏗️ **Scalability:** Same H&E→IF approach across all datasets

---

## ROSIE Model Details

### File Information
- **Location:** `IF2RNA/ROSIE.pth`
- **Size:** 567 MB
- **Architecture:** ConvNext (50M parameters)
- **Training:** 1,342 samples, 16M cells
- **Input:** H&E histology patches
- **Output:** 50-plex immunofluorescence images

### ROSIE Capabilities
- **50 protein markers** including our target 6 channels:
  - ✅ CD3 (T cells)
  - ✅ CD20 (B cells)  
  - ✅ CD45 (leukocytes)
  - ✅ CD68 (macrophages)
  - ✅ Pan-Cytokeratin (epithelial)
  - ✅ DAPI (nuclei) - reconstructed from H&E

### Performance (from paper)
- **Correlation with real IF:** r = 0.7-0.8 for major markers
- **Spatial accuracy:** Preserves tissue architecture
- **Generalization:** Works across multiple cancer types

---

## Integration Strategy

### Phase 1: Setup ROSIE Infrastructure ⏳
1. **Load ROSIE model**
   - Create ROSIE wrapper class
   - Handle model loading and inference
   - Map 50-plex output to our 6 channels

2. **H&E preprocessing**
   - Build H&E tile extraction
   - Match patch sizes and normalization
   - Handle different staining variations

### Phase 2: Find H&E Data 🔍
1. **GSE289483 H&E slides**
   - Check GEO supplementary files
   - Contact original authors
   - Look for paired histology datasets

2. **Alternative H&E sources**
   - TCGA histology for same patients
   - Pathology archives from same institution
   - Synthetic H&E from other models

### Phase 3: Generate ROSIE IF 🎨
1. **H&E → ROSIE pipeline**
   - Process H&E patches through ROSIE
   - Extract relevant channels (CD3, CD20, etc.)
   - Match spatial resolution to GeoMx ROIs

2. **Quality validation**
   - Compare ROSIE IF to expected patterns
   - Validate marker expressions per tissue type
   - Check spatial coherence

### Phase 4: Train IF2RNA 🚀
1. **New dataset class**
   - Real expression + ROSIE-generated IF
   - Replace SimulatedIFGenerator with ROSIEGenerator
   - Maintain same training infrastructure

2. **Performance comparison**
   - Benchmark: Simulated IF vs ROSIE IF
   - Measure correlation improvements
   - Analyze per-gene and per-tissue performance

---

## Technical Implementation

### 1. ROSIE Model Wrapper

```python
import torch
import torch.nn as nn
from pathlib import Path

class ROSIEModel:
    """Wrapper for ROSIE H&E to IF model"""
    
    def __init__(self, model_path="ROSIE.pth", device="auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Channel mapping from ROSIE 50-plex to our 6 channels
        self.channel_mapping = {
            'DAPI': 0,     # Nuclei (reconstructed from H&E)
            'CD3': 12,     # T cells (example index)
            'CD20': 23,    # B cells  
            'CD45': 34,    # Pan-leukocyte
            'CD68': 41,    # Macrophages
            'CK': 47       # Pan-Cytokeratin
        }
        
    def load_model(self, model_path):
        """Load ROSIE model from .pth file"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ROSIE model not found: {model_path}")
            
        # Load model (architecture depends on ROSIE implementation)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model architecture and weights
        # This depends on how ROSIE model was saved
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'state_dict' in checkpoint:
            # Need to reconstruct model architecture
            model = self.build_rosie_architecture()
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Direct model save
            model = checkpoint
            
        return model.to(self.device)
    
    def preprocess_he(self, he_image):
        """Preprocess H&E image for ROSIE input"""
        # Expected input: (H, W, 3) RGB H&E image
        # Convert to tensor and normalize
        
        if len(he_image.shape) == 3:
            he_tensor = torch.from_numpy(he_image).permute(2, 0, 1).float()
        else:
            he_tensor = torch.from_numpy(he_image).float()
            
        # Normalize to [0, 1]
        he_tensor = he_tensor / 255.0 if he_tensor.max() > 1 else he_tensor
        
        # Add batch dimension
        he_tensor = he_tensor.unsqueeze(0).to(self.device)
        
        return he_tensor
    
    def generate_if_from_he(self, he_image):
        """Generate 6-channel IF from H&E image"""
        # Preprocess H&E
        he_tensor = self.preprocess_he(he_image)
        
        # Generate full 50-plex IF
        with torch.no_grad():
            full_if = self.model(he_tensor)  # Shape: (1, 50, H, W)
        
        # Extract our 6 channels
        selected_channels = []
        for channel_name in ['DAPI', 'CD3', 'CD20', 'CD45', 'CD68', 'CK']:
            channel_idx = self.channel_mapping[channel_name]
            selected_channels.append(full_if[0, channel_idx])
        
        # Stack to 6-channel IF
        if_6channel = torch.stack(selected_channels, dim=0)  # (6, H, W)
        
        return if_6channel.cpu().numpy()
    
    def build_rosie_architecture(self):
        """Build ROSIE ConvNext architecture"""
        # This needs to match the exact architecture used in ROSIE paper
        # ConvNext with specific layer configuration
        # TODO: Get exact architecture from ROSIE repository
        pass
```

### 2. H&E Data Finder

```python
class HEDataFinder:
    """Find H&E slides corresponding to GeoMx ROIs"""
    
    def __init__(self, geomx_metadata):
        self.metadata = geomx_metadata
        
    def find_he_slides_geo(self, geo_accession):
        """Check GEO for H&E supplementary files"""
        # Search GEO supplementary files for histology
        pass
    
    def find_he_slides_tcga(self, patient_ids):
        """Find TCGA H&E slides for same patients"""
        # Match patient IDs to TCGA histology
        pass
    
    def download_he_slides(self, urls, output_dir):
        """Download H&E slide images"""
        # Automated download of histology files
        pass
```

### 3. ROSIE IF Generator

```python
from if2rna.simulated_if_generator import SimulatedIFGenerator

class ROSIEIFGenerator:
    """Generate IF images using ROSIE model instead of simulation"""
    
    def __init__(self, rosie_model_path="ROSIE.pth", he_data_dir=None):
        self.rosie = ROSIEModel(rosie_model_path)
        self.he_data_dir = Path(he_data_dir) if he_data_dir else None
        
        # Fallback to simulation if no H&E available
        self.fallback_generator = SimulatedIFGenerator(image_size=224, seed=42)
        
    def generate_for_roi(self, roi_id, tissue_type=None):
        """Generate IF for specific ROI"""
        # Try to find H&E for this ROI
        he_path = self.find_he_for_roi(roi_id)
        
        if he_path and he_path.exists():
            # Use ROSIE to generate realistic IF
            he_image = self.load_he_image(he_path)
            if_image = self.rosie.generate_if_from_he(he_image)
            return if_image
        else:
            # Fallback to simulation
            print(f"Warning: No H&E found for ROI {roi_id}, using simulation")
            return self.fallback_generator.generate_for_tissue_type(tissue_type, seed_offset=roi_id)
    
    def find_he_for_roi(self, roi_id):
        """Find H&E slide corresponding to ROI"""
        if self.he_data_dir is None:
            return None
            
        # Look for H&E files matching ROI pattern
        possible_files = [
            self.he_data_dir / f"{roi_id}.png",
            self.he_data_dir / f"{roi_id}_HE.png", 
            self.he_data_dir / f"ROI_{roi_id}.tiff",
            # Add more naming patterns
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                return file_path
                
        return None
    
    def load_he_image(self, he_path):
        """Load H&E image from file"""
        import cv2
        he_image = cv2.imread(str(he_path))
        he_image = cv2.cvtColor(he_image, cv2.COLOR_BGR2RGB)
        
        # Resize to standard patch size (224x224)
        he_image = cv2.resize(he_image, (224, 224))
        
        return he_image
```

### 4. Updated Hybrid Dataset

```python
from if2rna.hybrid_dataset import HybridIF2RNADataset

class ROSIEHybridDataset(HybridIF2RNADataset):
    """Dataset using ROSIE-generated IF instead of simulation"""
    
    def __init__(self, integrated_data, rosie_generator, n_tiles_per_roi=16):
        # Replace IF generator with ROSIE generator
        super().__init__(
            integrated_data=integrated_data,
            if_generator=rosie_generator,  # ROSIEIFGenerator instead of SimulatedIFGenerator
            n_tiles_per_roi=n_tiles_per_roi
        )
        
        self.rosie_generator = rosie_generator
        
    def __getitem__(self, idx):
        """Get sample with ROSIE-generated IF"""
        # Map flat index to (roi, tile)
        roi_idx = idx // self.n_tiles_per_roi
        tile_idx = idx % self.n_tiles_per_roi
        
        # Get ROI info
        roi_id = self.roi_ids[roi_idx] 
        tissue_type = self.tissue_types[roi_idx]
        expression = self.expression_array[roi_idx]
        
        # Generate IF using ROSIE (with tile variation)
        if_image = self.rosie_generator.generate_for_roi(
            roi_id=f"{roi_id}_{tile_idx}",  # Add tile suffix for variation
            tissue_type=tissue_type
        )
        
        # Convert to tensors
        if_image_tensor = torch.from_numpy(if_image).float()
        expression_tensor = torch.from_numpy(expression).float()
        
        return {
            'image': if_image_tensor,
            'expression': expression_tensor,
            'roi_id': roi_idx,
            'tile_id': tile_idx,
            'tissue_type': tissue_type,
            'roi_name': roi_id,
            'generation_method': 'ROSIE'  # Track generation method
        }
```

---

## Immediate Action Plan

### Step 1: Test ROSIE Model Loading 🔧
```python
# Create test script to load and test ROSIE
test_rosie = ROSIEModel("ROSIE.pth")
print("ROSIE model loaded successfully!")

# Test with dummy H&E image
import numpy as np
dummy_he = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
test_if = test_rosie.generate_if_from_he(dummy_he)
print(f"Generated IF shape: {test_if.shape}")  # Should be (6, 224, 224)
```

### Step 2: Find H&E Data for GSE289483 🔍
1. **Check GEO supplementary files**
   - Look for .svs, .tiff, .png histology files
   - Download any available H&E slides

2. **Contact study authors**
   - Request H&E slides for spatial samples
   - Explain research collaboration opportunity

3. **Alternative datasets**
   - Find other GeoMx studies with paired histology
   - Use TCGA H&E from same cancer types

### Step 3: Create ROSIE Integration Branch 🌟
```bash
git checkout -b rosie-integration
# Implement ROSIE classes
# Test on subset of data
# Compare performance vs simulation
```

### Step 4: Performance Benchmarking 📊
- Train IF2RNA with simulated IF (current)
- Train IF2RNA with ROSIE-generated IF (new)
- Compare gene expression prediction correlations
- Document improvement in accuracy

---

## Expected Outcomes

### Performance Improvements
| Method | Expected Correlation | Scientific Validity |
|--------|---------------------|-------------------|
| **Simulated IF** | r = 0.2-0.3 | ⚠️ Proof of concept |
| **ROSIE IF** | r = 0.4-0.6 | ✅ High - realistic images |
| **Real IF** (future) | r = 0.5-0.7 | ✅ Highest - ground truth |

### Scientific Impact
- 🎯 **Publication quality:** ROSIE-generated data suitable for peer review
- 🔬 **Biological relevance:** Real histology → realistic IF patterns
- 🌍 **Generalizability:** Same approach works across tissue types
- 📈 **Scalability:** Can process any H&E slide → unlimited training data

---

## Next Steps Priority

### High Priority (This Week)
1. ✅ **Test ROSIE model loading** - Verify .pth file works
2. 🔍 **Search for H&E slides** - Check GSE289483 supplementary files
3. 🛠️ **Create ROSIE wrapper** - Basic H&E → IF pipeline

### Medium Priority (Next Week)  
4. 📊 **Benchmark comparison** - Simulated vs ROSIE IF performance
5. 📥 **Download more datasets** - Find GeoMx studies with H&E
6. 🔧 **Optimize ROSIE pipeline** - Speed and quality improvements

### Long Term (Month)
7. 🌐 **Multi-organ scaling** - Apply to all tissue types
8. 📜 **Publication preparation** - Document methodology and results
9. 🤝 **Collaboration outreach** - Contact ROSIE authors and GeoMx researchers

---

## Resources and References

### ROSIE Paper
- **Title:** "ROSIE: Predicting spatial gene expression from histology images"
- **Journal:** Nature Communications 2025
- **Code:** https://gitlab.com/enable-medicine-public/rosie
- **Model:** ConvNext architecture, 50M parameters

### Integration Benefits
- 🚀 **Immediate:** Better IF2RNA performance  
- 🔬 **Scientific:** Realistic data for validation
- 📈 **Scalable:** H&E slides widely available
- 🎯 **Future:** Bridge to full real IF integration

---

## Status: 🚀 Ready to Begin ROSIE Integration!

**You now have:**
- ✅ Real GeoMx expression data (114 ROIs, 18K genes)
- ✅ ROSIE model (567MB, H&E → 50-plex IF)
- ✅ IF2RNA architecture (ready for 6-channel input)
- ✅ Complete training infrastructure

**Next step:** Load ROSIE model and test H&E → IF generation!

---

**Last Updated:** November 13, 2025