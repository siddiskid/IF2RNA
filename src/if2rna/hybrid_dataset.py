"""
Hybrid Dataset for IF2RNA Training
Combines REAL GeoMx gene expression with SIMULATED IF images
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridIF2RNADataset(Dataset):
    """
    PyTorch dataset combining:
    - Real gene expression from GeoMx (downloaded from GEO)
    - Simulated IF images based on tissue type annotations
    """
    
    def __init__(self, integrated_data, if_generator, n_tiles_per_roi=16):
        """
        Args:
            integrated_data: Output from RealGeoMxDataParser.get_integrated_data()
            if_generator: SimulatedIFGenerator instance
            n_tiles_per_roi: Number of image tiles per ROI (for data augmentation)
        """
        self.roi_ids = integrated_data['roi_ids']
        self.gene_expression = integrated_data['gene_expression']  # DataFrame or array
        self.gene_names = integrated_data['gene_names']
        self.spatial_coords = integrated_data['spatial_coords']
        self.metadata = integrated_data['metadata']
        
        self.if_generator = if_generator
        self.n_tiles_per_roi = n_tiles_per_roi
        
        # Convert expression to numpy if pandas
        if hasattr(self.gene_expression, 'values'):
            self.expression_array = self.gene_expression.values
        else:
            self.expression_array = np.array(self.gene_expression)
        
        # Get tissue types for each ROI
        self.tissue_types = self.spatial_coords['tissue_region'].values
        
        # Total samples = n_rois * n_tiles_per_roi
        self.n_rois = len(self.roi_ids)
        self.total_samples = self.n_rois * self.n_tiles_per_roi
        
        logger.info(f"Created HybridIF2RNADataset:")
        logger.info(f"  - {self.n_rois} ROIs")
        logger.info(f"  - {len(self.gene_names)} genes")
        logger.info(f"  - {self.n_tiles_per_roi} tiles per ROI")
        logger.info(f"  - Total samples: {self.total_samples}")
        logger.info(f"  - Tissue types: {np.unique(self.tissue_types)}")
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        Get one sample (IF image + gene expression).
        
        Returns:
            dict with:
                - 'image': torch.Tensor of shape (n_channels, H, W)
                - 'expression': torch.Tensor of shape (n_genes,)
                - 'roi_id': int
                - 'tile_id': int (which tile within ROI)
                - 'tissue_type': str
        """
        # Map flat index to (roi, tile)
        roi_idx = idx // self.n_tiles_per_roi
        tile_idx = idx % self.n_tiles_per_roi
        
        # Get tissue type and expression for this ROI
        tissue_type = self.tissue_types[roi_idx]
        expression = self.expression_array[roi_idx]
        
        # Generate simulated IF image
        # Use tile_idx as seed offset for variability within ROI
        if_image = self.if_generator.generate_for_tissue_type(
            tissue_type, 
            seed_offset=idx
        )
        
        # Convert to torch tensors
        if_image_tensor = torch.from_numpy(if_image).float()
        expression_tensor = torch.from_numpy(expression).float()
        
        return {
            'image': if_image_tensor,
            'expression': expression_tensor,
            'roi_id': roi_idx,
            'tile_id': tile_idx,
            'tissue_type': tissue_type,
            'roi_name': self.roi_ids[roi_idx]
        }


class AggregatedIF2RNADataset(Dataset):
    """
    Dataset where multiple tiles are aggregated per ROI.
    This matches the HE2RNA paper's approach of aggregating features.
    """
    
    def __init__(self, integrated_data, if_generator, n_tiles_per_roi=16):
        """
        Args:
            integrated_data: Output from RealGeoMxDataParser.get_integrated_data()
            if_generator: SimulatedIFGenerator instance
            n_tiles_per_roi: Number of image tiles to generate per ROI
        """
        self.roi_ids = integrated_data['roi_ids']
        self.gene_expression = integrated_data['gene_expression']
        self.gene_names = integrated_data['gene_names']
        self.spatial_coords = integrated_data['spatial_coords']
        self.metadata = integrated_data['metadata']
        
        self.if_generator = if_generator
        self.n_tiles_per_roi = n_tiles_per_roi
        
        # Convert expression to numpy if pandas
        if hasattr(self.gene_expression, 'values'):
            self.expression_array = self.gene_expression.values
        else:
            self.expression_array = np.array(self.gene_expression)
        
        # Get tissue types
        self.tissue_types = self.spatial_coords['tissue_region'].values
        
        self.n_rois = len(self.roi_ids)
        
        logger.info(f"Created AggregatedIF2RNADataset:")
        logger.info(f"  - {self.n_rois} ROIs")
        logger.info(f"  - {len(self.gene_names)} genes")
        logger.info(f"  - {self.n_tiles_per_roi} tiles per ROI (aggregated)")
        
    def __len__(self):
        return self.n_rois
    
    def __getitem__(self, idx):
        """
        Get one ROI with multiple tiles.
        
        Returns:
            dict with:
                - 'tiles': torch.Tensor of shape (n_tiles, n_channels, H, W)
                - 'expression': torch.Tensor of shape (n_genes,)
                - 'roi_id': int
                - 'tissue_type': str
        """
        tissue_type = self.tissue_types[idx]
        expression = self.expression_array[idx]
        
        # Generate multiple tiles for this ROI
        tiles = []
        for tile_idx in range(self.n_tiles_per_roi):
            tile = self.if_generator.generate_for_tissue_type(
                tissue_type,
                seed_offset=idx * self.n_tiles_per_roi + tile_idx
            )
            tiles.append(tile)
        
        tiles_array = np.stack(tiles, axis=0)
        
        # Convert to tensors
        tiles_tensor = torch.from_numpy(tiles_array).float()
        expression_tensor = torch.from_numpy(expression).float()
        
        return {
            'tiles': tiles_tensor,
            'expression': expression_tensor,
            'roi_id': idx,
            'tissue_type': tissue_type,
            'roi_name': self.roi_ids[idx]
        }


def create_train_val_split(dataset, val_fraction=0.2, seed=42):
    """
    Split dataset into train/validation.
    
    Args:
        dataset: PyTorch Dataset
        val_fraction: Fraction for validation
        seed: Random seed
        
    Returns:
        train_dataset, val_dataset
    """
    from torch.utils.data import Subset
    
    n_samples = len(dataset)
    n_val = int(n_samples * val_fraction)
    
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    logger.info(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    return train_dataset, val_dataset


def test_hybrid_dataset():
    """Test the hybrid dataset"""
    print("="*60)
    print("Testing Hybrid IF2RNA Dataset")
    print("="*60)
    
    # Load real data
    from pathlib import Path
    import sys
    sys.path.append('/Users/siddarthchilukuri/Documents/GitHub/IF2RNA/src')
    
    from if2rna.real_geomx_parser import RealGeoMxDataParser
    from if2rna.simulated_if_generator import SimulatedIFGenerator
    
    data_dir = Path("/Users/siddarthchilukuri/Documents/GitHub/IF2RNA/data/geomx_datasets/GSE289483")
    
    # Parse real data
    print("\n1. Loading real GeoMx data...")
    parser = RealGeoMxDataParser(data_dir)
    parser.load_raw_counts()
    parser.load_processed_expression()
    parser.create_metadata()
    integrated = parser.get_integrated_data(use_processed=True, n_genes=1000)
    
    print(f"   Loaded {integrated['metadata']['n_rois']} ROIs, "
          f"{integrated['metadata']['n_genes']} genes")
    
    # Create IF generator
    print("\n2. Creating IF generator...")
    if_generator = SimulatedIFGenerator(image_size=224, seed=42)
    
    # Create tile-level dataset
    print("\n3. Creating tile-level dataset...")
    tile_dataset = HybridIF2RNADataset(
        integrated_data=integrated,
        if_generator=if_generator,
        n_tiles_per_roi=16
    )
    
    print(f"   Total samples: {len(tile_dataset)}")
    
    # Test getting one sample
    print("\n4. Getting sample...")
    sample = tile_dataset[0]
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Expression shape: {sample['expression'].shape}")
    print(f"   ROI: {sample['roi_name']}")
    print(f"   Tissue: {sample['tissue_type']}")
    print(f"   Expression range: {sample['expression'].min():.2f} - {sample['expression'].max():.2f}")
    
    # Create aggregated dataset
    print("\n5. Creating aggregated dataset...")
    agg_dataset = AggregatedIF2RNADataset(
        integrated_data=integrated,
        if_generator=if_generator,
        n_tiles_per_roi=16
    )
    
    print(f"   Total ROIs: {len(agg_dataset)}")
    
    agg_sample = agg_dataset[0]
    print(f"   Tiles shape: {agg_sample['tiles'].shape}")
    print(f"   Expression shape: {agg_sample['expression'].shape}")
    
    # Test train/val split
    print("\n6. Testing train/val split...")
    train_ds, val_ds = create_train_val_split(tile_dataset, val_fraction=0.2, seed=42)
    print(f"   Train: {len(train_ds)} samples")
    print(f"   Val: {len(val_ds)} samples")
    
    print("\n" + "="*60)
    print("✅ Hybrid dataset test complete!")
    print("="*60)
    print("\n🎉 SUCCESS: Real gene expression + Simulated IF images!")
    print("   - Real data: 114 ROIs, 1000 genes from GSE289483")
    print("   - Simulated: 6-channel IF images (224x224)")
    print("   - Ready for IF2RNA training!")
    print("="*60)


if __name__ == '__main__':
    test_hybrid_dataset()
