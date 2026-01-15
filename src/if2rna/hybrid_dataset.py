

import numpy as np
import torch
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridIF2RNADataset(Dataset):
    
    def __init__(self, integrated_data, if_generator, n_tiles_per_roi=16, use_real_if=False):
        self.roi_ids = integrated_data['roi_ids']
        self.gene_expression = integrated_data['gene_expression']
        self.gene_names = integrated_data['gene_names']
        self.spatial_coords = integrated_data['spatial_coords']
        self.metadata = integrated_data['metadata']
        
        self.if_generator = if_generator
        self.n_tiles_per_roi = n_tiles_per_roi
        self.use_real_if = use_real_if
        
        if hasattr(self.gene_expression, 'values'):
            self.expression_array = self.gene_expression.values
        else:
            self.expression_array = np.array(self.gene_expression)
        
        self.tissue_types = self.spatial_coords['tissue_region'].values
        
        self.n_rois = len(self.roi_ids)
        self.total_samples = self.n_rois * self.n_tiles_per_roi
        
        logger.info(f"Created HybridIF2RNADataset:")
        logger.info(f"  - {self.n_rois} ROIs")
        logger.info(f"  - {len(self.gene_names)} genes")
        logger.info(f"  - {self.n_tiles_per_roi} tiles per ROI")
        logger.info(f"  - Total samples: {self.total_samples}")
        logger.info(f"  - Tissue types: {np.unique(self.tissue_types)}")
        logger.info(f"  - Using real IF data: {self.use_real_if}")
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        roi_idx = idx // self.n_tiles_per_roi
        tile_idx = idx % self.n_tiles_per_roi
        
        tissue_type = self.tissue_types[roi_idx]
        expression = self.expression_array[roi_idx]
        
        if self.use_real_if:
            if_image = self.if_generator.generate_for_roi(
                roi_idx, 
                tissue_type,
                seed_offset=tile_idx
            )
        else:
            if_image = self.if_generator.generate_for_tissue_type(
                tissue_type, 
                seed_offset=idx
            )
        
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
    
    def __init__(self, integrated_data, if_generator, n_tiles_per_roi=16, use_real_if=False):
        """
        Args:
            integrated_data: Output from RealGeoMxDataParser.get_integrated_data()
            if_generator: IF generator instance (SimulatedIFGenerator or RealIFImageLoader)
            n_tiles_per_roi: Number of image tiles to generate per ROI
            use_real_if: Whether to use real IF data loader interface
        """
        self.roi_ids = integrated_data['roi_ids']
        self.gene_expression = integrated_data['gene_expression']
        self.gene_names = integrated_data['gene_names']
        self.spatial_coords = integrated_data['spatial_coords']
        self.metadata = integrated_data['metadata']
        
        self.if_generator = if_generator
        self.n_tiles_per_roi = n_tiles_per_roi
        self.use_real_if = use_real_if
        
        if hasattr(self.gene_expression, 'values'):
            self.expression_array = self.gene_expression.values
        else:
            self.expression_array = np.array(self.gene_expression)
        
        self.tissue_types = self.spatial_coords['tissue_region'].values
        
        self.n_rois = len(self.roi_ids)
        
        logger.info(f"Created AggregatedIF2RNADataset:")
        logger.info(f"  - {self.n_rois} ROIs")
        logger.info(f"  - {len(self.gene_names)} genes")
        logger.info(f"  - {self.n_tiles_per_roi} tiles per ROI (aggregated)")
        logger.info(f"  - Using real IF data: {self.use_real_if}")
        
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
        
        tiles = []
        for tile_idx in range(self.n_tiles_per_roi):
            if self.use_real_if:
                tile = self.if_generator.generate_for_roi(
                    idx,
                    tissue_type,
                    seed_offset=tile_idx
                )
            else:
                tile = self.if_generator.generate_for_tissue_type(
                    tissue_type,
                    seed_offset=idx * self.n_tiles_per_roi + tile_idx
                )
            tiles.append(tile)
        
        tiles_array = np.stack(tiles, axis=0)
        
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
    # Add src to path relative to this file
    sys.path.append(str(Path(__file__).parent.parent))
    
    from if2rna.real_geomx_parser import RealGeoMxDataParser
    from if2rna.simulated_if_generator import SimulatedIFGenerator
    
    # Use relative path
    data_dir = Path(__file__).parent.parent.parent / "data" / "geomx_datasets" / "GSE289483"
    
    # Parse real data
    print("Loading GeoMx data...")
    parser = RealGeoMxDataParser(data_dir)
    parser.load_raw_counts()
    parser.load_processed_expression()
    parser.create_metadata()
    integrated = parser.get_integrated_data(use_processed=True, n_genes=1000)
    
    print(f"Loaded {integrated['metadata']['n_rois']} ROIs, {integrated['metadata']['n_genes']} genes")
    
    # Create IF generator
    print("Creating IF generator...")
    if_generator = SimulatedIFGenerator(image_size=224, seed=42)
    
    # Create tile-level dataset
    print("Creating tile dataset...")
    tile_dataset = HybridIF2RNADataset(
        integrated_data=integrated,
        if_generator=if_generator,
        n_tiles_per_roi=16
    )
    
    print(f"Total samples: {len(tile_dataset)}")
    
    # Test getting one sample
    sample = tile_dataset[0]
    print(f"Sample: {sample['image'].shape}, {sample['tissue_type']}")
    
    # Create aggregated dataset
    print("Creating aggregated dataset...")
    agg_dataset = AggregatedIF2RNADataset(
        integrated_data=integrated,
        if_generator=if_generator,
        n_tiles_per_roi=16
    )
    
    print(f"Total ROIs: {len(agg_dataset)}")
    
    agg_sample = agg_dataset[0]
    print(f"Agg sample: {agg_sample['tiles'].shape}")
    
    # Test train/val split
    train_ds, val_ds = create_train_val_split(tile_dataset, val_fraction=0.2, seed=42)
    print(f"Split: {len(train_ds)} train, {len(val_ds)} val")
    
    print("Test complete")


if __name__ == '__main__':
    test_hybrid_dataset()
