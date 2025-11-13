"""
Real GeoMx Data Parser for IF2RNA
Parses actual GEO downloaded files (CSV format, not DCC format)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealGeoMxDataParser:
    """Parser for real GeoMx data downloaded from GEO."""
    
    def __init__(self, data_dir):
        """
        Initialize parser with data directory.
        
        Args:
            data_dir: Path to directory containing GeoMx files
        """
        self.data_dir = Path(data_dir)
        self.raw_counts = None
        self.processed_expr = None
        self.probe_config = None
        self.metadata = None
        
    def load_raw_counts(self, filename='GSE289483_raw_counts.csv'):
        """
        Load raw count data from CSV file.
        
        Format: genes (rows) x samples (columns)
        First column is gene/probe ID
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading raw counts from {filepath}")
        
        # Read CSV with first column as index
        self.raw_counts = pd.read_csv(filepath, index_col=0)
        
        logger.info(f"Loaded {self.raw_counts.shape[0]} genes x {self.raw_counts.shape[1]} samples")
        logger.info(f"Sample IDs: {list(self.raw_counts.columns[:5])} ...")
        
        return self.raw_counts
    
    def load_processed_expression(self, filename='GSE289483_processed.csv'):
        """
        Load processed/normalized expression data.
        
        Format: genes (rows) x samples (columns)
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading processed expression from {filepath}")
        
        self.processed_expr = pd.read_csv(filepath, index_col=0)
        
        logger.info(f"Loaded {self.processed_expr.shape[0]} genes x {self.processed_expr.shape[1]} samples")
        
        return self.processed_expr
    
    def load_probe_config(self, filename='GSE289483_pkc'):
        """
        Load probe kit configuration file.
        
        PKC files contain gene annotations and probe information.
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading probe configuration from {filepath}")
        
        # PKC files might be tab-separated or have special format
        # Try different delimiters
        try:
            self.probe_config = pd.read_csv(filepath, sep='\t')
        except:
            try:
                self.probe_config = pd.read_csv(filepath, sep=',')
            except Exception as e:
                logger.warning(f"Could not parse PKC file: {e}")
                self.probe_config = None
                return None
        
        logger.info(f"Loaded probe config with {len(self.probe_config)} entries")
        if self.probe_config is not None:
            logger.info(f"Columns: {list(self.probe_config.columns)}")
        
        return self.probe_config
    
    def create_metadata(self):
        """
        Create metadata DataFrame from sample names.
        
        Sample names often encode useful information:
        DSP_1001660021707_D_A02
        - DSP: Platform
        - Numbers: Slide/batch ID  
        - Letters: Well position
        """
        if self.raw_counts is None:
            logger.error("Load raw counts first!")
            return None
        
        sample_ids = list(self.raw_counts.columns)
        
        # Parse sample IDs
        metadata_dict = {
            'sample_id': sample_ids,
            'slide_id': [sid.split('_')[1] if len(sid.split('_')) > 1 else 'unknown' 
                        for sid in sample_ids],
            'well_position': [sid.split('_')[-1] if len(sid.split('_')) > 2 else 'unknown'
                             for sid in sample_ids]
        }
        
        self.metadata = pd.DataFrame(metadata_dict)
        self.metadata['roi_id'] = self.metadata.index
        
        # Add placeholder spatial coordinates (will be random for now)
        np.random.seed(42)
        self.metadata['x_coord_um'] = np.random.uniform(100, 2000, len(self.metadata))
        self.metadata['y_coord_um'] = np.random.uniform(100, 1500, len(self.metadata))
        self.metadata['area_um2'] = np.random.uniform(40000, 160000, len(self.metadata))
        
        # Assign tissue regions based on slide (just for simulation)
        tissue_types = ['Tumor', 'Stroma', 'Immune_Aggregate', 'Normal']
        self.metadata['tissue_region'] = np.random.choice(tissue_types, len(self.metadata))
        
        logger.info(f"Created metadata for {len(self.metadata)} samples")
        
        return self.metadata
    
    def get_integrated_data(self, use_processed=True, n_genes=None):
        """
        Get integrated data ready for IF2RNA training.
        
        Args:
            use_processed: Use processed (normalized) expression instead of raw counts
            n_genes: Number of top variable genes to select (None = all)
            
        Returns:
            dict with:
                - roi_ids: Sample/ROI identifiers
                - gene_expression: Gene expression matrix (samples x genes)
                - gene_names: Gene identifiers
                - spatial_coords: DataFrame with ROI spatial info
                - metadata: Full metadata
        """
        if self.raw_counts is None:
            logger.error("Load data first using load_raw_counts()")
            return None
        
        # Choose expression data
        if use_processed and self.processed_expr is not None:
            expr_data = self.processed_expr
            logger.info("Using processed/normalized expression data")
        else:
            expr_data = self.raw_counts
            logger.info("Using raw count data")
        
        # Transpose: we want samples x genes
        expr_matrix = expr_data.T
        
        # Select top variable genes if requested
        if n_genes is not None and n_genes < expr_matrix.shape[1]:
            logger.info(f"Selecting top {n_genes} most variable genes")
            gene_vars = expr_matrix.var(axis=0)
            top_genes = gene_vars.nlargest(n_genes).index
            expr_matrix = expr_matrix[top_genes]
        
        # Create metadata if not exists
        if self.metadata is None:
            self.create_metadata()
        
        # Log-normalize if using raw counts
        if not use_processed:
            logger.info("Applying log1p transformation")
            expr_matrix = np.log1p(expr_matrix)
        
        integrated = {
            'roi_ids': list(expr_matrix.index),
            'gene_expression': expr_matrix,
            'gene_names': list(expr_matrix.columns),
            'spatial_coords': self.metadata,
            'metadata': {
                'n_rois': len(expr_matrix),
                'n_genes': len(expr_matrix.columns),
                'dataset': 'GSE289483',
                'description': 'Pulmonary Pleomorphic Carcinoma'
            }
        }
        
        logger.info(f"✅ Integrated data ready: {integrated['metadata']['n_rois']} ROIs, "
                   f"{integrated['metadata']['n_genes']} genes")
        
        return integrated


def test_parser():
    """Test the parser with real data"""
    print("="*60)
    print("Testing Real GeoMx Data Parser")
    print("="*60)
    
    data_dir = Path("/Users/siddarthchilukuri/Documents/GitHub/IF2RNA/data/geomx_datasets/GSE289483")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run the download script first!")
        return
    
    # Create parser
    parser = RealGeoMxDataParser(data_dir)
    
    # Load data
    print("\n1. Loading raw counts...")
    raw_counts = parser.load_raw_counts()
    print(f"   Shape: {raw_counts.shape}")
    print(f"   First few genes: {list(raw_counts.index[:5])}")
    
    print("\n2. Loading processed expression...")
    processed = parser.load_processed_expression()
    print(f"   Shape: {processed.shape}")
    
    print("\n3. Creating metadata...")
    metadata = parser.create_metadata()
    print(f"   Samples: {len(metadata)}")
    print(f"   Columns: {list(metadata.columns)}")
    print(f"   Tissue regions: {metadata['tissue_region'].value_counts().to_dict()}")
    
    print("\n4. Getting integrated data...")
    integrated = parser.get_integrated_data(use_processed=True, n_genes=1000)
    print(f"   ROIs: {integrated['metadata']['n_rois']}")
    print(f"   Genes: {integrated['metadata']['n_genes']}")
    print(f"   Expression matrix shape: {integrated['gene_expression'].shape}")
    
    print("\n5. Sample expression statistics:")
    expr = integrated['gene_expression']
    print(f"   Mean expression: {expr.values.mean():.2f}")
    print(f"   Std expression: {expr.values.std():.2f}")
    print(f"   Min: {expr.values.min():.2f}, Max: {expr.values.max():.2f}")
    
    print("\n" + "="*60)
    print("✅ Parser test complete!")
    print("="*60)
    
    return parser, integrated


if __name__ == '__main__':
    parser, integrated = test_parser()
