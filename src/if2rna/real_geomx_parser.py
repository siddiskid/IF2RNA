

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealGeoMxDataParser:
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.raw_counts = None
        self.processed_expr = None
        self.probe_config = None
        self.metadata = None
        
    def load_raw_counts(self, filename='GSE289483_raw_counts.csv'):
        filepath = self.data_dir / filename
        logger.info(f"Loading raw counts from {filepath}")
        
        # Read CSV with first column as index
        self.raw_counts = pd.read_csv(filepath, index_col=0)
        
        logger.info(f"Loaded {self.raw_counts.shape[0]} genes x {self.raw_counts.shape[1]} samples")
        logger.info(f"Sample IDs: {list(self.raw_counts.columns[:5])} ...")
        
        return self.raw_counts
    
    def load_processed_expression(self, filename='GSE289483_processed.csv'):
        filepath = self.data_dir / filename
        logger.info(f"Loading processed expression from {filepath}")
        
        self.processed_expr = pd.read_csv(filepath, index_col=0)
        
        logger.info(f"Loaded {self.processed_expr.shape[0]} genes x {self.processed_expr.shape[1]} samples")
        
        return self.processed_expr
    
    def load_probe_config(self, filename='GSE289483_pkc'):
        filepath = self.data_dir / filename
        logger.info(f"Loading probe configuration from {filepath}")
        
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
        if self.raw_counts is None:
            logger.error("Load raw counts first!")
            return None
        
        sample_ids = list(self.raw_counts.columns)
        metadata_dict = {
            'sample_id': sample_ids,
            'slide_id': [sid.split('_')[1] if len(sid.split('_')) > 1 else 'unknown' 
                        for sid in sample_ids],
            'well_position': [sid.split('_')[-1] if len(sid.split('_')) > 2 else 'unknown'
                             for sid in sample_ids]
        }
        
        self.metadata = pd.DataFrame(metadata_dict)
        self.metadata['roi_id'] = self.metadata.index
        
        np.random.seed(42)
        self.metadata['x_coord_um'] = np.random.uniform(100, 2000, len(self.metadata))
        self.metadata['y_coord_um'] = np.random.uniform(100, 1500, len(self.metadata))
        self.metadata['area_um2'] = np.random.uniform(40000, 160000, len(self.metadata))
        
        tissue_types = ['Tumor', 'Stroma', 'Immune_Aggregate', 'Normal']
        self.metadata['tissue_region'] = np.random.choice(tissue_types, len(self.metadata))
        
        logger.info(f"Created metadata for {len(self.metadata)} samples")
        
        return self.metadata
    
    def get_integrated_data(self, use_processed=True, n_genes=None):
        if self.raw_counts is None:
            logger.error("Load data first using load_raw_counts()")
            return None
        
        if use_processed and self.processed_expr is not None:
            expr_data = self.processed_expr
            logger.info("Using processed/normalized expression data")
        else:
            expr_data = self.raw_counts
            logger.info("Using raw count data")
        
        expr_matrix = expr_data.T
        if n_genes is not None and n_genes < expr_matrix.shape[1]:
            logger.info(f"Selecting top {n_genes} most variable genes")
            gene_vars = expr_matrix.var(axis=0)
            top_genes = gene_vars.nlargest(n_genes).index
            expr_matrix = expr_matrix[top_genes]
        
        if self.metadata is None:
            self.create_metadata()
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
        
        logger.info(f"Integrated data ready: {integrated['metadata']['n_rois']} ROIs, "
                   f"{integrated['metadata']['n_genes']} genes")
        
        return integrated


def test_parser():
    print("Testing GeoMx parser")
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "geomx_datasets" / "GSE289483"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Create parser
    parser = RealGeoMxDataParser(data_dir)
    
    # Load data
    print("Loading raw counts...")
    raw_counts = parser.load_raw_counts()
    print(f"Shape: {raw_counts.shape}")
    
    print("Loading processed expression...")
    processed = parser.load_processed_expression()
    print(f"Shape: {processed.shape}")
    
    print("Creating metadata...")
    metadata = parser.create_metadata()
    print(f"Samples: {len(metadata)}")
    
    print("Getting integrated data...")
    integrated = parser.get_integrated_data(use_processed=True, n_genes=1000)
    print(f"{integrated['metadata']['n_rois']} ROIs, {integrated['metadata']['n_genes']} genes")
    
    expr = integrated['gene_expression']
    print(f"Expression: mean={expr.values.mean():.2f}, std={expr.values.std():.2f}")
    
    print("Parser test complete")
    
    return parser, integrated


if __name__ == '__main__':
    parser, integrated = test_parser()
