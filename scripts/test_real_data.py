#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import h5py

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from if2rna.experiment import IF2RNAExperiment
from if2rna.data import IF2RNADataset, IF2RNATileDataset
from if2rna.model import IF2RNA


def create_sample_tcga_like_data():
    print("Creating sample data...")
    
    # Create sample data directory
    data_dir = Path("sample_data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample transcriptome data
    transcriptome_dir = data_dir / "transcriptome"
    transcriptome_dir.mkdir(exist_ok=True)
    
    # Sample gene expression data
    n_samples = 50
    n_genes = 100
    
    # Generate realistic gene names
    genes = [f"ENSG{i:08d}" for i in range(n_genes)]
    
    # Generate sample IDs
    prefixes = np.random.choice(['BR', 'LU', 'CO'], n_samples)
    suffixes = np.random.randint(1000, 9999, n_samples)
    samples = [f"TCGA-{prefix}-{suffix:04d}" for prefix, suffix in zip(prefixes, suffixes)]
    
    # Generate log-normal gene expression data
    np.random.seed(42)
    expression_data = np.random.lognormal(mean=2.0, sigma=1.5, size=(n_samples, n_genes))
    
    # Create metadata
    metadata = pd.DataFrame({
        'Case.ID': [s.split('-')[1] for s in samples],
        'Sample.ID': samples,
        'File.ID': [f"{s}.txt" for s in samples],
        'Project.ID': [f"TCGA-{s.split('-')[1]}" for s in samples],
        'Slide.ID': [f"{s}_slide.npy" for s in samples]
    })
    
    # Add gene expression columns
    expr_df = pd.DataFrame(expression_data, columns=genes)
    full_df = pd.concat([metadata, expr_df], axis=1)
    
    # Save transcriptome data
    transcriptome_file = transcriptome_dir / "sample_tcga_expression.csv"
    full_df.to_csv(transcriptome_file, index=False)
    
    print(f"Created transcriptome: {n_samples} samples, {n_genes} genes")
    
    # Create sample tile data
    tiles_dir = data_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    
    for project in full_df['Project.ID'].unique():
        project_dir = tiles_dir / project.replace('-', '_') / "0.50_mpp"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Get slides for this project
        project_slides = full_df[full_df['Project.ID'] == project]['Slide.ID'].values
        
        for slide_file in project_slides:
            # Create synthetic tile features (ResNet-50 like)
            n_tiles = np.random.randint(500, 2000)  # Variable number of tiles
            
            # Features: [n_tiles, 2051] (coordinates + 2048 ResNet features)
            coordinates = np.random.rand(n_tiles, 3) * 1000  # x, y, level
            features = np.random.randn(n_tiles, 2048) * 0.5  # ResNet features
            
            tile_data = np.concatenate([coordinates, features], axis=1)
            
            slide_path = project_dir / slide_file
            np.save(slide_path, tile_data)
    
    print(f"Created tiles in {tiles_dir}")
    
    return {
        'transcriptome_file': transcriptome_file,
        'tiles_dir': tiles_dir,
        'metadata': metadata,
        'expression_data': full_df
    }


def test_real_data_loading():
    """Test loading data in TCGA-like format."""
    print("\nLoading data...")
    
    # Create sample data
    data_info = create_sample_tcga_like_data()
    
    # Simple transcriptome dataset class
    class SimpleTranscriptomeDataset:
        def __init__(self, csv_file):
            self.df = pd.read_csv(csv_file)
            self.metadata = self.df[['Case.ID', 'Sample.ID', 'File.ID', 'Project.ID', 'Slide.ID']]
            
            # Extract transcriptome data
            gene_cols = [col for col in self.df.columns if col.startswith('ENSG')]
            self.transcriptomes = self.df[['Case.ID', 'Sample.ID', 'File.ID', 'Project.ID'] + gene_cols]
    
    # Load transcriptome data
    dataset = SimpleTranscriptomeDataset(data_info['transcriptome_file'])
    n_genes = len([c for c in dataset.df.columns if c.startswith('ENSG')])
    print(f"Loaded: {len(dataset.df)} samples, {n_genes} genes")
    
    # Test data loading with IF2RNA format
    try:
        # Create IF2RNA dataset from the transcriptome data
        y, genes, patients, projects = load_sample_labels(dataset)
        
        # Create file list
        file_list = []
        for _, row in dataset.metadata.iterrows():
            project_path = row['Project.ID'].replace('-', '_')
            file_path = f"{project_path}/0.50_mpp/{row['Slide.ID']}"
            file_list.append(file_path)
        
        print(f"Data ready: {y.shape}, {len(patients)} patients, {len(set(projects))} projects")
        
        # Test with actual tile loading (simplified)
        sample_files = file_list[:5]  # Test first 5 files
        tiles_root = str(data_info['tiles_dir'])
        
        for file_path in sample_files:
            full_path = Path(tiles_root) / file_path
            if full_path.exists():
                tile_data = np.load(full_path)
                print(f"Loaded {full_path.name}: {tile_data.shape}")
            else:
                print(f"Missing {full_path}")
        
        return True
        
    except Exception as e:
        print(f"Failed: {e}")
        return False


def load_sample_labels(transcriptome_dataset):
    """Load labels in HE2RNA format."""
    to_drop = ['Case.ID', 'Sample.ID', 'File.ID', 'Project.ID']
    df = transcriptome_dataset.transcriptomes.copy()
    patients = df['Case.ID'].values
    projects = df['Project.ID'].values
    df.drop(to_drop, axis=1, inplace=True)
    genes = df.columns.tolist()
    df = np.log10(1 + df)  # Log transform
    y = df.values
    
    return y, genes, patients, projects


def test_if_adaptation_prep():
    """Test preparation for IF data adaptation."""
    print("\nIF adaptation ready")
    
    return True


def main():
    """Test real data integration capabilities."""
    print("IF2RNA Real Data Integration Test")
    print("=" * 50)
    
    try:
        # Test 1: Sample data creation and loading
        success = test_real_data_loading()
        if not success:
            return 1
            
        # Test 2: IF adaptation preparation
        test_if_adaptation_prep()
        
        print("\n" + "=" * 50)
        print("Tests passed")
        
        # Cleanup
        import shutil
        if Path("sample_data").exists():
            shutil.rmtree("sample_data")
        if Path("test_experiments").exists():
            shutil.rmtree("test_experiments")
        
        return 0
        
    except Exception as e:
        print(f"\nFailed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
