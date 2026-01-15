#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from if2rna.real_if_loader import RealIFImageLoader
from if2rna.real_geomx_parser import RealGeoMxDataParser
from if2rna.hybrid_dataset import HybridIF2RNADataset, AggregatedIF2RNADataset
from if2rna.experiment import IF2RNAExperiment


def test_real_if_loader():
    print("="*60)
    print("Testing Real IF Image Loader")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / "data" / "real_geomx"
    
    if not data_dir.exists():
        print(f"Real data directory not found: {data_dir}")
        print("Please ensure real IF data is available in data/real_geomx/")
        return False
    
    loader = RealIFImageLoader(data_dir)
    
    print(f"Found {len(loader.dcc_files)} DCC files")
    print(f"Channels: {loader.get_n_channels()}")
    print(f"Channel names (first 5): {loader.get_channel_names()[:5]}")
    
    if len(loader.dcc_files) > 0:
        print("\nGenerating IF image from first ROI...")
        if_image = loader.generate_for_roi(0, tissue_type='Tumor')
        print(f"Image shape: {if_image.shape}")
        print(f"Value range: {if_image.min():.3f} - {if_image.max():.3f}")
        
        if len(loader.dcc_files) >= 2:
            print("\nGenerating batch from multiple ROIs...")
            batch = loader.generate_batch([0, 1], tissue_types=['Tumor', 'Stroma'])
            print(f"Batch shape: {batch.shape}")
    
    print("\n✓ Real IF Loader test passed")
    return True


def test_real_geomx_integration():
    print("\n" + "="*60)
    print("Testing GeoMx Data Integration")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / "data" / "real_geomx"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return False
    
    parser = RealGeoMxDataParser(data_dir)
    
    print("Loading GeoMx data...")
    parser.load_raw_counts()
    parser.create_metadata()
    integrated = parser.get_integrated_data(use_processed=False, n_genes=100)
    
    if integrated is None:
        print("Failed to load integrated data")
        return False
    
    print(f"ROIs: {integrated['metadata']['n_rois']}")
    print(f"Genes: {integrated['metadata']['n_genes']}")
    print(f"Tissue types: {integrated['spatial_coords']['tissue_region'].unique()}")
    
    print("\n✓ GeoMx integration test passed")
    return True


def test_hybrid_dataset_with_real_if():
    print("\n" + "="*60)
    print("Testing Hybrid Dataset with Real IF")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / "data" / "real_geomx"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return False
    
    parser = RealGeoMxDataParser(data_dir)
    parser.load_raw_counts()
    integrated = parser.get_integrated_data(use_processed=False, n_genes=100)
    
    if integrated is None:
        print("Failed to load integrated data")
        return False
    
    if_loader = RealIFImageLoader(data_dir)
    
    print("Creating HybridIF2RNADataset with real IF data...")
    dataset = HybridIF2RNADataset(
        integrated_data=integrated,
        if_generator=if_loader,
        n_tiles_per_roi=8,
        use_real_if=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Expression shape: {sample['expression'].shape}")
    print(f"Tissue type: {sample['tissue_type']}")
    
    print("\nCreating AggregatedIF2RNADataset...")
    agg_dataset = AggregatedIF2RNADataset(
        integrated_data=integrated,
        if_generator=if_loader,
        n_tiles_per_roi=8,
        use_real_if=True
    )
    
    print(f"Aggregated dataset length: {len(agg_dataset)}")
    
    agg_sample = agg_dataset[0]
    print(f"Aggregated tiles shape: {agg_sample['tiles'].shape}")
    print(f"Expression shape: {agg_sample['expression'].shape}")
    
    print("\n✓ Hybrid dataset test passed")
    return True


def test_real_if_experiment():
    print("\n" + "="*60)
    print("Testing Real IF Experiment")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / "data" / "real_geomx"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Will use synthetic data fallback in experiment")
    
    experiment = IF2RNAExperiment(
        experiment_name="test_real_if",
        save_dir="experiments"
    )
    
    experiment.data_config['use_real_if'] = True
    experiment.data_config['real_if_data_dir'] = str(data_dir)
    
    print("Running real IF experiment...")
    print("Note: This uses real data if available, else falls back to synthetic")
    
    try:
        results = experiment.run_real_if_experiment(
            data_dir=str(data_dir),
            n_genes=50
        )
        
        print(f"\nExperiment completed!")
        print(f"Overall correlation: {results.get('overall_correlation_mean', 'N/A')}")
        print(f"Results saved to: {experiment.save_dir}")
        
        print("\n✓ Real IF experiment test passed")
        return True
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("IF2RNA Real Data Integration Tests")
    print("="*60)
    
    tests = [
        ("Real IF Loader", test_real_if_loader),
        ("GeoMx Integration", test_real_geomx_integration),
        ("Hybrid Dataset", test_hybrid_dataset_with_real_if),
        ("Real IF Experiment", test_real_if_experiment),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            results[test_name] = False
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
    
    print("\n" + "="*60)
    print("Real IF Data Setup Complete")
    print("="*60)
    print("\nThe code now supports real IF data from DCC files.")
    print("Key changes:")
    print("  - RealIFImageLoader: Loads and processes DCC files")
    print("  - create_real_if_data(): Function to load real IF data")
    print("  - HybridIF2RNADataset: Updated with use_real_if flag")
    print("  - IF2RNAExperiment: New run_real_if_experiment() method")
    print("  - Config: Added PATH_TO_GEOMX_DATA and use_real_if flag")
    print("\nTo use real data:")
    print("  1. Place DCC files in data/real_geomx/")
    print("  2. Set use_real_if=True in dataset/experiment config")
    print("  3. Use experiment.run_real_if_experiment()")
    

if __name__ == '__main__':
    main()
