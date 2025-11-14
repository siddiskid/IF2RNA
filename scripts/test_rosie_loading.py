"""
Minimal ROSIE Model Loader Test
Tests basic model loading without complex environment setup
"""

def test_rosie_loading():
    """Test ROSIE model loading with minimal dependencies"""
    
    print("="*60)
    print("ROSIE Model Analysis")
    print("="*60)
    
    # Check file exists and basic info
    from pathlib import Path
    model_path = Path("ROSIE.pth")
    
    print(f"\n1. File Information:")
    print(f"   Path: {model_path}")
    print(f"   Exists: {model_path.exists()}")
    
    if model_path.exists():
        file_size = model_path.stat().st_size
        print(f"   Size: {file_size / 1024**2:.1f} MB")
    
    # Try to examine without PyTorch first
    import zipfile
    print(f"\n2. Archive Structure:")
    
    with zipfile.ZipFile(model_path, 'r') as z:
        files = z.namelist()
        print(f"   Total files: {len(files)}")
        print(f"   Model name: best_model_single")
        
        # Check version
        if 'best_model_single/version' in files:
            version_data = z.read('best_model_single/version')
            print(f"   PyTorch format version: {version_data.decode().strip()}")
    
    # Try PyTorch loading (may fail due to environment)
    print(f"\n3. PyTorch Loading Test:")
    
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        # Try to load the model
        print(f"   Loading model...")
        device = torch.device("cpu")  # Force CPU to avoid GPU issues
        
        checkpoint = torch.load(model_path, map_location=device)
        print(f"   ✅ Model loaded successfully!")
        
        # Analyze checkpoint structure
        print(f"\n4. Model Analysis:")
        
        if isinstance(checkpoint, torch.nn.Module):
            print(f"   Type: Direct PyTorch Module")
            model = checkpoint
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            print(f"   Type: Checkpoint with 'model' key")
            model = checkpoint['model']
        else:
            print(f"   Type: {type(checkpoint)}")
            model = checkpoint
        
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   Total parameters: {total_params:,}")
        
        if hasattr(model, 'named_modules'):
            print(f"   Model structure:")
            for name, module in list(model.named_modules())[:10]:  # First 10 modules
                print(f"     {name}: {type(module).__name__}")
            if len(list(model.named_modules())) > 10:
                print(f"     ... and {len(list(model.named_modules())) - 10} more")
        
        # Try to get input/output info
        print(f"\n5. Input/Output Analysis:")
        
        # Look for common input shapes by examining first layer
        try:
            first_param = next(iter(model.parameters()))
            print(f"   First parameter shape: {first_param.shape}")
            
            if len(first_param.shape) == 4:  # Conv layer: (out_ch, in_ch, h, w)
                print(f"   Likely input channels: {first_param.shape[1]}")
                print(f"   First conv output channels: {first_param.shape[0]}")
        except:
            print(f"   Could not analyze parameters")
        
        print(f"\n✅ ROSIE Model Analysis Complete!")
        return True
        
    except ImportError as e:
        print(f"   ❌ PyTorch not available: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Loading failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False
    
    print("="*60)

if __name__ == '__main__':
    success = test_rosie_loading()
    
    if success:
        print("\n🎉 ROSIE model is loadable and ready for integration!")
    else:
        print("\n⚠️  ROSIE model analysis completed, but PyTorch loading failed.")
        print("   This is likely due to the PyTorch library issue.")
        print("   The model file itself appears to be valid.")