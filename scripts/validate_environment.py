#!/usr/bin/env python3

import sys
import importlib.util

def check_package(package_name, required_version=None):
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False, "Not installed"
        
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'Unknown')
        return True, version
    except Exception as e:
        return False, str(e)

def main():
    print("IF2RNA Environment Validation")
    print("=" * 40)
    print(f"Python version: {sys.version}")
    print()
    
    required_packages = [
        ("torch", "1.7.0"),
        ("torchvision", "0.8.0"),
        ("numpy", "1.19.0"),
        ("pandas", "1.2.0"),
        ("sklearn", "0.24.0"),
        ("cv2", None),
        ("PIL", None),
        ("h5py", "3.1.0"),
        ("matplotlib", "3.3.0"),
        ("tqdm", "4.50.0"),
    ]
    
    all_good = True
    
    for package, min_version in required_packages:
        is_installed, version_info = check_package(package)
        status = "✓" if is_installed else "✗"
        print(f"{status} {package:15} {version_info}")
        if not is_installed:
            all_good = False
    
    print("\n" + "=" * 40)
    if all_good:
        print("Environment OK")
    else:
        print("Missing packages")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
