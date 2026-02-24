#!/usr/bin/env python3
"""
STARTUP SCRIPT
==============

Run this to verify everything is installed and ready to use.
"""

import sys
import importlib
from pathlib import Path

def check_imports():
    """Check all required imports."""
    print("\n" + "=" * 80)
    print("CHECKING IMPORTS")
    print("=" * 80)
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
    }
    
    missing = []
    for module, name in required.items():
        try:
            importlib.import_module(module)
            print(f"✓ {name:<20} OK")
        except ImportError:
            print(f"✗ {name:<20} MISSING")
            missing.append(module)
    
    if missing:
        print("\n⚠️  Missing packages. Install with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All imports OK")
    return True


def check_files():
    """Check all new files exist."""
    print("\n" + "=" * 80)
    print("CHECKING FILES")
    print("=" * 80)
    
    base_dir = Path(__file__).parent
    
    required_files = [
        'hair_spotter_class.py',
        'contrastive_loss.py',
        'classification_verifier.py',
        'train_with_contrastive.py',
        'examples.py',
        'dataset_setup.py',
        'INTEGRATION_GUIDE.py',
        'QUICK_REFERENCE.py',
        'CONTRASTIVE_README.md',
        'IMPLEMENTATION_SUMMARY.md',
    ]
    
    missing = []
    for fname in required_files:
        fpath = base_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"✓ {fname:<35} ({size:,} bytes)")
        else:
            print(f"✗ {fname:<35} MISSING")
            missing.append(fname)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    
    print("\n✓ All files present")
    return True


def check_functionality():
    """Test basic functionality."""
    print("\n" + "=" * 80)
    print("TESTING FUNCTIONALITY")
    print("=" * 80)
    
    try:
        import torch
        from contrastive_loss import WatermarkDetector, NTXentLoss
        print("✓ WatermarkDetector imported")
        
        detector = WatermarkDetector()
        img = torch.rand(3, 224, 224)
        wm = detector.has_watermark(img)
        print("✓ Watermark detection works")
        
        logo = detector.has_dermnet_logo(img)
        print("✓ Logo detection works")
        
        # Test NTXentLoss
        loss = NTXentLoss(temperature=0.07)
        z_i = torch.randn(4, 128)
        z_j = torch.randn(4, 128)
        loss_val = loss(z_i, z_j)
        print("✓ NTXentLoss works")
        
        from contrastive_loss import ContrastiveHairLoss
        combined_loss = ContrastiveHairLoss()
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        loss_val = combined_loss(logits, labels)
        print("✓ ContrastiveHairLoss works")
        
        print("\n✓ All functionality checks passed")
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_next_steps():
    """Show what to do next."""
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    
    steps = """
1. PREPARE YOUR DATA
   Organize training data in ImageFolder format:
   training_hair/
   ├── hair/
   │   ├── curly/
   │   ├── straight/
   │   └── wavy/
   └── not_hair/
       ├── train_set/
       └── test_set/

2. QUICK START (Train with penalties)
   python train_with_contrastive.py \\
       --data_dir src/self/hair_spotter/training_hair \\
       --epochs 10 \\
       --watermark_penalty 2.0 \\
       --logo_penalty 3.0

3. RUN EXAMPLES
   python examples.py

4. VERIFY SETUP
   python -c "from contrastive_loss import WatermarkDetector; print('OK!')"

5. READ DOCUMENTATION
   See: CONTRASTIVE_README.md
   See: IMPLEMENTATION_SUMMARY.md

6. CHECK QUICK REFERENCE
   python QUICK_REFERENCE.py

7. GET HELP
   python train_with_contrastive.py --help
   python INTEGRATION_GUIDE.py 1  # Minimal integration
    """
    
    print(steps)


def main():
    """Run all checks."""
    print("\n" * 2)
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "CONTRASTIVE LEARNING - STARTUP VERIFICATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    all_ok = True
    
    all_ok = check_imports() and all_ok
    all_ok = check_files() and all_ok
    all_ok = check_functionality() and all_ok
    
    if all_ok:
        show_next_steps()
        
        print("\n" + "=" * 80)
        print("✓ SYSTEM READY - You can start training!")
        print("=" * 80 + "\n")
        
        return 0
    else:
        print("\n" + "=" * 80)
        print("✗ SYSTEM NOT READY - Fix issues above")
        print("=" * 80 + "\n")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
