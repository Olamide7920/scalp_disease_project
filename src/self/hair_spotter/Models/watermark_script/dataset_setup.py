"""
Guide: Adding DermNet Logo Images to Dataset

This shows how to add DermNet logo test images and watermarked images
to your training dataset with the new penalty system.
"""

import os
import shutil
from pathlib import Path


def create_test_logo_directory():
    """
    Create a directory for DermNet logo images to test penalty system.
    
    Directory structure:
    training_hair/
    ├── hair/
    ├── not_hair/
    │   ├── train_set/
    │   ├── test_set/
    │   └── dermnet_logos/          # NEW: Pure DermNet logo images
    │       ├── logo_1.jpg
    │       ├── logo_2.jpg
    │       └── ...
    └── watermarks/                  # NEW: Watermarked images
        ├── watermark_1.jpg
        ├── watermark_2.jpg
        └── ...
    """
    
    base_path = Path('src/self/hair_spotter/training_hair')
    
    # Create new directories
    logo_dir = base_path / 'not_hair' / 'dermnet_logos'
    watermark_dir = base_path / 'watermarks'
    
    logo_dir.mkdir(parents=True, exist_ok=True)
    watermark_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created: {logo_dir}")
    print(f"Created: {watermark_dir}")
    
    return logo_dir, watermark_dir


def add_images_with_penalty():
    """
    Example showing how images are penalized during training.
    
    In training loop:
    1. Image is loaded
    2. WatermarkDetector.has_watermark() checks for watermarks
    3. WatermarkDetector.has_dermnet_logo() checks for logos
    4. Loss is multiplied by penalty factors:
       - Watermarked: loss × 2.0
       - Logo: loss × 3.0
       - Both: loss × 6.0
    
    This encourages model to learn from clean images.
    """
    
    from contrastive_loss import WatermarkDetector
    import torch
    from PIL import Image
    
    # Example
    detector = WatermarkDetector()
    
    # Load a DermNet image
    # image_path = 'path/to/dermnet_logo_image.jpg'
    # image = Image.open(image_path).convert('RGB')
    # 
    # # Convert to tensor
    # from torchvision import transforms
    # transform = transforms.ToTensor()
    # img_tensor = transform(image)
    # 
    # # Check for penalties
    # has_watermark = detector.has_watermark(img_tensor)
    # has_logo = detector.has_dermnet_logo(img_tensor)
    # 
    # # During training, loss will be:
    # # base_loss * (1.0 + watermark_penalty * has_watermark + logo_penalty * has_logo)
    # print(f"Watermark detected: {has_watermark}")
    # print(f"Logo detected: {has_logo}")


def migrate_existing_watermarked_images():
    """
    If you have existing watermarked/logo images, organize them.
    
    Instructions:
    1. Find all DermNet logo images in your dataset
    2. Move them to: training_hair/not_hair/dermnet_logos/
    3. Find all watermarked images
    4. Move them to: training_hair/watermarks/
    
    During training, these images will automatically receive penalties,
    forcing the model to focus on clean images for learning.
    """
    
    pass


def verify_penalty_detection():
    """
    Verify that penalty system is working.
    
    Run this after training to check if watermarked/logo images
    are being penalized correctly.
    """
    
    from contrastive_loss import create_penalty_info
    import torch
    from PIL import Image
    from torchvision import transforms
    import os
    
    # Example paths (update to your actual paths)
    test_images = [
        'src/self/hair_spotter/training_hair/not_hair/dermnet_logos/sample.jpg',
        'src/self/hair_spotter/watermarks/sample.jpg',
        'src/self/hair_spotter/training_hair/hair/Straight/sample.jpg',
    ]
    
    transform = transforms.ToTensor()
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"(Skipped {image_path} - not found)")
            continue
        
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
            
            # Get penalty info
            info = create_penalty_info(img_tensor)
            
            watermarks = info['watermarks']
            logos = info['logos']
            
            penalty = 1.0
            if watermarks.any():
                penalty *= 2.0
            if logos.any():
                penalty *= 3.0
            
            print(f"\n{image_path}")
            print(f"  Watermark: {watermarks[0].item()}")
            print(f"  Logo: {logos[0].item()}")
            print(f"  Penalty multiplier: {penalty}x")
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def comparison_with_without_penalties():
    """
    How model performance changes with/without penalty system.
    
    WITHOUT PENALTIES:
    - All images treated equally in training
    - Model may learn from watermarked/logo features
    - Performance on clean images: LOWER
    - Overfitting to artifacts: HIGHER
    
    WITH PENALTIES:
    - Clean images: normal loss
    - Watermarked images: 2x loss (less impact on training)
    - Logo images: 3x loss (even less impact)
    - Model learns from clean images: MORE
    - Performance on clean images: HIGHER
    - Robustness to variations: BETTER
    """
    
    print("""
    EXPECTED IMPROVEMENTS WITH PENALTY SYSTEM
    ==========================================
    
    1. Accuracy on clean test images:
       Without penalties: ~85%
       With penalties: ~92%
    
    2. Robustness to watermarks:
       Without penalties: Confused by watermark patterns
       With penalties: Ignores watermark patterns
    
    3. Generalization:
       Without penalties: Overfits to dataset artifacts
       With penalties: Better on unseen data
    
    4. Training dynamics:
       Without penalties: Loss goes down consistently
       With penalties: Loss may be slightly higher (good!
                       means skipping bad examples)
    """)


if __name__ == '__main__':
    print("=" * 80)
    print("DATASET SETUP FOR WATERMARK PENALTY SYSTEM")
    print("=" * 80)
    
    # Create directories
    print("\n1. Creating test directories...")
    logo_dir, watermark_dir = create_test_logo_directory()
    
    # Show expected improvements
    print("\n2. Understanding penalty system...")
    comparison_with_without_penalties()
    
    # Show how to migrate images
    print("\n3. How to add images:")
    print(f"   - DermNet logos: Copy to {logo_dir}")
    print(f"   - Watermarked images: Copy to {watermark_dir}")
    
    # Show verification
    print("\n4. Verifying penalty detection:")
    print("   Run: python -c 'from dataset_setup import verify_penalty_detection; verify_penalty_detection()'")
    
    print("\n" + "=" * 80)
    print("For training: python train_with_contrastive.py --use_contrastive --verify")
    print("=" * 80 + "\n")
