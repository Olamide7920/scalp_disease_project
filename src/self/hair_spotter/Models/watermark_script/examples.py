"""
Example usage of contrastive learning with watermark penalties.

This script demonstrates:
1. Training with contrastive loss
2. Detecting watermarks and logos
3. Verifying classification decisions
"""

import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from hair_spotter_class import HairClassifier
from contrastive_loss import ContrastiveHairLoss, WatermarkDetector, create_penalty_info
from classification_verifier import ClassificationVerifier


def example_1_watermark_detection():
    """Example: Detect watermarks in images."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: WATERMARK AND LOGO DETECTION")
    print("=" * 80)
    
    detector = WatermarkDetector()
    
    # Example with dummy image
    dummy_img = torch.rand(3, 224, 224)  # Random image in [0, 1]
    
    has_watermark = detector.has_watermark(dummy_img)
    has_logo = detector.has_dermnet_logo(dummy_img)
    
    print(f"\nDummy image analysis:")
    print(f"  Has watermark: {has_watermark}")
    print(f"  Has DermNet logo: {has_logo}")
    
    # Batch processing
    batch = torch.rand(4, 3, 224, 224)
    watermarks = detector.has_watermark(batch)
    logos = detector.has_dermnet_logo(batch)
    
    print(f"\nBatch of 4 images:")
    print(f"  Watermarks detected in: {watermarks.sum().item()} images")
    print(f"  Logos detected in: {logos.sum().item()} images")


def example_2_loss_computation():
    """Example: Compute loss with penalties."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: LOSS COMPUTATION WITH PENALTIES")
    print("=" * 80)
    
    # Create loss function
    loss_fn = ContrastiveHairLoss(
        temperature=0.07,
        watermark_penalty=2.0,
        logo_penalty=3.0,
        use_contrastive=False  # Just CE loss for this example
    )
    
    # Dummy data
    batch_size = 4
    num_classes = 2
    
    logits = torch.randn(batch_size, num_classes)
    labels = torch.tensor([0, 1, 0, 1])
    images = torch.rand(batch_size, 3, 224, 224)
    
    # Compute loss
    loss = loss_fn(logits, labels, images=images)
    
    print(f"\nLoss computation:")
    print(f"  Batch size: {batch_size}")
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Loss is higher for images with watermarks/logos (penalty applied)")


def example_3_classification_verification():
    """Example: Verify classification decisions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: CLASSIFICATION VERIFICATION")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load a pretrained model (or use an existing one)
    try:
        # Try to load existing weights
        hc = HairClassifier(data_dir='src/self/hair_spotter/training_hair')
        hc.load_weights('src/self/hair_spotter/hair_classifier_weights_epoch1.pth')
        
        verifier = ClassificationVerifier(hc.model, device=device)
        
        # Example: Get confidence for a single image
        example_image = 'src/self/hair_spotter/training_hair/hair/Straight/sample.jpg'
        
        try:
            result = verifier.get_confidence(example_image)
            print(f"\nExample prediction:")
            print(f"  Image: {example_image}")
            print(f"  Predicted: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.2%}")
        except FileNotFoundError:
            print(f"  (Example image not found, skipping)")
        
        print(f"\nVerifier capabilities:")
        print(f"  ✓ get_confidence() - Get prediction and confidence")
        print(f"  ✓ grad_cam() - Visualize attention (which regions matter)")
        print(f"  ✓ saliency_map() - Compute pixel importance")
        print(f"  ✓ analyze_batch() - Analyze all images in a folder")
        print(f"  ✓ generate_report() - Generate comprehensive report")
        print(f"  ✓ visualize_predictions() - Create visualization with attention maps")
    
    except Exception as e:
        print(f"Note: Classifier loading skipped ({type(e).__name__})")
        print("See train_with_contrastive.py for full training example")


def example_4_usage_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 80)
    print("USAGE INSTRUCTIONS")
    print("=" * 80)
    
    instructions = """
1. TRAINING WITH CONTRASTIVE LOSS:
   python train_with_contrastive.py \\
       --data_dir src/self/hair_spotter/training_hair \\
       --epochs 10 \\
       --use_contrastive \\
       --watermark_penalty 2.0 \\
       --logo_penalty 3.0 \\
       --verify

2. DETECTING WATERMARKS:
   from contrastive_loss import WatermarkDetector
   import torch
   
   detector = WatermarkDetector()
   image = torch.rand(3, 224, 224)  # Your image
   
   has_watermark = detector.has_watermark(image)
   has_logo = detector.has_dermnet_logo(image)

3. VERIFYING CLASSIFICATIONS:
   from classification_verifier import ClassificationVerifier
   
   verifier = ClassificationVerifier(model, device='cuda')
   
   # Single image
   result = verifier.get_confidence('path/to/image.jpg')
   
   # Batch analysis
   report = verifier.analyze_batch('path/to/folder', 'hair')
   
   # With visualization
   verifier.visualize_predictions('path/to/image.jpg', 
                                  output_path='visualization.png')
   
   # Grad-CAM (attention visualization)
   cam_info = verifier.grad_cam('path/to/image.jpg')
   
   # Full report
   report = verifier.generate_report({
       'hair': 'path/to/hair/images',
       'not_hair': 'path/to/not_hair/images'
   })

4. CUSTOM LOSS FUNCTION:
   from contrastive_loss import ContrastiveHairLoss
   
   loss_fn = ContrastiveHairLoss(
       temperature=0.07,
       watermark_penalty=2.0,
       logo_penalty=3.0,
       use_contrastive=True
   )
   
   # Compute loss
   loss = loss_fn(logits, labels, images=images, embeddings=embeddings)

5. UNDERSTANDING PREDICTIONS:
   - Grad-CAM shows which regions the model looks at
   - Saliency maps show pixel importance
   - low_confidence images (<70%) may need review
   - misclassified images indicate dataset/model issues
"""
    print(instructions)


if __name__ == '__main__':
    print("=" * 80)
    print("CONTRASTIVE LEARNING WITH WATERMARK PENALTIES - EXAMPLES")
    print("=" * 80)
    
    example_1_watermark_detection()
    example_2_loss_computation()
    example_3_classification_verification()
    example_4_usage_instructions()
    
    print("\n" + "=" * 80)
    print("For full training: python train_with_contrastive.py --help")
    print("=" * 80 + "\n")
