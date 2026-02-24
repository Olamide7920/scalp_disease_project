#!/usr/bin/env python3
"""
QUICK REFERENCE CARD
====================

Copy & paste commands for common tasks.
"""

# ============================================================================
# 1. TRAIN WITH WATERMARK PENALTIES (Recommended First Try)
# ============================================================================

"""
python train_with_contrastive.py \
    --data_dir src/self/hair_spotter/training_hair \
    --epochs 10 \
    --watermark_penalty 2.0 \
    --logo_penalty 3.0 \
    --verify
"""

# ============================================================================
# 2. TRAIN WITH FULL CONTRASTIVE LEARNING
# ============================================================================

"""
python train_with_contrastive.py \
    --data_dir src/self/hair_spotter/training_hair \
    --epochs 15 \
    --use_contrastive \
    --watermark_penalty 2.0 \
    --logo_penalty 3.0 \
    --verify \
    --output_weights hair_classifier_full.pth
"""

# ============================================================================
# 3. DETECT WATERMARKS IN IMAGES
# ============================================================================

python_code_1 = """
from contrastive_loss import WatermarkDetector
import torch
from torchvision import transforms
from PIL import Image

detector = WatermarkDetector()
img = Image.open('your_image.jpg').convert('RGB')
img_tensor = transforms.ToTensor()(img)

has_watermark = detector.has_watermark(img_tensor)
has_logo = detector.has_dermnet_logo(img_tensor)

print(f"Watermark: {has_watermark}")
print(f"Logo: {has_logo}")
"""

# ============================================================================
# 4. ANALYZE MODEL PREDICTIONS
# ============================================================================

python_code_2 = """
from classification_verifier import ClassificationVerifier
from hair_spotter_class import HairClassifier

hc = HairClassifier('src/self/hair_spotter/training_hair')
hc.load_weights('hair_classifier_contrastive.pth')

verifier = ClassificationVerifier(hc.model, device='cuda')

# Single image
result = verifier.get_confidence('image.jpg')
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
"""

# ============================================================================
# 5. GENERATE VERIFICATION REPORT
# ============================================================================

python_code_3 = """
from classification_verifier import ClassificationVerifier
from hair_spotter_class import HairClassifier

hc = HairClassifier('src/self/hair_spotter/training_hair')
hc.load_weights('hair_classifier_contrastive.pth')

verifier = ClassificationVerifier(hc.model, device='cuda')

report = verifier.generate_report({
    'hair': 'src/self/hair_spotter/training_hair/hair',
    'not_hair': 'src/self/hair_spotter/training_hair/not_hair'
})

print(report)
"""

# ============================================================================
# 6. VISUALIZE ATTENTION MAPS (GRAD-CAM)
# ============================================================================

python_code_4 = """
from classification_verifier import ClassificationVerifier
from hair_spotter_class import HairClassifier

hc = HairClassifier('src/self/hair_spotter/training_hair')
hc.load_weights('hair_classifier_contrastive.pth')

verifier = ClassificationVerifier(hc.model, device='cuda')

# Create visualization showing what model pays attention to
verifier.visualize_predictions('image.jpg', output_path='viz.png')

# Get raw Grad-CAM data
cam_info = verifier.grad_cam('image.jpg')
sal_info = verifier.saliency_map('image.jpg')
"""

# ============================================================================
# 7. BATCH ANALYSIS
# ============================================================================

python_code_5 = """
from classification_verifier import ClassificationVerifier
from hair_spotter_class import HairClassifier

hc = HairClassifier('src/self/hair_spotter/training_hair')
hc.load_weights('hair_classifier_contrastive.pth')

verifier = ClassificationVerifier(hc.model)

# Analyze all images in a folder
analysis = verifier.analyze_batch(
    'src/self/hair_spotter/training_hair/hair',
    class_name='hair'
)

print(f"Total: {analysis['total_images']}")
print(f"Correct: {analysis['correct']}")
print(f"Incorrect: {analysis['incorrect']}")
print(f"Low confidence: {len(analysis['low_confidence'])}")
print(f"Misclassified: {len(analysis['misclassified'])}")
"""

# ============================================================================
# 8. CUSTOM LOSS FUNCTION
# ============================================================================

python_code_6 = """
from contrastive_loss import ContrastiveHairLoss
import torch

# Create loss
loss_fn = ContrastiveHairLoss(
    temperature=0.07,
    watermark_penalty=2.0,
    logo_penalty=3.0,
    use_contrastive=True
)

# Compute loss
logits = torch.randn(16, 2)
labels = torch.tensor([0, 1, 0, 1, ...])
images = torch.rand(16, 3, 224, 224)

loss = loss_fn(logits, labels, images=images)
print(f"Loss: {loss.item():.4f}")
"""

# ============================================================================
# 9. COMPARE TWO MODELS
# ============================================================================

python_code_7 = """
from classification_verifier import ClassificationVerifier
from hair_spotter_class import HairClassifier

# Load two models
hc_old = HairClassifier('src/self/hair_spotter/training_hair')
hc_old.load_weights('hair_classifier_weights_epoch1.pth')

hc_new = HairClassifier('src/self/hair_spotter/training_hair')
hc_new.load_weights('hair_classifier_contrastive.pth')

# Compare on test data
verifier_old = ClassificationVerifier(hc_old.model)
verifier_new = ClassificationVerifier(hc_new.model)

report_old = verifier_old.analyze_batch('test_folder', 'hair')
report_new = verifier_new.analyze_batch('test_folder', 'hair')

print("Old Model:")
print(f"  Accuracy: {100*report_old['correct']/report_old['total_images']:.1f}%")
print("\\nNew Model:")
print(f"  Accuracy: {100*report_new['correct']/report_new['total_images']:.1f}%")
"""

# ============================================================================
# 10. SET UP DATASET WITH PENALTY IMAGES
# ============================================================================

python_code_8 = """
from pathlib import Path

# Create directories for penalty images
base = Path('src/self/hair_spotter/training_hair')

# DermNet logos
logos_dir = base / 'not_hair' / 'dermnet_logos'
logos_dir.mkdir(parents=True, exist_ok=True)

# Watermarked images
watermarks_dir = base / 'watermarks'
watermarks_dir.mkdir(parents=True, exist_ok=True)

# Now copy images:
# - DermNet logo images → dermnet_logos/
# - Watermarked images → watermarks/
# - During training, these will get 3x and 2x penalties respectively

print(f"Created {logos_dir}")
print(f"Created {watermarks_dir}")
"""

# ============================================================================
# PARAMETER TUNING GUIDE
# ============================================================================

parameter_guide = """
HYPERPARAMETER TUNING
====================

1. Watermark Detection Sensitivity:
   - threshold=0.10  → Very sensitive (catches watermarks easily)
   - threshold=0.15  → Default (good balance)
   - threshold=0.25  → Strict (only obvious watermarks)
   
   Use higher threshold if too many false positives

2. Penalty Values:
   - watermark_penalty=1.5  → Weak penalty (less aggressive)
   - watermark_penalty=2.0  → Default (recommended)
   - watermark_penalty=3.0  → Strong penalty (more aggressive)
   
   Increase if model still learning from watermarks
   Decrease if losing too much training data

3. Logo Penalty:
   - logo_penalty=2.0  → Same as watermarks
   - logo_penalty=3.0  → Default (stronger for logos)
   - logo_penalty=4.0  → Very strong (almost ignore logo images)
   
   Use 3.0+ because logos are more distracting than watermarks

4. Training:
   - epochs=5   → Quick test
   - epochs=10  → Default
   - epochs=20  → Long training (diminishing returns after 10)
   
   - lr=0.001   → Default (safe)
   - lr=0.0001  → Very conservative
   - lr=0.01    → Aggressive (may overshoot)

5. Contrastive Loss:
   - temperature=0.05  → Sharp distribution (strong contrast)
   - temperature=0.07  → Default
   - temperature=0.10  → Soft distribution (easier learning)
"""

# ============================================================================
# QUICK DIAGNOSTICS
# ============================================================================

diagnostics = """
QUICK DIAGNOSTICS
=================

1. Check if watermarks are being detected:
   python -c "
   from contrastive_loss import create_penalty_info
   import torch
   
   img = torch.rand(1, 3, 224, 224)
   info = create_penalty_info(img)
   print(f'Watermark: {info[\"watermarks\"]}')
   print(f'Logo: {info[\"logos\"]}')
   "

2. Check training loss trend:
   Look at training_curves_contrastive.png after training

3. Check model accuracy:
   python train_with_contrastive.py ... --verify
   → Generates verification_report.txt

4. Debug specific image:
   python -c "
   from classification_verifier import ClassificationVerifier
   
   verifier = ClassificationVerifier(model)
   result = verifier.get_confidence('image.jpg')
   print(f'Confidence: {result[\"confidence\"]:.2%}')
   "

5. Find misclassified images:
   python -c "
   report = verifier.analyze_batch('folder', 'class')
   for item in report['misclassified']:
       print(f'{item[\"image\"]}: {item[\"confidence\"]:.2%}')
   "
"""

# ============================================================================
# COMMON ERRORS & SOLUTIONS
# ============================================================================

errors = """
COMMON ERRORS & SOLUTIONS
=========================

Error: "ModuleNotFoundError: No module named 'cv2'"
Fix:   pip install opencv-python

Error: "CUDA out of memory"
Fix:   python train_with_contrastive.py ... --batch_size 8

Error: "Watermark not detected"
Fix:   Adjust threshold in WatermarkDetector.has_watermark()

Error: "Grad-CAM not working"
Fix:   Ensure model is ResNet18 with layer4 defined

Error: "Training very slow"
Fix:   This is normal with penalties (10-15% slower)
       Use num_workers=4 for faster data loading

Error: "Loss getting higher"
Fix:   This is GOOD! Watermarked images have higher loss
       by design. Check if accuracy is improving instead.

Error: "No images in verification"
Fix:   Ensure directory structure is:
       training_hair/
       ├── hair/
       └── not_hair/
"""

# ============================================================================
# MAIN FUNCTION
# ============================================================================

if __name__ == '__main__':
    import sys
    
    print("=" * 80)
    print("QUICK REFERENCE CARD - CONTRASTIVE LEARNING WITH WATERMARK PENALTIES")
    print("=" * 80)
    
    print("\n📚 PYTHON CODE EXAMPLES:\n")
    
    print("1. Detect Watermarks:")
    print(python_code_1)
    
    print("\n2. Single Image Prediction:")
    print(python_code_2)
    
    print("\n3. Generate Report:")
    print(python_code_3)
    
    print("\n4. Visualize Attention:")
    print(python_code_4)
    
    print("\n5. Batch Analysis:")
    print(python_code_5)
    
    print("\n" + "=" * 80)
    print("For command-line training, see top of this file")
    print("For troubleshooting, run with --help")
    print("For full docs, see CONTRASTIVE_README.md")
    print("=" * 80 + "\n")
