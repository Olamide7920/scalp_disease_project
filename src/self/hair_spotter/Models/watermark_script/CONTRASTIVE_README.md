# Hair Classifier with Contrastive Learning and Watermark Penalties

This module implements a hair/not-hair classifier with advanced features for handling images with watermarks and DermNet logos.

## Features

### 1. **Contrastive Learning Loss** (`contrastive_loss.py`)
- **NTXentLoss**: Normalized Temperature-scaled Cross Entropy loss for contrastive learning
- **Per-sample weighting**: Apply different loss weights based on image quality
- **Contrastive + Classification**: Combine contrastive and cross-entropy losses

### 2. **Watermark & Logo Detection** (`contrastive_loss.py`)
- **WatermarkDetector.has_watermark()**: Detects watermarks using edge detection and morphological analysis
- **WatermarkDetector.has_dermnet_logo()**: Specifically detects DermNet logos in top-right and bottom regions
- **Batch processing**: Process multiple images simultaneously

### 3. **Automatic Penalty Application** (`contrastive_loss.py`)
- **ContrastiveHairLoss**: Combined loss with automatic watermark/logo penalties
  - Watermarked images: 2x loss penalty (default)
  - DermNet logos: 3x loss penalty (default)
  - Customizable penalty values

### 4. **Classification Verification & Analysis** (`classification_verifier.py`)
- **Confidence scores**: Per-image prediction confidence
- **Grad-CAM**: Visualize which image regions influenced predictions
- **Saliency maps**: Show pixel-level importance
- **Batch analysis**: Analyze all images in a folder with statistics
- **Mismatch detection**: Identify low-confidence and misclassified images
- **Report generation**: Comprehensive analysis reports with visualizations

## Installation

Requires `opencv-python`:
```bash
pip install opencv-python
```

## Quick Start

### 1. Training with Contrastive Loss

```bash
python train_with_contrastive.py \
    --data_dir src/self/hair_spotter/training_hair \
    --epochs 10 \
    --use_contrastive \
    --watermark_penalty 2.0 \
    --logo_penalty 3.0 \
    --verify
```

**Command-line arguments:**
- `--data_dir`: Path to training data (ImageFolder format)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 16)
- `--use_contrastive`: Enable contrastive loss
- `--watermark_penalty`: Penalty for watermarked images (default: 2.0)
- `--logo_penalty`: Penalty for DermNet logos (default: 3.0)
- `--verify`: Run verification after training
- `--output_weights`: Save path for trained weights

### 2. Detecting Watermarks and Logos

```python
from contrastive_loss import WatermarkDetector
import torch

detector = WatermarkDetector()

# Load image as tensor (3, 224, 224) in range [0, 1]
image = torch.rand(3, 224, 224)

# Detect watermark
has_watermark = detector.has_watermark(image, threshold=0.15)

# Detect DermNet logo
has_logo = detector.has_dermnet_logo(image)

# Batch processing
batch = torch.rand(4, 3, 224, 224)
watermarks = detector.has_watermark(batch)  # Returns tensor of bools
```

### 3. Verifying Classification Decisions

```python
from classification_verifier import ClassificationVerifier
from hair_spotter_class import HairClassifier

# Load model
hc = HairClassifier(data_dir='...')
hc.load_weights('model.pth')

# Create verifier
verifier = ClassificationVerifier(hc.model, device='cuda')

# Single image confidence
result = verifier.get_confidence('path/to/image.jpg')
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch analysis
report = verifier.analyze_batch('path/to/folder', class_name='hair')
print(report)

# Attention visualization (shows which regions matter)
cam_info = verifier.grad_cam('path/to/image.jpg')

# Pixel importance visualization
sal_info = verifier.saliency_map('path/to/image.jpg')

# Create visualization
verifier.visualize_predictions('path/to/image.jpg', 
                              output_path='viz.png')

# Full verification report
report = verifier.generate_report({
    'hair': 'path/to/hair/images',
    'not_hair': 'path/to/not_hair/images'
})
print(report)
```

### 4. Custom Loss Function

```python
from contrastive_loss import ContrastiveHairLoss
import torch

# Create loss function
loss_fn = ContrastiveHairLoss(
    temperature=0.07,
    watermark_penalty=2.0,
    logo_penalty=3.0,
    use_contrastive=True
)

# Compute loss
logits = torch.randn(batch_size, 2)  # (B, 2) for binary classification
labels = torch.tensor([0, 1, 0, 1])  # Binary labels
images = torch.rand(batch_size, 3, 224, 224)  # Raw images for watermark detection
embeddings = torch.randn(batch_size, 512)  # Feature embeddings (optional)

loss = loss_fn(logits, labels, images=images, embeddings=embeddings)
```

## Understanding Output

### Classification Verifier Report

```
================================================================================
CLASSIFICATION VERIFICATION REPORT
================================================================================

Class: HAIR
Directory: path/to/hair/images
Total images: 100
Correct: 95 (95.0%)
Incorrect: 5 (5.0%)

Confidence Stats:
  Mean: 0.9234
  Std:  0.0512
  Min:  0.7123
  Max:  0.9987

Low Confidence Images (<70%):
  - image1.jpg: 68.50%
  - image2.jpg: 65.12%

Misclassified Images:
  - bad_image.jpg: predicted not_hair (62.34%)
  - unclear_image.jpg: predicted not_hair (71.23%)
```

### Grad-CAM Visualization

Shows which regions of the image the model looks at:
- **Red/Hot colors**: Important regions
- **Blue/Cold colors**: Less important regions

### Saliency Map

Shows pixel-level importance:
- **Bright areas**: High pixel importance
- **Dark areas**: Low pixel importance

## Advanced Usage

### Training the Enhanced Classifier

```python
from train_with_contrastive import EnhancedHairClassifier

classifier = EnhancedHairClassifier(data_dir='...')
train_loader, val_loader, test_loader = classifier.get_dataloaders_with_raw()

history = classifier.train_with_contrastive(
    train_loader,
    val_loader,
    epochs=10,
    lr=0.001,
    use_contrastive=True,
    watermark_penalty=2.0,
    logo_penalty=3.0
)
```

### Feature Extraction

```python
# Get features from model (useful for contrastive learning)
features = classifier.get_features(image_tensor)  # (B, 512)
```

### Comparing Different Image Sets

```python
# Compare predictions on two different datasets
comparison = verifier.compare_predictions(
    'path/to/set1', 'path/to/set2',
    class_1='hair', class_2='not_hair'
)
print(comparison['set_1']['report'])
print(comparison['set_2']['report'])
```

## Watermark Detection Details

### Algorithm
1. **Edge Detection**: Uses Canny edge detection to find boundaries
2. **Morphological Operations**: Dilates edges to find text regions
3. **Contour Analysis**: Looks for characteristic watermark regions
4. **Location Heuristics**: Watermarks typically appear in corners or edges

### Parameters
- `threshold`: Edge ratio threshold (default: 0.15)
  - Lower = more sensitive to watermarks
  - Higher = only obvious watermarks detected

### DermNet Logo Detection
- Looks for text in top-right quadrant and bottom area
- Checks for moderate text coverage (5-30% in top-right, 2-15% in bottom)
- Returns True if pattern matches typical DermNet logo placement

## Loss Function Details

### NTXentLoss (Contrastive)
Uses normalized temperature-scaled cross-entropy loss:
- **Temperature**: Controls sharpness of distribution (default: 0.07)
- **Positive pairs**: Same sample with different augmentations
- **Negative pairs**: Different samples in the batch
- **Output**: Maximizes similarity of positive pairs, minimizes negative

### Quality Weights
Applied per-sample in the loss:
- Base weight: 1.0
- If watermarked: × watermark_penalty (default: 2.0)
- If DermNet logo: × logo_penalty (default: 3.0)
- Weights stack (watermarked logo image: 2.0 × 3.0 = 6.0×)

## Examples

See `examples.py` for complete working examples:

```bash
python examples.py
```

## Troubleshooting

### High False Positive Watermark Detection
- Increase `threshold` parameter in `has_watermark()`
- Example: `detector.has_watermark(image, threshold=0.25)`

### Missing Watermarks
- Decrease `threshold` parameter
- Example: `detector.has_watermark(image, threshold=0.10)`

### Low Classification Confidence
- Check for misclassified images in verification report
- Use Grad-CAM to see which regions are important
- May indicate training data issues or need for more training

### Memory Issues
- Reduce batch size
- Disable contrastive loss if using on limited GPU memory
- Use `use_contrastive=False`

## File Structure

```
hair_spotter/
├── hair_spotter_class.py          # Original classifier
├── contrastive_loss.py            # Contrastive loss + watermark detection
├── classification_verifier.py     # Verification & analysis
├── train_with_contrastive.py      # Enhanced training script
├── examples.py                     # Usage examples
└── README.md                      # This file
```

## References

- **Contrastive Learning**: Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)
- **Grad-CAM**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- **Saliency Maps**: Simonyan et al. "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps"

## License

Same as parent project

## Author Notes

- Watermark detection uses computer vision heuristics, not ML, for robustness
- Penalty system encourages model to ignore watermarked/logo images during training
- Verification tools help debug why model makes specific predictions
- Consider using both contrastive loss and verification for best results
