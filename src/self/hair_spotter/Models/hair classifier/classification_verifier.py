"""
Classification verification and analysis tools.

This module helps understand why the model classifies images a certain way
through various interpretability and debugging techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from PIL import Image
import os


class ClassificationVerifier:
    """Verify and analyze classification decisions."""
    
    def __init__(self, model, device='cpu'):
        """
        Args:
            model: Trained classifier model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def get_confidence(self, image_path):
        """Get classification confidence for an image."""
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim=1)
        
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
        
        return {
            'image': image_path,
            'predicted_class': 'hair' if pred_class == 0 else 'not_hair',
            'confidence': confidence,
            'probs': probs[0].cpu().numpy(),
            'logits': logits[0].cpu().numpy()
        }
    
    def grad_cam(self, image_path, target_layer=None):
        """
        Compute Grad-CAM visualization to see which regions influenced prediction.
        
        Args:
            image_path: Path to image
            target_layer: Layer to compute gradients for (default: last conv layer)
            
        Returns:
            Dict with visualization data
        """
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True
        
        # Get the target layer (assume ResNet18: layer4[-1] for last conv block)
        if target_layer is None:
            target_layer = self.model.layer4[-1]
        
        # Forward pass
        activations = None
        def hook_fn(module, input, output):
            nonlocal activations
            activations = output.detach()
        
        hook = target_layer.register_forward_hook(hook_fn)
        logits = self.model(img_tensor)
        hook.remove()
        
        # Get predicted class
        pred_class = torch.argmax(logits, dim=1).item()
        
        # Backward pass
        logits[0, pred_class].backward()
        gradients = img_tensor.grad
        
        # Compute Grad-CAM
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze(0).squeeze(0).cpu().detach().numpy()
        
        # Normalize
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return {
            'cam': cam,
            'predicted_class': pred_class,
            'original_image': img_tensor.detach().cpu()
        }
    
    def saliency_map(self, image_path):
        """Compute saliency map showing pixel importance."""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True
        
        # Forward and backward
        logits = self.model(img_tensor)
        pred_class = torch.argmax(logits, dim=1).item()
        logits[0, pred_class].backward()
        
        # Get saliency
        saliency = img_tensor.grad.abs().max(dim=1)[0].squeeze(0).cpu().detach().numpy()
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return {
            'saliency': saliency,
            'predicted_class': pred_class,
        }
    
    def analyze_batch(self, image_dir, class_name):
        """
        Analyze all images in a directory and report issues.
        
        Args:
            image_dir: Directory containing images
            class_name: Class name (hair, not_hair) for context
            
        Returns:
            Analysis report dict
        """
        results = {
            'total_images': 0,
            'correct': 0,
            'incorrect': 0,
            'low_confidence': [],  # Images with <70% confidence
            'misclassified': [],   # Wrongly classified
            'confidence_stats': {},
        }
        
        confidences = []
        
        for fname in os.listdir(image_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            try:
                image_path = os.path.join(image_dir, fname)
                info = self.get_confidence(image_path)
                results['total_images'] += 1
                
                # True label from directory structure
                true_class = 'hair' if class_name.lower() == 'hair' else 'not_hair'
                
                if info['predicted_class'] == true_class:
                    results['correct'] += 1
                else:
                    results['incorrect'] += 1
                    results['misclassified'].append({
                        'image': fname,
                        'predicted': info['predicted_class'],
                        'confidence': info['confidence']
                    })
                
                if info['confidence'] < 0.70:
                    results['low_confidence'].append({
                        'image': fname,
                        'predicted': info['predicted_class'],
                        'confidence': info['confidence']
                    })
                
                confidences.append(info['confidence'])
            
            except Exception as e:
                print(f"Error processing {fname}: {e}")
        
        if confidences:
            results['confidence_stats'] = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
            }
        
        return results
    
    def compare_predictions(self, image_dir_1, image_dir_2, class_1, class_2):
        """
        Compare predictions on two different image sets.
        Useful for understanding differences between datasets.
        """
        report_1 = self.analyze_batch(image_dir_1, class_1)
        report_2 = self.analyze_batch(image_dir_2, class_2)
        
        return {
            'set_1': {'path': image_dir_1, 'report': report_1},
            'set_2': {'path': image_dir_2, 'report': report_2},
        }
    
    def visualize_predictions(self, image_path, output_path=None):
        """
        Create visualization showing prediction and attention.
        
        Args:
            image_path: Path to image
            output_path: Where to save visualization (optional)
        """
        # Get prediction info
        conf_info = self.get_confidence(image_path)
        
        # Get Grad-CAM
        try:
            cam_info = self.grad_cam(image_path)
            has_cam = True
        except:
            has_cam = False
        
        # Get saliency
        try:
            sal_info = self.saliency_map(image_path)
            has_sal = True
        except:
            has_sal = False
        
        # Load original image
        img_pil = Image.open(image_path).convert('RGB')
        img_np = np.array(img_pil)
        if img_np.shape != (224, 224, 3):
            img_pil = img_pil.resize((224, 224))
            img_np = np.array(img_pil)
        
        # Create figure
        n_cols = 1 + (1 if has_cam else 0) + (1 if has_sal else 0)
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        if n_cols == 1:
            axes = [axes]
        
        # Plot original
        axes[0].imshow(img_np)
        axes[0].set_title(f"Prediction: {conf_info['predicted_class']}\nConfidence: {conf_info['confidence']:.2%}")
        axes[0].axis('off')
        
        # Plot Grad-CAM
        if has_cam:
            axes[1].imshow(img_np)
            im = axes[1].imshow(cam_info['cam'], cmap='hot', alpha=0.5)
            axes[1].set_title("Grad-CAM Attention")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
        
        # Plot saliency
        if has_sal:
            axes[2 if has_cam else 1].imshow(sal_info['saliency'], cmap='viridis')
            axes[2 if has_cam else 1].set_title("Saliency Map")
            axes[2 if has_cam else 1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        return fig
    
    def generate_report(self, image_dirs_dict):
        """
        Generate comprehensive analysis report for multiple directories.
        
        Args:
            image_dirs_dict: Dict of {class_name: directory_path}
            
        Returns:
            Formatted report
        """
        report = "=" * 80 + "\n"
        report += "CLASSIFICATION VERIFICATION REPORT\n"
        report += "=" * 80 + "\n\n"
        
        for class_name, image_dir in image_dirs_dict.items():
            if not os.path.isdir(image_dir):
                report += f"Directory not found: {image_dir}\n"
                continue
            
            analysis = self.analyze_batch(image_dir, class_name)
            
            report += f"\nClass: {class_name.upper()}\n"
            report += f"Directory: {image_dir}\n"
            report += f"Total images: {analysis['total_images']}\n"
            report += f"Correct: {analysis['correct']} "
            report += f"({100*analysis['correct']/max(analysis['total_images'], 1):.1f}%)\n"
            report += f"Incorrect: {analysis['incorrect']} "
            report += f"({100*analysis['incorrect']/max(analysis['total_images'], 1):.1f}%)\n"
            
            if analysis['confidence_stats']:
                stats = analysis['confidence_stats']
                report += f"\nConfidence Stats:\n"
                report += f"  Mean: {stats['mean']:.4f}\n"
                report += f"  Std:  {stats['std']:.4f}\n"
                report += f"  Min:  {stats['min']:.4f}\n"
                report += f"  Max:  {stats['max']:.4f}\n"
            
            if analysis['low_confidence']:
                report += f"\nLow Confidence Images (<70%):\n"
                for item in analysis['low_confidence'][:5]:
                    report += f"  - {item['image']}: {item['confidence']:.2%}\n"
                if len(analysis['low_confidence']) > 5:
                    report += f"  ... and {len(analysis['low_confidence']) - 5} more\n"
            
            if analysis['misclassified']:
                report += f"\nMisclassified Images:\n"
                for item in analysis['misclassified'][:5]:
                    report += f"  - {item['image']}: predicted {item['predicted']} "
                    report += f"({item['confidence']:.2%})\n"
                if len(analysis['misclassified']) > 5:
                    report += f"  ... and {len(analysis['misclassified']) - 5} more\n"
        
        report += "\n" + "=" * 80 + "\n"
        return report
