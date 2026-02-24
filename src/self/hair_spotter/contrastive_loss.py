"""
Contrastive learning loss with penalties for watermarks and DermNet logos.

This module implements:
- NTXentLoss (contrastive loss)
- Watermark/Logo detection and penalty system
- Loss weighting based on image quality indicators
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


class WatermarkDetector:
    """Detect watermarks and DermNet logos in images."""
    
    @staticmethod
    def has_watermark(image_tensor, threshold=0.15):
        """
        Detect watermarks using edge detection and text region analysis.
        
        Args:
            image_tensor: Tensor of shape (3, H, W) or (B, 3, H, W), range [0, 1]
            threshold: Sensitivity threshold for watermark detection
            
        Returns:
            bool or tensor of bools indicating watermark presence
        """
        # Handle batch dimension
        if image_tensor.dim() == 4:
            return torch.stack([
                WatermarkDetector.has_watermark(img, threshold) 
                for img in image_tensor
            ])
        
        # Convert to numpy for OpenCV
        if isinstance(image_tensor, torch.Tensor):
            img_np = image_tensor.cpu().numpy()
            # Denormalize: [0, 1] to [0, 255]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            # Convert CHW to HWC
            img_np = np.transpose(img_np, (1, 2, 0))
        else:
            img_np = image_tensor
        
        # Convert to grayscale
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Text region detection using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for characteristic watermark regions (corners, edges)
        watermark_regions = 0
        h, w = gray.shape
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Watermarks typically small to medium sized features
            if 100 < area < (h * w * 0.1):
                x, y, cw, ch = cv2.boundingRect(contour)
                # Check if in corners or edges (typical watermark location)
                is_corner = (x < w * 0.2 or x > w * 0.8) or (y < h * 0.2 or y > h * 0.8)
                is_edge = (y < h * 0.15 or y > h * 0.85)
                if is_corner or is_edge:
                    watermark_regions += 1
        
        # Determine watermark presence
        has_watermark = (edge_ratio > threshold) or (watermark_regions > 3)
        return torch.tensor(has_watermark, dtype=torch.bool)
    
    @staticmethod
    def has_dermnet_logo(image_tensor):
        """
        Detect DermNet logo (typically text-based, top-right or bottom area).
        
        Args:
            image_tensor: Tensor of shape (3, H, W) or (B, 3, H, W), range [0, 1]
            
        Returns:
            bool or tensor of bools indicating DermNet logo presence
        """
        if image_tensor.dim() == 4:
            return torch.stack([
                WatermarkDetector.has_dermnet_logo(img) 
                for img in image_tensor
            ])
        
        # Convert to numpy
        if isinstance(image_tensor, torch.Tensor):
            img_np = image_tensor.cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            img_np = np.transpose(img_np, (1, 2, 0))
        else:
            img_np = image_tensor
        
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        h, w = gray.shape
        
        # DermNet logos appear mostly in top-right or bottom areas
        # Check top-right quadrant (typical logo location)
        top_right = gray[:h//4, w//2:]
        bottom = gray[3*h//4:, :]
        
        # Detect text using edge and threshold
        _, binary_tr = cv2.threshold(top_right, 200, 255, cv2.THRESH_BINARY)
        _, binary_bt = cv2.threshold(bottom, 200, 255, cv2.THRESH_BINARY)
        
        text_ratio_tr = np.sum(binary_tr > 0) / binary_tr.size if binary_tr.size > 0 else 0
        text_ratio_bt = np.sum(binary_bt > 0) / binary_bt.size if binary_bt.size > 0 else 0
        
        # Heuristic: DermNet logo has moderate text coverage in specific regions
        has_logo = (text_ratio_tr > 0.05 and text_ratio_tr < 0.3) or \
                   (text_ratio_bt > 0.02 and text_ratio_bt < 0.15)
        
        return torch.tensor(has_logo, dtype=torch.bool)


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for contrastive learning.
    """
    
    def __init__(self, temperature=0.07, batch_size=256):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
    
    def forward(self, z_i, z_j, weights=None):
        """
        Args:
            z_i: Anchor embeddings (B, D)
            z_j: Positive embeddings (B, D)
            weights: Optional per-sample weights for penalty (B,)
            
        Returns:
            Scalar loss value
        """
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate anchors and positives
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Create mask for positive pairs (diagonal of each half)
        batch_size = z_i.shape[0]
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        mask_fill_diagonal = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z_i.device)
        mask_fill_diagonal[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = True
        mask_fill_diagonal[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = True
        
        # Compute loss
        pos_mask = mask_fill_diagonal
        neg_mask = ~mask
        
        pos_sim = similarity_matrix[pos_mask].view(2 * batch_size, 1)
        neg_sim = similarity_matrix[neg_mask].view(2 * batch_size, -1)
        
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        
        loss = F.cross_entropy(logits / self.temperature, labels, reduction='none')
        
        # Apply per-sample weights if provided
        if weights is not None:
            # Duplicate weights for z_i and z_j
            weights_expanded = torch.cat([weights, weights], dim=0)
            loss = loss * weights_expanded
        
        return loss.mean()


class ContrastiveHairLoss(nn.Module):
    """
    Combined loss for hair classifier with watermark/logo penalties.
    """
    
    def __init__(self, 
                 temperature=0.07,
                 watermark_penalty=2.0,
                 logo_penalty=3.0,
                 use_contrastive=True):
        """
        Args:
            temperature: Temperature for contrastive loss
            watermark_penalty: Penalty multiplier for watermarked images
            logo_penalty: Penalty multiplier for DermNet logo images
            use_contrastive: Whether to use contrastive loss (True) or CE loss (False)
        """
        super().__init__()
        self.temperature = temperature
        self.watermark_penalty = watermark_penalty
        self.logo_penalty = logo_penalty
        self.use_contrastive = use_contrastive
        
        self.ntxent_loss = NTXentLoss(temperature=temperature)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.watermark_detector = WatermarkDetector()
    
    def forward(self, logits, labels, images=None, embeddings=None):
        """
        Compute combined loss with quality penalties.
        
        Args:
            logits: Classification logits (B, num_classes)
            labels: True labels (B,)
            images: Optional raw images for watermark detection (B, 3, H, W) in [0, 1]
            embeddings: Optional embeddings for contrastive loss (B, D)
            
        Returns:
            Scalar loss value
        """
        # Compute base classification loss
        ce_loss = self.ce_loss(logits, labels)
        
        # Detect watermarks and compute penalties
        quality_weights = torch.ones(labels.shape[0], device=labels.device)
        
        if images is not None:
            # Detect watermarks
            has_watermarks = self.watermark_detector.has_watermark(images)
            if isinstance(has_watermarks, torch.Tensor):
                has_watermarks = has_watermarks.to(labels.device)
                quality_weights[has_watermarks] *= self.watermark_penalty
            
            # Detect DermNet logos
            has_logos = self.watermark_detector.has_dermnet_logo(images)
            if isinstance(has_logos, torch.Tensor):
                has_logos = has_logos.to(labels.device)
                quality_weights[has_logos] *= self.logo_penalty
        
        # Apply quality weights to CE loss
        weighted_ce_loss = (ce_loss * quality_weights).mean()
        
        # Add contrastive loss if embeddings provided
        if self.use_contrastive and embeddings is not None:
            # For contrastive, we need pairs. Simple approach: use embeddings for positive pair
            # In practice, you'd use augmented pairs or hard negatives
            contrastive_loss = self.ntxent_loss(embeddings, embeddings, weights=quality_weights)
            total_loss = weighted_ce_loss + 0.5 * contrastive_loss
        else:
            total_loss = weighted_ce_loss
        
        return total_loss


def create_penalty_info(images):
    """
    Utility to create penalty information for debugging/analysis.
    
    Args:
        images: Tensor of shape (B, 3, H, W)
        
    Returns:
        Dict with penalty information per image
    """
    detector = WatermarkDetector()
    info = {
        'watermarks': detector.has_watermark(images),
        'logos': detector.has_dermnet_logo(images),
    }
    return info
