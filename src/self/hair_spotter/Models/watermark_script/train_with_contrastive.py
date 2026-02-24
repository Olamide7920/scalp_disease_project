"""
Enhanced training script for HairClassifier with contrastive loss and watermark penalties.

Usage:
    python train_with_contrastive.py --data_dir <path> --epochs 10 --use_contrastive
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

from hair_spotter_class import HairClassifier
from contrastive_loss import ContrastiveHairLoss, WatermarkDetector
from classification_verifier import ClassificationVerifier


class EnhancedHairDataset(Dataset):
    """Dataset wrapper that loads raw images for watermark detection."""
    
    def __init__(self, image_folder_dataset):
        """
        Args:
            image_folder_dataset: torchvision ImageFolder dataset
        """
        self.dataset = image_folder_dataset
        self.transform_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # Don't normalize for watermark detection
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get preprocessed image and label from original dataset
        img_preprocessed, label = self.dataset[idx]
        
        # Also get raw image tensor for watermark detection
        raw_img_path = self.dataset.imgs[idx][0]
        from PIL import Image
        raw_img = Image.open(raw_img_path).convert('RGB')
        raw_img_tensor = self.transform_to_tensor(raw_img)
        
        return img_preprocessed, label, raw_img_tensor


class EnhancedHairClassifier(HairClassifier):
    """Extended HairClassifier with contrastive learning support."""
    
    def __init__(self, data_dir, batch_size=16, num_workers=2):
        super().__init__(data_dir, batch_size, num_workers)
        
        # Add feature extraction capability
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.eval()
    
    def get_features(self, image_tensor):
        """Extract features from image without classification."""
        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
            features = features.view(features.size(0), -1)
        return features
    
    def get_dataloaders_with_raw(self):
        """Get dataloaders that also include raw images for watermark detection."""
        dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        
        # Clean samples
        cleaned_samples = []
        for path, label in dataset.samples:
            norm_path = os.path.normpath(path)
            if os.path.exists(norm_path):
                cleaned_samples.append((norm_path, label))
            else:
                print(f"[WARNING] Missing file skipped: {norm_path}")
        
        dataset.samples = cleaned_samples
        dataset.imgs = cleaned_samples
        
        # Wrap with raw image dataset
        dataset = EnhancedHairDataset(dataset)
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                  shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                 shuffle=False, num_workers=self.num_workers)
        
        return train_loader, val_loader, test_loader
    
    def train_with_contrastive(self, train_loader, val_loader, epochs=5, lr=0.001,
                               use_contrastive=True, watermark_penalty=2.0,
                               logo_penalty=3.0):
        """
        Train with contrastive loss and watermark penalties.
        
        Args:
            train_loader: Training data loader (returns img, label, raw_img)
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            use_contrastive: Whether to use contrastive loss
            watermark_penalty: Penalty for watermarked images
            logo_penalty: Penalty for DermNet logo images
        """
        # Create loss function
        loss_fn = ContrastiveHairLoss(
            temperature=0.07,
            watermark_penalty=watermark_penalty,
            logo_penalty=logo_penalty,
            use_contrastive=use_contrastive
        )
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        train_losses = []
        val_losses = []
        val_accs = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            train_preds = []
            train_labels = []
            
            for batch_idx, batch in enumerate(train_loader):
                if len(batch) == 3:
                    inputs, labels, raw_inputs = batch
                else:
                    inputs, labels = batch
                    raw_inputs = None
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if raw_inputs is not None:
                    raw_inputs = raw_inputs.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(inputs)
                
                # Get features for contrastive loss
                embeddings = None
                if use_contrastive:
                    embeddings = self.get_features(inputs)
                
                # Compute loss
                if raw_inputs is not None:
                    loss = loss_fn(logits, labels, images=raw_inputs, embeddings=embeddings)
                else:
                    loss = loss_fn(logits, labels, embeddings=embeddings)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Predictions for metrics
                _, preds = torch.max(logits, 1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
            
            # Compute training metrics
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='weighted')
            avg_train_loss = running_loss / len(train_loader)
            
            # Validation
            val_loss, val_acc, val_f1 = self.evaluate_with_loss(val_loader, loss_fn)
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            scheduler.step()
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, val_accs, epochs)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accs': val_accs
        }
    
    def evaluate_with_loss(self, dataloader, loss_fn):
        """Evaluate model with loss computation."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    inputs, labels, raw_inputs = batch
                else:
                    inputs, labels = batch
                    raw_inputs = None
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if raw_inputs is not None:
                    raw_inputs = raw_inputs.to(self.device)
                
                logits = self.model(inputs)
                
                if raw_inputs is not None:
                    loss = loss_fn(logits, labels, images=raw_inputs)
                else:
                    loss = loss_fn(logits, labels)
                
                running_loss += loss.item()
                
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = running_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, acc, f1
    
    def plot_training_curves(self, train_losses, val_losses, val_accs, epochs):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(range(1, epochs+1), train_losses, 'b-o', label='Train Loss')
        ax1.plot(range(1, epochs+1), val_losses, 'r-o', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(range(1, epochs+1), val_accs, 'g-o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.grid(True)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('training_curves_contrastive.png', dpi=100)
        print("Training curves saved to training_curves_contrastive.png")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train hair classifier with contrastive loss')
    parser.add_argument('--data_dir', type=str, default='src/self/hair_spotter/training_hair',
                        help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--use_contrastive', action='store_true', 
                        help='Use contrastive loss')
    parser.add_argument('--watermark_penalty', type=float, default=2.0,
                        help='Penalty multiplier for watermarked images')
    parser.add_argument('--logo_penalty', type=float, default=3.0,
                        help='Penalty multiplier for DermNet logo images')
    parser.add_argument('--verify', action='store_true',
                        help='Run verification after training')
    parser.add_argument('--verify_dir', type=str, default='src/self/hair_spotter/training_hair',
                        help='Directory to verify predictions on')
    parser.add_argument('--output_weights', type=str, default='hair_classifier_contrastive.pth',
                        help='Path to save trained weights')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HAIR CLASSIFIER - CONTRASTIVE LEARNING WITH WATERMARK PENALTIES")
    print("=" * 80)
    
    # Initialize classifier
    print(f"\nInitializing classifier with data from: {args.data_dir}")
    classifier = EnhancedHairClassifier(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # Get dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = classifier.get_dataloaders_with_raw()
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Contrastive loss: {args.use_contrastive}")
    print(f"  Watermark penalty: {args.watermark_penalty}")
    print(f"  Logo penalty: {args.logo_penalty}")
    
    history = classifier.train_with_contrastive(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        use_contrastive=args.use_contrastive,
        watermark_penalty=args.watermark_penalty,
        logo_penalty=args.logo_penalty
    )
    
    # Test
    print("\nEvaluating on test set...")
    loss_fn = ContrastiveHairLoss(
        watermark_penalty=args.watermark_penalty,
        logo_penalty=args.logo_penalty,
        use_contrastive=args.use_contrastive
    )
    test_loss, test_acc, test_f1 = classifier.evaluate_with_loss(test_loader, loss_fn)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    
    # Save weights
    classifier.save_weights(args.output_weights)
    
    # Verify predictions if requested
    if args.verify:
        print("\n" + "=" * 80)
        print("CLASSIFICATION VERIFICATION")
        print("=" * 80)
        
        verifier = ClassificationVerifier(classifier.model, device=classifier.device)
        
        # Analyze hair directory
        hair_dir = os.path.join(args.verify_dir, 'hair')
        not_hair_dir = os.path.join(args.verify_dir, 'not_hair')
        
        dirs_to_check = {}
        if os.path.isdir(hair_dir):
            dirs_to_check['hair'] = hair_dir
        if os.path.isdir(not_hair_dir):
            dirs_to_check['not_hair'] = not_hair_dir
        
        if dirs_to_check:
            report = verifier.generate_report(dirs_to_check)
            print(report)
            
            # Save report
            with open('verification_report.txt', 'w') as f:
                f.write(report)
            print("\nReport saved to verification_report.txt")


if __name__ == '__main__':
    main()
