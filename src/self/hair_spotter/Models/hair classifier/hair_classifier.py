import os
import random
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import f1_score

class HairClassifier:
    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        pretrained: bool = True,
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # build model (ResNet18)
        self.model = models.resnet18(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # binary: straight vs coily
        self.model = self.model.to(self.device)

    def get_dataloaders(self, split: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Expects self.data_dir to be an ImageFolder-style directory with subfolders per class.
        Returns train_loader, val_loader, test_loader.
        """
        assert sum(split) == 1.0, "split must sum to 1.0"
        dataset = datasets.ImageFolder(self.data_dir, transform=self.train_transform)

        total = len(dataset)
        train_len = int(split[0] * total)
        val_len = int(split[1] * total)
        test_len = total - train_len - val_len

        train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len],
                                                    generator=torch.Generator().manual_seed(self.seed))

        # override transforms for validation/test
        val_set.dataset.transform = self.eval_transform
        test_set.dataset.transform = self.eval_transform

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # save class mapping
        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        return train_loader, val_loader, test_loader

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader = None,
              epochs: int = 5,
              lr: float = 1e-3,
              save_dir: str = None):
        """
        Train for `epochs`. If val_loader provided, evaluate after each epoch.
        If save_dir provided, saves weights per epoch as hair_classifier_epoch{n}.pth.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        history = {"train_loss": [], "val_acc": [], "val_f1": []}

        for e in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / max(1, len(train_loader))
            print(f"Epoch {e}/{epochs}, Loss: {avg_loss:.4f}")
            history["train_loss"].append(avg_loss)

            if val_loader is not None:
                acc, f1 = self.evaluate(val_loader, set_name=f"Validation (epoch {e})")
                history["val_acc"].append(acc)
                history["val_f1"].append(f1)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f"hair_classifier_epoch{e}.pth")
                torch.save(self.model.state_dict(), path)
                print(f"Saved weights to {path}")

            scheduler.step()

        return history

    def evaluate(self, data_loader: DataLoader, set_name: str = "Set") -> Tuple[float, float]:
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if len(all_labels) == 0:
            print(f"{set_name}: no samples")
            return 0.0, 0.0

        accuracy = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average="weighted")
        print(f"{set_name} Accuracy: {accuracy:.2f}%  F1: {f1:.4f}")
        return accuracy, f1

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Weights saved to {path}")

    def load_weights(self, path: str, map_location: Any = None):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        map_loc = map_location if map_location is not None else self.device
        self.model.load_state_dict(torch.load(path, map_location=map_loc))
        self.model.to(self.device)
        print(f"Weights loaded from {path}")

    def predict_image(self, pil_image) -> Dict[str, Any]:
        """
        Accepts a PIL image. Returns predicted label, probability scores.
        """
        self.model.eval()
        img = self.eval_transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(img)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx = int(torch.argmax(out, dim=1).cpu().numpy()[0])
            label = self.idx_to_class.get(pred_idx, str(pred_idx))
        return {"label": label, "pred_idx": pred_idx, "probs": probs}

if __name__ == "__main__":
    # Example quick run (adjust paths and params)
    data_dir = "datasets/hair_types"  # expect subfolders e.g. datasets/hair_types/straight and .../coily
    hc = HairClassifier(data_dir=data_dir, batch_size=16, num_workers=2, pretrained=True)
    train_loader, val_loader, test_loader = hc.get_dataloaders()

    # Train with validation and save per-epoch weights
    hc.train(train_loader, val_loader=val_loader, epochs=5, lr=1e-4, save_dir="models")

    # Final evaluation on test set
    hc.evaluate(test_loader, set_name="Test")