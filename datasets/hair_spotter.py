import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

class HairClassifier:
    def __init__(self, data_dir, batch_size=16, num_workers=2):
        """
        Initialize the classifier, set up model, device, and image transforms.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load a pre-trained ResNet18 model
        self.model = models.resnet18(pretrained=True)
        # Replace the final layer to output 2 classes: hair, not hair
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model = self.model.to(self.device)
        # Define image preprocessing steps
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),          # Convert images to PyTorch tensors
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
        ])

    def get_dataloaders(self):
        """
        Create PyTorch DataLoaders for training and validation.
        Assumes data_dir contains 'hair' and 'not_hair' subfolders.
        """
        dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        # Split dataset: 80% train, 20% validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader

    def train(self, epochs=5, lr=1e-3):
        """
        Train the classifier on the dataset.
        """
        train_loader, val_loader = self.get_dataloaders()
        criterion = nn.CrossEntropyLoss()  # Loss function for classification
        optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Adam optimizer
        for epoch in range(epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()           # Clear gradients
                outputs = self.model(inputs)    # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()                 # Backpropagation
                optimizer.step()                # Update weights
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    def predict(self, image_path):
        """
        Predict if a single image is 'hair' or 'not_hair'.
        """
        from PIL import Image
        self.model.eval()  # Set model to evaluation mode
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)  # Preprocess image
        with torch.no_grad():
            output = self.model(img_t)  # Forward pass
            _, pred = torch.max(output, 1)  # Get predicted class
        return "hair" if pred.item() == 0 else "not_hair"

    def save_weights(self, path):
        """
        Save the model weights to the specified file.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load_weights(self, path):
        """
        Load the model weights from the specified file.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model weights loaded from {path}")

# Example usage:
# hc = HairClassifier(data_dir="datasets/ddi")
# hc.train(epochs=5)