import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
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
        from torchvision.models import ResNet18_Weights
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)        # Replace the final layer to output 2 classes: hair, not hair
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model = self.model.to(self.device)
        # Define image preprocessing steps
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),          # Convert images to PyTorch tensors
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
        ])

    def get_dataloaders(self):

        dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)

        cleaned_samples = []
        # Normalize paths and check if files exist
        for path, label in dataset.samples:
            norm_path = os.path.normpath(path)
            # Check if the file exists after normalization
            # This is important to ensure the path is correct across different OS
            if os.path.exists(norm_path):
                cleaned_samples.append((norm_path, label))
            else:
                print(f"[WARNING] Missing file skipped: {norm_path}")

        dataset.samples = cleaned_samples
        dataset.imgs = cleaned_samples  # required by torchvision

        # Split dataset: 70% train, 15% val, 15% test
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader

    def train(self, train_loader, epochs=1, lr=0.001):
        """
        Train the model using the provided DataLoader.
        """
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