from class import HairClassifier
import torch
from sklearn.metrics import f1_score

# Set your dataset directory
data_dir = "datasets/training_hair"

hc = HairClassifier(data_dir=data_dir)
hc.train(epochs=5)

# Evaluate accuracy and F1 score on the validation set
_, val_loader = hc.get_dataloaders()
hc.model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(hc.device), labels.to(hc.device)
        outputs = hc.model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Validation Accuracy: {accuracy:.2f}%")
print(f"Validation F1 Score: {f1:.4f}")