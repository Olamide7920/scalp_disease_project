from hair_spotter_class import HairClassifier
import torch
from sklearn.metrics import f1_score
import numpy as np

def evaluate(model, data_loader, device, set_name="Set"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"{set_name} Accuracy: {accuracy:.2f}%")
    print(f"{set_name} F1 Score: {f1:.4f}")
    return accuracy, f1

def main():
    data_dir = "src/self/hair_spotter/training_hair"
    hc = HairClassifier(data_dir=data_dir)
    train_loader, val_loader, test_loader = hc.get_dataloaders()

    # Train and save weights after each epoch, print validation metrics
    for epoch in range(1, 6):  # 5 epochs
        hc.train(train_loader, epochs=1)
        weights_path = f"src/self/hair_spotter/hair_classifier_weights_epoch{epoch}.pth"
        torch.save(hc.model.state_dict(), weights_path)
        print(f"Model weights saved to {weights_path}")

        print(f"Epoch {epoch}:")
        evaluate(hc.model, val_loader, hc.device, set_name="Validation")

    # After training, evaluate on test set
    print("Final Test Set Evaluation:")
    evaluate(hc.model, test_loader, hc.device, set_name="Test")

    # Embedding extraction (run after training)
    embeddings_by_class = {0: [], 1: []}
    feature_extractor = torch.nn.Sequential(*(list(hc.model.children())[:-1]))

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(hc.device), labels.to(hc.device)
            features = feature_extractor(inputs)
            features = features.view(features.size(0), -1)
            for emb, label in zip(features.cpu().numpy(), labels.cpu().numpy()):
                embeddings_by_class[label].append(emb)

    # Compute average embedding for each class
    avg_embeddings = {}
    for label in embeddings_by_class:
        avg_embeddings[label] = np.mean(embeddings_by_class[label], axis=0)
        class_name = "hair" if label == 0 else "not_hair"
        print(f"Average embedding for class '{class_name}':\n{avg_embeddings[label]}\n")

    # Save average embeddings for later use
    np.save("src/self/hair_spotter/avg_embeddings.npy", avg_embeddings)
    print("Average embeddings saved to src/self/hair_spotter/avg_embeddings.npy")

if __name__ == "__main__":
    main()