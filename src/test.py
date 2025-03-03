import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

def test_model(model, test_loader, device, save_results=False):
    """
    Tests the model on the test dataset and generates evaluation metrics.

    Args:
        model: The PyTorch model to test.
        test_loader: DataLoader for the test dataset.
        device: Device to perform testing on.
        save_results: Boolean flag to save classification report and confusion matrix.

    Returns:
        Classification report and confusion matrix.
    """
    # Ensure the outputs folder exists
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    misclassified = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Identify misclassified samples
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append((inputs[i].cpu(), labels[i].cpu().item(), preds[i].cpu().item()))

    # Classification Report
    report = classification_report(all_labels, all_preds)
    print("Classification Report:")
    print(report)
    
    # Save classification report
    if save_results:
        with open("outputs/classification_report.txt", "w") as report_file:
            report_file.write("Classification Report:\n")
            report_file.write(report)

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix plot
    if save_results:
        plt.savefig("outputs/confusion_matrix.png")
    plt.show()

    # Save or Print Misclassified Samples
    # Save or Print Misclassified Samples
    if save_results and len(misclassified) > 0:
        os.makedirs("outputs/misclassified", exist_ok=True)
        print(f"Total Misclassified Samples: {len(misclassified)}")
    
        for idx, (image, true_label, predicted_label) in enumerate(misclassified[:10]):  # Save first 10 misclassified
            plt.figure()

            # Convert 3D tensor (RGB) to 2D (grayscale)
            if image.shape[0] == 3:  # If the image has 3 channels (RGB)
                image = torch.mean(image, dim=0)  # Convert to grayscale by averaging the RGB channels

            plt.imshow(image.squeeze(), cmap="gray")  # Show the 2D grayscale image
            plt.title(f"True: {true_label}, Pred: {predicted_label}")
            plt.axis('off')
            plt.savefig(f"outputs/misclassified/misclassified_{idx}.png")  # Save the misclassified image
            plt.close()
            print(f"Misclassified Image {idx}: True Label = {true_label}, Predicted Label = {predicted_label}")
    
    # Analyze frequent misclassifications
    misclass_stats = Counter([(true, pred) for _, true, pred in misclassified])
    print("Most Frequent Misclassifications:")
    for pair, count in misclass_stats.most_common(5):
        print(f"True Label = {pair[0]}, Predicted Label = {pair[1]}, Count = {count}")

    return report, conf_matrix
