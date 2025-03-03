import torch

def validate_model(model, val_loader, device, criterion):
    """
    Validates the model on the validation dataset.

    Args:
        model: The PyTorch model to validate.
        val_loader: DataLoader for the validation dataset.
        device: Device to perform validation on.
        criterion: Loss function for validation.

    Returns:
        A tuple containing validation loss and accuracy.
    """
    model.to(device)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total

    return avg_val_loss, val_accuracy
