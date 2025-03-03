import torch
import torch.nn as nn

def train_model(model, train_loader, epochs, optimizer, criterion, device, log_path, scheduler=None):
    model.to(device)
    model.train()

    best_val_loss = float('inf')
    patience = 5  # Stop if no improvement for 5 epochs
    no_improve_epochs = 0

    for epoch in range(epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Step scheduler, if available
        if scheduler:
            scheduler.step(avg_loss)

        # Log epoch loss
        with open(log_path, "a") as log_file:
            log_file.write(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}\n")
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Early stopping logic
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), "outputs/best_model.pth")  # Save best model
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break
