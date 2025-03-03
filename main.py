from src.config import Config
from src.model import EmotionNet
from src.data_loader import load_data
from src.train import train_model
from src.validation import validate_model
from src.test import test_model
import torch
import os

def main():
    # Ensure the outputs folder exists
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.dirname(Config.save_path), exist_ok=True)


    # Load datasets (train, validation, and test)
    train_loader, val_loader, test_loader = load_data(Config.csv_file, Config.batch_size, shuffle=True)

    print(f"Training data size: {len(train_loader.dataset)}")
    print(f"Validation data size: {len(val_loader.dataset)}")
    print(f"Testing data size: {len(test_loader.dataset)}")

    # Initialize the model
    model = EmotionNet().to(Config.device)

    # Define class weights and criterion
    class_weights = Config.class_weights  # Add class weights from Config
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(Config.device))

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=1e-4)  # Add L2 regularization (weight decay)

    # Define learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


    # Train the model and log outputs
    train_log_path = "outputs/training_logs.txt"
    with open(train_log_path, "w") as train_log:
        train_log.write("Training Logs:\n")
    train_model(
        model=model,
        train_loader=train_loader,
        epochs=Config.num_epochs,
        optimizer=optimizer,
        criterion=criterion,
        device=Config.device,
        log_path=train_log_path,
        scheduler=scheduler
    )

    # Validate the model and log validation results
    val_loss, val_accuracy = validate_model(model, val_loader, Config.device, criterion)
    with open("outputs/validation_results.txt", "w") as val_log:
        val_log.write(f"Validation Loss: {val_loss:.4f}\n")
        val_log.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Test the model and save outputs
    report, conf_matrix = test_model(model, test_loader, Config.device, save_results=True)
    print("Testing complete. Results saved to outputs.")

    # Save the trained model in the 'model' folder
    torch.save(model.state_dict(), Config.save_path)
    print(f"Model saved to {Config.save_path}")

if __name__ == "__main__":
    main()
