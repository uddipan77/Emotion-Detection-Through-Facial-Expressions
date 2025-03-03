import torch

class Config:
    # Dataset and file paths
    csv_file = 'data/fer2013.csv'  # Path to the FER2013 dataset CSV file
    save_path = 'model/fer_model.pth'  # Path to save the trained model
    
    # Training parameters
    batch_size = 32  # Batch size for DataLoader
    num_epochs = 50  # Number of epochs
    learning_rate = 1e-4  # Lowered learning rate for better convergence
    scheduler_step_size = 10  # Step size for learning rate scheduler
    scheduler_gamma = 0.1  # Reduce LR by 90% every step
    class_weights = [1.9804, 4.1837, 1.9470, 1.3844, 1.7759, 2.1936, 1.7561]  # Adjust weights for imbalanced classes
    
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Apple MPS or CPU fallback
