import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

# Define a regular function to replace the lambda
def to_rgb(image):
    """Converts a single-channel grayscale image to 3-channel RGB."""
    return image.repeat(3, 1, 1)

class EmotionDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        """
        Custom PyTorch Dataset for Emotion Detection.
        Args:
            data_frame (pd.DataFrame): The FER2013 data as a pandas DataFrame.
            transform: Transformations to apply to the image.
        """
        self.data_frame = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Returns an image and its label at the given index.
        Handles exceptions during processing.
        """
        row = self.data_frame.iloc[idx]
        try:
            image = np.array(row['pixels'].split(), dtype=np.float32).reshape(48, 48)
            label = int(row['emotion'])
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            return None, None

def load_data(csv_file, batch_size, shuffle=True, augment=True):
    """
    Load and split the FER2013 dataset into training, validation, and test sets.
    Args:
        csv_file (str): Path to the FER2013 CSV file.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the training dataset.
        augment (bool): Whether to apply augmentation for training data.
    Returns:
        Tuple of DataLoaders for training, validation, and testing datasets.
    """
    df = pd.read_csv(csv_file)
    train_df = df[df['Usage'] == 'Training']
    val_df = df[df['Usage'] == 'PublicTest']
    test_df = df[df['Usage'] == 'PrivateTest']

    # Define data transformations
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Lambda(to_rgb),  # Convert grayscale to RGB
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    common_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Lambda(to_rgb),  # Convert grayscale to RGB
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create datasets
    train_dataset = EmotionDataset(train_df, transform=train_transforms)
    val_dataset = EmotionDataset(val_df, transform=common_transforms)
    test_dataset = EmotionDataset(test_df, transform=common_transforms)

    # Create DataLoaders
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    )
