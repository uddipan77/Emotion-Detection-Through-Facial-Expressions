# Emotion Detection Through Facial Expression 

This project implements a deep learning model to detect facial emotions using the FER2013 dataset. The model is built using PyTorch and leverages **EfficientNet-B3** for feature extraction, incorporating advanced techniques like data augmentation, class weighting, and dropout regularization to improve performance. The project also includes real-time emotion detection via a webcam.

---

## **Features**
- Emotion detection from grayscale images of faces.
- **EfficientNet-B3** as the backbone architecture for efficient feature extraction.
- Supports real-time emotion detection via webcam.
- Preprocessing includes normalization and data augmentation for better generalization.
- Implements training, validation, and testing pipelines.
- Logs and saves results, including confusion matrices and misclassified samples.

---

## **Dataset**
### FER2013 Dataset
The **FER2013** dataset is used for training and evaluating the model. It consists of 35,887 grayscale images of size 48x48, labeled with one of seven emotion categories.

- **Emotion Categories**:
  | Label | Emotion   |
  |-------|-----------|
  | 0     | Angry     |
  | 1     | Disgust   |
  | 2     | Fear      |
  | 3     | Happy     |
  | 4     | Sad       |
  | 5     | Surprise  |
  | 6     | Neutral   |

- **Dataset Splits**:
  | Usage       | Number of Samples | Percentage |
  |-------------|-------------------|------------|
  | Training    | 28,709            | ~80%       |
  | PublicTest  | 3,589             | ~10%       |
  | PrivateTest | 3,589             | ~10%       |

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition).

For more details about the dataset, see [data_details.md](data/data_details.md).

---

## **Folder Structure**

```
emotion_detection_project/
├── src/                    # Source code
│   ├── config.py           # Configuration settings
│   ├── data_loader.py      # Data loading and augmentation
│   ├── model.py            # EmotionNet model definition (EfficientNet-B3)
│   ├── train.py            # Training pipeline
│   ├── validation.py       # Validation pipeline
│   ├── test.py             # Evaluation pipeline with misclassification
│   ├── class_weights_calculation.py # Script to calculate class weights dynamically
├── data/                   # Dataset folder
│   ├── fer2013.csv         # FER2013 dataset (excluded via .gitignore)
│   └── data_details.md     # Dataset details
├── model/                  # Saved models (excluded via .gitignore)
│   └── fer_model.pth       # Final trained model weights
├── outputs/                # Outputs and logs (excluded via .gitignore)
│   ├── training_logs.txt       # Training logs
│   ├── validation_results.txt  # Validation loss and accuracy results
│   ├── classification_report.txt # Classification report
│   ├── confusion_matrix.png     # Confusion matrix visualization
│   ├── best_model.pth           # Checkpoint with best validation accuracy
│   └── misclassified/           # Misclassified images for analysis
├── .gitignore              # Git ignore rules to exclude unnecessary files
├── real_time.py            # Real-time emotion detection script
├── main.py                 # Main script to train and validate the model
├── README.md               # Project documentation
├── requirements.txt        # Project dependencies
├── LICENSE                 # Apache 2.0 License
```

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/Abantika02/Emotion-Detection-Through-Facial-Expressions.git
   cd Emotion-Detection-Through-Facial-Expressions
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the FER2013 dataset and place the `fer2013.csv` file in the `data/` directory.

---

## **Usage**

### 1. Training and Evaluation
Run the `main.py` script to train, validate, and test the model:
```bash
python main.py
```
- Logs and evaluation metrics will be saved in the `outputs/` directory.
- The trained model weights will be saved in the `model/` directory.

### 2. Real-Time Emotion Detection
Run the `real_time.py` script for real-time emotion detection using your webcam:
```bash
python real_time.py
```

### 3. Analyze Outputs
- **Training Logs**: `outputs/training_logs.txt`
- **Validation Accuracy & Loss**: `outputs/validation_results.txt`
- **Classification Report**: `outputs/classification_report.txt`
- **Confusion Matrix**: `outputs/confusion_matrix.png`

---

## **Model Architecture**

The `EmotionNet` model is based on **EfficientNet-B3**, a highly efficient deep learning architecture optimized for performance and resource usage. Key features include:
- **Pre-trained Weights** for transfer learning.
- **Dropout Layers** for regularization.
- **Fully Connected Layer** for classification into 7 emotion categories.

---

## **Evaluation Metrics**
The project evaluates model performance using:
- **Accuracy**: Overall accuracy on the test set.
- **Precision, Recall, and F1-Score**: Provided in the classification report.
- **Confusion Matrix**: Visual representation of prediction errors.
- **Misclassified Samples**: Up to 10 examples saved for analysis in `outputs/misclassified`.

---

## **References**
- Dataset: [Kaggle - Facial Expression Recognition](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)
- PyTorch Documentation: [https://pytorch.org/docs/](https://pytorch.org/docs/)

---

## **License**
This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details.

