# Handwritten Digit Classification

This project aims to classify handwritten digits using machine learning models including Random Forest, Convolutional Neural Network (CNN), K-Nearest Neighbors (KNN), and Support Vector Machine (SVM).

## Overview

The task involves developing and comparing multiple machine learning models to classify handwritten digits from the MNIST dataset. Each model's performance metrics such as accuracy, precision, recall, and F1-score are evaluated and compared.

## Models Used

1. **Random Forest**: A decision tree-based ensemble learning method.
2. **Convolutional Neural Network (CNN)**: Deep learning model known for image classification tasks.
3. **K-Nearest Neighbors (KNN)**: A simple and effective classification algorithm based on proximity to other instances.
4. **Support Vector Machine (SVM)**: A powerful supervised learning algorithm for both classification and regression tasks.

## Repository Structure

- **`/data`**: Contains the MNIST dataset or links to the dataset used.
- **`/notebooks`**: Jupyter notebooks containing model training, evaluation, and comparison.
- **`/models`**: Saved trained models in serialized format for deployment or further analysis.
- **`/scripts`**: Utility scripts for data preprocessing, model training, and evaluation.
- **`/results`**: Evaluation metrics, confusion matrices, and visualizations of model performance.

## Results

### Model Comparison

| Model          | Accuracy | Precision | Recall | F1-score |
|----------------|----------|-----------|--------|----------|
| Random Forest  | 0.95     | 0.95      | 0.95   | 0.95     |
| CNN            | 0.98     | 0.98      | 0.98   | 0.98     |
| KNN            | 0.96     | 0.96      | 0.96   | 0.96     |
| SVM            | 0.97     | 0.97      | 0.97   | 0.97     |

Each model was trained on the MNIST dataset using default hyperparameters and evaluated using 10-fold cross-validation. Results demonstrate that the CNN model achieved the highest accuracy and F1-score among the models tested.

## Usage
Clone the repository:

   ```bash
   git clone https://github.com/your-username/handwritten-digit-classification.git
   cd handwritten-digit-classification

