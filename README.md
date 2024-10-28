# Binary Classification Streamlit App 🍄

This Streamlit app is designed to classify mushrooms as either edible or poisonous using binary classification models. It provides an interactive interface for training, evaluating, and comparing different machine learning models.

## Features
- **Model Selection**: Choose between SVM, Logistic Regression, and Random Forest.
- **Hyperparameter Tuning**: Adjust model hyperparameters for customized training.
- **Performance Metrics**: View metrics including Accuracy, Precision, Recall, Confusion Matrix, ROC Curve, and Precision-Recall Curve.
- **Data Display**: Option to view raw mushroom dataset.

## Getting Started

### Prerequisites
Ensure you have the required libraries:
```bash
pip install -r requirements.txt

### Running the App
```bash
streamlit run app.py

### Project Structure
- app.py: Main Streamlit app file.
- data/: Directory containing the mushrooms.csv dataset.
- requirements.txt: Lists required dependencies.

Usage
Use the sidebar to select a classifier, adjust parameters, and view model performance metrics. The app dynamically displays evaluation metrics to help analyze each model's effectiveness.
