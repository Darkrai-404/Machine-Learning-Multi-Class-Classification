# Machine Learning | Multi-Class Classification Project

This repository contains a comprehensive machine learning project focused on multi-class classification tasks. The project demonstrates proficiency in data preprocessing, feature engineering, and model evaluation techniques.

## Project Overview

This classification project addresses a multi-class prediction problem with the following key components:

- Data exploration and preprocessing
- Feature engineering and selection
- Handling class imbalance using SMOTE
- Model training and hyperparameter tuning
- Performance evaluation and comparison of multiple classifiers

## Key Features

- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling
- **Dimensionality Reduction**: Feature importance ranking and selection
- **Class Imbalance Handling**: Implementation of SMOTE for balanced training
- **Model Comparison**: Evaluation of 6 different classification algorithms
- **Cross-Validation**: Stratified K-Fold validation for reliable performance assessment

## Models Implemented

- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Random Forest
- Multi-layer Perceptron (Neural Network)
- Support Vector Machine (SVM)

## Results

After comprehensive evaluation, the KNN and Neural Network models demonstrated superior performance:
- KNN Accuracy: 89%
- Neural Network Accuracy: 90.4%


## Technologies Used

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- imbalanced-learn (for SMOTE)

## Getting Started

1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the classification model: `python code/classification_model.py`
