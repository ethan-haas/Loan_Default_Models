# Loan Default Prediction Models

This repository includes multiple machine learning models used for predicting loan defaults. All models are trained on the `loan_data_set.csv` dataset.

## Models

### 1. Random Forest Classifier

The first model is a simple implementation of the `RandomForestClassifier` from sklearn. 

Key Features:
- Handles missing data using modes for categorical features and mean for numerical features.
- Uses LabelEncoder for converting categorical data to numerical data.
- Predicts the loan status and evaluates performance using classification report and accuracy.

### 2. Random Forest with RFECV and GridSearchCV

This model enhances the previous one by incorporating feature selection and hyperparameter tuning.

Key Features:
- Utilizes Recursive Feature Elimination with Cross-Validation (RFECV) for feature selection.
- Employs GridSearchCV for hyperparameter tuning.
- After determining the best parameters, it retrains the model and evaluates performance.

### 3. XGBoost Classifier

This model implements the XGBoost classifier. 

Key Features:
- Uses cross-validation on the training set.
- Scales the features using StandardScaler from sklearn.
- Fits the model, makes predictions, and evaluates performance.

### 4. Gradient Boosting Classifier with Polynomial Features

This model makes use of the GradientBoostingClassifier and introduces polynomial features.

Key Features:
- Uses PolynomialFeatures from sklearn to create interaction features.
- Fits the model, makes predictions, and evaluates performance.

### 5. Random Forest with Feature Engineering

This model adds feature engineering to the previous Random Forest model.

Key Features:
- Adds a new feature 'TotalIncome' that is the sum of 'ApplicantIncome' and 'CoapplicantIncome'.
- Employs GridSearchCV for hyperparameter tuning.
- Fits the model and makes predictions using the best estimator from GridSearchCV.

### 6. XGBoost Classifier with SMOTE

This model handles imbalanced datasets by performing oversampling of the minority class using SMOTE.

Key Features:
- Uses Synthetic Minority Over-sampling Technique (SMOTE) to handle imbalanced data.
- Fits the model, makes predictions, and evaluates performance.

## Accuracy

The accuracy of each model varies depending on the dataset and the model's configurations. Please refer to the outputs of each model for the accuracy.

## Getting Started

To run these models, clone the repository, install the necessary libraries mentioned in the scripts, and run the Python notebook files.

Please note that all models are designed to run with Python 3.x and require the dataset `loan_data_set.csv` to be in the same directory.

