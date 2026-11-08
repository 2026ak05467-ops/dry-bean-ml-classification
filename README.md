# Dry Bean Classification Using Machine Learning

## Machine Learning Assignment 2

**Student Name:** Akash Shukla  
**Course:** M.Tech Artificial Intelligence and Machine Learning  
**Institution:** BITS Pilani WILP  

---

## 1. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for identifying dry bean varieties using geometric and shape-related features extracted from bean images.

Five classification algorithms were trained and evaluated using the same training and testing datasets:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

A Streamlit web application was developed to evaluate the saved models using an uploaded test CSV file. The application loads the previously trained model and does not retrain it during evaluation.

---

## 2. Dataset Description

The Dry Bean Dataset from the UCI Machine Learning Repository was used for this project.

The dataset contains:

- 13,611 observations
- 16 numerical input features
- 7 dry bean classes
- No missing values
- One target column named `Class`

The seven bean classes are:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

The input features represent geometric and shape-related characteristics such as area, perimeter, major axis length, minor axis length, aspect ratio, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness, and shape factors.

Dataset source:

https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

---

## 3. Data Preprocessing

The following preprocessing steps were performed:

1. The dataset was loaded using pandas.
2. Dataset dimensions, data types, descriptive statistics, class distribution, and missing values were examined.
3. The `Class` column was used as the target variable.
4. The remaining 16 columns were used as input features.
5. The dataset was divided into 80% training data and 20% testing data.
6. Stratified sampling was used to maintain class proportions.
7. `random_state=42` was used for reproducibility.
8. StandardScaler was applied to Logistic Regression, KNN, and SVM using scikit-learn pipelines.
9. The preprocessing step and classifier were saved together inside each applicable pipeline.

The split produced:

- Training records: 10,888
- Testing records: 2,723

---

## 4. Machine Learning Models

### 4.1 Logistic Regression

Logistic Regression was implemented using standardized numerical features. The maximum number of iterations was increased to support convergence for the multiclass dataset.

### 4.2 Decision Tree

The Decision Tree classifier was trained without feature scaling because tree-based methods are generally not sensitive to differences in feature magnitude.

### 4.3 K-Nearest Neighbors

KNN was implemented with five neighbors. StandardScaler was used because distance-based algorithms are affected by feature magnitude.

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes was implemented using the original numerical feature values. The model assumes conditional independence among features for each class.

### 4.5 Support Vector Machine

SVM was implemented with probability estimation enabled. StandardScaler was used before model training. Probability estimates were used to calculate multiclass AUC.

---

## 5. Evaluation Metrics

All models were evaluated using the following metrics:

- Accuracy
- Area Under the ROC Curve
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Matthews Correlation Coefficient

For multiclass AUC, the One-vs-Rest method with weighted averaging was used.

---

## 6. Model Performance Results

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.8917 | 0.9330 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| K-Nearest Neighbors | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Gaussian Naive Bayes | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Support Vector Machine | **0.9221** | **0.9937** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

---

## 7. Model Observations

### Logistic Regression

Logistic Regression achieved 92.14% accuracy, an AUC of 0.9934, and an MCC of 0.9050. The model provided strong multiclass classification performance and performed almost as well as SVM.

### Decision Tree

Decision Tree achieved 89.17% accuracy and an MCC of 0.8691. Its performance was lower than Logistic Regression, KNN, and SVM. The result indicates that a single tree may not generalize as effectively as the stronger models used in this experiment.

### K-Nearest Neighbors

KNN achieved 91.66% accuracy, an AUC of 0.9812, and an F1 score of 0.9168. The result indicates that similar bean samples form useful neighborhoods in the standardized feature space.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved 76.39% accuracy, an F1 score of 0.7615, and an MCC of 0.7154. It produced the lowest overall performance among the five models.

Several geometric measurements in the dataset are related to one another. Therefore, the conditional-independence assumption of Naive Bayes may not represent the feature relationships as effectively as the other models.

### Support Vector Machine

SVM achieved the best overall results:

- Accuracy: 92.21%
- AUC: 0.9937
- Precision: 92.24%
- Recall: 92.21%
- F1 Score: 92.22%
- MCC: 0.9059

The confusion matrix and classification report also demonstrated strong classification performance across the seven dry bean varieties.

### Best Model

Support Vector Machine was identified as the best-performing model because it achieved the highest value for all six evaluation measures.

The performance difference between SVM and Logistic Regression was small. SVM accuracy was only 0.07 percentage points higher than Logistic Regression accuracy. Therefore, Logistic Regression may still be considered when model simplicity and interpretability are important.

---

## 8. Streamlit Application

The Streamlit application allows the user to:

1. Upload a CSV test dataset.
2. Select one of the five trained classifiers.
3. Load the selected saved model.
4. Generate predictions without retraining the classifier.
5. Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
6. Display the confusion matrix.
7. Display the detailed classification report.

The uploaded CSV file must contain:

- The same 16 input features used during training
- A target column named `Class`

---

## 9. Repository Structure
