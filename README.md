# Botnet Detection with Hybrid FeatureSelection Combining Intersection, Weighting and Voting Analysis

## 📌 Overview
This application is developed for Botnet Detection using several dataset (CTU-13, NCC, NCC-2). The framework is designed to perform **data pre-processing, feature selection, and machine learning classsification with model performance evaluation**. Including various analytical methods and classification algorithms available, this application provides flexibility in exploring the best methods for machine learning-based Botnet Detection.

## 🎯 Objective
The primary goal of this application is to support research in **cybersecurity**, particularly in detecting network attacks using **machine learning** approaches. By utilizing various data processing techniques and feature selection methods, this model aims to enhance attack detection accuracy and optimize execution time.

## 🔧 Processing Stages
This framework consists of **four main stages**:

### 1️⃣ Input Dataset
- Uses the **CTU-13, NCC and NCC-2** dataset.
- The dataset contains botnet activities that may indicate cyber attacks.

### 2️⃣ Pre-processing
Includes several steps:
- **Data Cleansing**: Removes missing (null) values and redundant records based on **IDMEF** standards (need improvement).
- **Label Encoder**: Encode categorial value.
- **Data Normalization**: Standardizes the range of each feature, including converting categorical data into numerical format.
- **Feature Selection**: Utilizes **Chi2**, **ANOVA** and **Information Gain** methods, followed by **Intersection Analysis**, **Weighted Rank** and **Voting Analysis** to determine the best feature set.

### 3️⃣ Anomaly Activity Detection
- Supported algorithms: **k-NN, Decision Tree, Naïve Bayes, Logistic Regression, AdaBoost, ExtraTree, XGBoost, Random Forest, SVC, ANN**.
- Classification is performed using the selected features.

### 4️⃣ Performance Evaluation
- **Model evaluation** based on **accuracy, precision, and recall**, derived from the **confusion matrix (TP, FP, FN, TN)**.
- **Execution time evaluation**: Measures pre-processing time, training time, and testing time.
- Classification results are automatically saved in **collection/classification_result.csv**.

## 🚀 How to Run the Program
### 1️⃣ Environment Setup
1. Create a new **.env** file or duplicate **.env.example**.
2. Specify dataset locations in the following variables:
   ```
   DATA_TRAINING_LOCATION=path_to_training_dataset
   DATA_TESTING_LOCATION=path_to_testing_dataset
   ```
3. Ensure **Python 3** is installed.
4. Install dependencies by running:
   ```
   python install.py  
   or  
   python3 install.py
   ```
5. Run the main program:
   ```
   python main.py  
   or  
   python3 main.py
   ```
6. Use the application as needed.

## 🛠️ Configuration & Features
- **Dataset**: Currently supports **CTU-13, NCCC and NCC-2** or datasets with a similar structure.
- **Encoder**: Uses **Label Encoder**.
- **Feature Selection**: Hybrid Analysis
- **Classification**:
  - Supports multiple **machine learning algorithms**.
  - Allows flexible algorithm selection in a single process.
- **Result Recording**: Classification output is automatically stored in **CSV format**.

## 🤝 How to Contribute
We welcome collaborations for further development! To contribute:
1. **Fork** this repository.
2. Create a new branch for your feature or improvement.
3. **Commit** your changes and push them to your repository.
4. Submit a **Pull Request** to the main repository.

If you have ideas or suggestions, feel free to open an **Issue** or contact us!

---
🔥 Happy researching! We hope this framework helps improve accuracy and efficiency in cyber attack detection. 🚀

