# 💳 Credit Card Fraud Detection
This project aims to detect fraudulent credit card transactions using machine learning and deep learning techniques.  
It was built as part of my internship to understand how to handle **imbalanced datasets**, **build classification models**, and **evaluate fraud detection systems** effectively.

---

## 📂 Project Overview

Credit card fraud is a major concern in financial systems. The goal of this project is to build a model that can identify potentially fraudulent transactions from a large dataset with highly imbalanced classes.

This notebook includes:
- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Handling class imbalance using **SMOTE**  
- Model training using **Logistic Regression**, **Random Forest**, and **XGBoost**  
- **Hyperparameter tuning** for model optimization  
- **Cross-validation** for reliable results  
- **Autoencoder-based anomaly detection** (deep learning approach)  
- **Visual dashboards** for performance and insights

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, TensorFlow, Imbalanced-learn  
- **Tools:** Jupyter Notebook / Google Colab  

---

## Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 🚀 Implementation Steps

1. **Data Loading and Preprocessing**
   - Imported and explored the dataset
   - Checked for missing values and class distribution

2. **Exploratory Data Analysis (EDA)**
   - Visualized data imbalance
   - Compared statistical differences between legit and fraudulent transactions

3. **Class Imbalance Handling**
   - Used **SMOTE** to oversample the minority class

4. **Model Training**
   - Trained Logistic Regression, Random Forest, and XGBoost models

5. **Model Evaluation**
   - Measured accuracy, ROC-AUC, and confusion matrices
   - Compared models using visual ROC curves and bar charts

6. **Model Optimization**
   - Applied **RandomizedSearchCV** and **GridSearchCV** for hyperparameter tuning
   - Used **cross-validation** to ensure reliability

7. **Deep Learning (Autoencoder)**
   - Built an Autoencoder model for unsupervised fraud detection
   - Calculated reconstruction error and evaluated ROC-AUC

8. **Visual Dashboard**
   - Added plots for:
     - Class distribution (before and after SMOTE)
     - Feature importance
     - Confusion matrix heatmap
     - Model accuracy and ROC-AUC comparison

---

## 📊 Results

| Model | Accuracy | ROC-AUC | Notes |
|:------|:---------:|:--------:|:------|
| Logistic Regression | ~94% | 0.94 | Baseline linear model |
| Random Forest | ~99% | 0.99 | Best performing classical model |
| XGBoost | ~99% | 0.99 | Competitive, efficient |
| Autoencoder | ~97% | 0.97 | Effective unsupervised detection |

---

## 🔍 Future Improvements
- Use more advanced models like **CatBoost** or **LightGBM**  
- Build a **Streamlit dashboard** for real-time fraud detection  
- Deploy the model using **Flask** or **FastAPI**  
- Use **SHAP** for feature interpretability  

---

## 🏁 Conclusion

This project demonstrates how machine learning and deep learning techniques can detect fraudulent credit card transactions efficiently.  
It highlights the importance of **data balance**, **model comparison**, and **visual insights** for better understanding and decision-making.


[LinkedIn](https://www.linkedin.com) | [GitHub](https://github.com)  

