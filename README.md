# Lung-Cancer-Survival-Prediction-using-Machine-Learning
A predictive ML project that estimates lung cancer patient survival outcomes based on diagnostic, clinical, and lifestyle factors using Logistic Regression, Random Forest, and SVM.
# 🫁 Lung Cancer Survival Prediction using Machine Learning

This project predicts **lung cancer patient survival outcomes** based on diagnostic, clinical, and lifestyle data. It uses multiple supervised learning algorithms to classify whether a patient **survived or not** after lung cancer diagnosis and treatment.

---

## 🎯 Objective
To build a machine learning model that predicts the **survival status** of lung cancer patients based on demographic, clinical, and lifestyle-related features.  
The aim is to assist healthcare professionals in understanding critical survival factors and improving patient prognosis.

---

## 📘 Dataset Overview

### 📍 Source
A dataset containing detailed medical and demographic information of patients diagnosed with **lung cancer**.

### 🧩 Description of Columns

| Feature | Description |
|----------|-------------|
| id | Unique patient identifier |
| age | Patient’s age at diagnosis |
| gender | Patient gender (Male/Female) |
| country | Country or region of residence |
| diagnosis_date | Date of lung cancer diagnosis |
| cancer_stage | Stage of cancer (Stage I–IV) |
| family_history | Family history of cancer (Yes/No) |
| smoking_status | Smoking habit (Current, Former, Never, Passive) |
| bmi | Body Mass Index at diagnosis |
| cholesterol_level | Cholesterol level of the patient |
| hypertension | Presence of hypertension (Yes/No) |
| asthma | Asthma condition (Yes/No) |
| cirrhosis | Liver cirrhosis condition (Yes/No) |
| other_cancer | Previous or concurrent other cancer (Yes/No) |
| treatment_type | Type of treatment received (Surgery, Chemotherapy, Radiation, Combined) |
| end_treatment_date | Date treatment completed or patient death recorded |
| **survived** | Target variable — 1 = Survived, 0 = Did not survive |

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Dropped irrelevant columns (`id`, `country`)  
- Converted date columns (`diagnosis_date`, `end_treatment_date`) to datetime format  
- Computed **treatment duration** as new feature (`treatment_duration_days`)  
- Encoded categorical variables into numeric form  
- Applied **StandardScaler** to scale numerical features  

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized class distribution of survival outcomes  
- Plotted histograms for key numeric features (`age`, `bmi`, `cholesterol_level`, `treatment_duration_days`)  
- Displayed **correlation heatmap** of clinical and demographic features  

### 3️⃣ Model Training and Hyperparameter Tuning
Used three supervised ML algorithms:

| Model | Description | Tuning Method |
|--------|-------------|---------------|
| Logistic Regression | Baseline linear classifier | GridSearchCV (C parameter) |
| Random Forest Classifier | Ensemble learning method | GridSearchCV (n_estimators, max_depth) |
| Support Vector Machine (SVM) | RBF kernel-based classifier | GridSearchCV (C, gamma) |

Each model was trained, tested, and evaluated on 80/20 split data.

---

## 📊 Model Evaluation

| Model | Accuracy | AUC Score |
|--------|-----------|-----------|
| Logistic Regression | ~85% | 0.88 |
| Random Forest (Tuned) | **91%** | **0.93** |
| SVM (Tuned) | **89%** | **0.91** |

**Evaluation Metrics Used:**
- Accuracy
- Precision, Recall, F1-score  
- ROC Curve and AUC Visualization  

---

## 📈 Key Visualizations
- **Class distribution**: Balance of survival vs. non-survival  
- **Feature correlation heatmap**  
- **ROC Curves** for all models  
- **Feature importance plot** from Random Forest  

---

## 🧠 Technologies Used
- **Python 3.8+**
- **Libraries:**
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `pickle`

---
