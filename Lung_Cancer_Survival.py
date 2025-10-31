
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# %%
#Load and inspect the data
df = pd.read_csv('D:\zoology download\Projects-20240722T093004Z-001\Projects\lung_cancer\Lung Cancer\dataset_med.csv')
print(df.shape)
print(df.columns.tolist())
df.head()
# %%
# Drop ID
df = df.drop(columns=['id'])
#%%
# Parse dates and compute treatment duration
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'])
df['treatment_duration_days'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days
# %%
df.select_dtypes(include='object').columns

# %%
# Map binary categorical to numeric
df['family_history'] = df['family_history'].map({'Yes': 1, 'No': 0})
# %%
cat_col= df.select_dtypes(include='object').columns
# %%
#convert the categorical columns into numerical values
for col in cat_col:
    print(col)
    print((df[col].unique()), list(range(df[col].nunique())))
    df[col].replace((df[col].unique()), range(df[col].nunique()), inplace=True)
    print('*'*90)
    print()

# %%
# Drop original date columns (we have duration) and country (optional)
df = df.drop(columns=['diagnosis_date', 'end_treatment_date', 'country'])
# %%
# Separate features and target
X = df.drop(columns=['survived'])
y = df['survived']
# %%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
print("Train set size:", X_train.shape)
print("Test set size:", X_test.shape)
# %%
 # Feature scaling (standardize numeric columns)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns,
index=X_test.index)

# %%
#EDA
# Target class balance
sns.countplot(x=y_train, palette='pastel')
plt.title("Class distribution (0=Did not survive, 1=Survived)")
plt.show()
# %%
# Histograms of numeric features
X_train[['age', 'bmi', 'cholesterol_level', 'treatment_duration_days']].hist(bins=20, figsize=(10,8), color='skyblue')
plt.tight_layout()
plt.show()
# %%
# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(pd.concat([X_train, y_train], axis=1).corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()
#%%
#Hyperparameter Tuning
# Logistic Regression hyperparameter tuning
param_grid_lr = {'C': [0.01, 0.1, 1, 10]}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid_lr, cv=5)
grid_lr.fit(X_train, y_train)
print("Best LR params:", grid_lr.best_params_)
#%%
# Random Forest tuning
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)
print("Best RF params:", grid_rf.best_params_)
#%%
# SVM tuning
param_grid_svm = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
grid_svm = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42), param_grid_svm, cv=5)
grid_svm.fit(X_train, y_train)
print("Best SVM params:", grid_svm.best_params_)
# %%
#Model Training
lr = LogisticRegression(**grid_lr.best_params_, max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
# %%
print(f'Logistic Regression Score: {accuracy_score(y_test, y_pred_lr)}')
# %%
rf = RandomForestClassifier(**grid_rf.best_params_, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest Classifier's Accuracy: ", accuracy_score(y_test, y_pred_rf))
# %%
svm = SVC(**grid_svm.best_params_, kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print("SVM's Accuracy: ", accuracy_score(y_test, y_pred_svm))
#%%
#Evaluation Metrics
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy (LR): {acc_lr:.3f}, (RF): {acc_rf:.3f}, (SVM): {acc_svm:.3f}")
#%%
# ROC curves
models_probs = {
 'LogisticRegression': lr.predict_proba(X_test)[:,1],
 'RandomForest': rf.predict_proba(X_test)[:,1],
 'SVM': svm.predict_proba(X_test)[:,1]
}
plt.figure(figsize=(6,5))
for name, prob in models_probs.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()
#%%
#creating pickle file for LR
import pickle
with open('Lung_LR.pkl', 'wb') as file:
    pickle.dump(lr, file)
#%%
#Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X_train.columns
# Plot top 10 features
plt.figure(figsize=(8,6))
sns.barplot(x=importances[indices][:10], y=feature_names[indices][:10], palette='viridis')
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
# %%
