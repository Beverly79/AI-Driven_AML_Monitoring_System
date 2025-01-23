#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import shap


# In[ ]:


import sys
print(sys.executable)


# ### Step 1: Load the Dataset
# downloaded the dataset from Kaggle and placed it in the appropriate directory: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
# 
# or use Kaggle's API:
# 
# !kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml
# 
# !unzip ibm-transactions-for-anti-money-laundering-aml.zip
# 

# In[2]:


# Load the dataset
data_path = 'D:/project_data/AI-Driven_AML_Monitoring_System'
df = pd.read_csv(os.path.join(data_path, 'LI-Small_Trans.csv'))  # Update the path as necessary
df = df.sample(n=1000000)

# Display basic information about the dataset
print(df.info())
print(df.head())


# In[3]:


# Rename columns for easier handling
df = df.rename(columns={
    'Amount Received': 'amount_received',
    'Receiving Currency': 'receiving_currency',
    'Amount Paid': 'amount_paid',
    'Payment Currency': 'payment_currency',
    'Payment Format': 'payment_format',
    'Is Laundering': 'is_laundering'
})


# ### Step 2: Data Preprocessing

# In[4]:


# Handle missing values
df = df.dropna()

# Convert the target column to binary format
df['is_laundering'] = df['is_laundering'].astype(int)

# Encode categorical columns using Label Encoding
categorical_columns = ['From Bank', 'To Bank', 'receiving_currency', 'payment_currency', 'payment_format']
for col in categorical_columns:
    df[col] = df[col].astype('category').cat.codes

# Initialize LabelEncoder
le = LabelEncoder()

# Encode the 'Account' and 'Account.1' columns
for col in ['Account', 'Account.1']:
    df[col] = le.fit_transform(df[col])

# Select relevant features and target
X = df.drop(['Timestamp', 'is_laundering'], axis=1)  # Drop Timestamp and target
y = df['is_laundering']

# Check the transformed data types
print(X.dtypes)



# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


# ### Step 3: Exploratory Data Analysis (EDA)

# In[5]:


# Visualize the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='is_laundering', data=df)
plt.title('Class Distribution (Non-Laundering vs. Laundering)')
plt.show()

# Visualize correlations between features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()


# ### Step 4: Model Training

# In[ ]:


# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=1)

# Define hyperparameters for tuning
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform Grid Search with Cross-Validation
rf_grid = GridSearchCV(rf, rf_params, scoring='roc_auc', cv=3, n_jobs=-1)
rf_grid.fit(X_train, y_train)

# Display the best parameters
print("Best Parameters:", rf_grid.best_params_)


# ### Step 5: Model Evaluation

# In[ ]:


# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print("AUC-ROC Score:", roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Laundering', 'Laundering'], 
                yticklabels=['Non-Laundering', 'Laundering'])
    plt.title('Confusion Matrix')
    plt.show()

# Evaluate the best model
best_rf = rf_grid.best_estimator_
evaluate_model(best_rf, X_test, y_test)


# ### Step 6: Explainability with SHAP

# In[ ]:


# Initialize the SHAP explainer
explainer = shap.Explainer(best_rf)
shap_values = explainer(X_test)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test)


# ### Step 7: Deployment Simulation

# In[ ]:


# Example of new transaction data
new_data = pd.DataFrame({
    'transaction_amount': [5000, 15000],
    'transaction_type': [1, 2],  # Ensure these match the encoded values
    'account_type': [0, 1],      # Ensure these match the encoded values
    'transaction_date': [20250123, 20250123],  # Use appropriate date format
    'account_balance': [20000, 5000]
})

# Predict suspicious activities
predictions = best_rf.predict(new_data)
proba = best_rf.predict_proba(new_data)[:, 1]
print("Predictions for New Data:", predictions)
print("Prediction Probabilities:", proba)

