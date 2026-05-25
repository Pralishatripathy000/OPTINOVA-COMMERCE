#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas numpy matplotlib seaborn scikit-learn xgboost')


# In[1]:


# DATASET LOADING

import pandas as pd
import numpy as np
from tkinter.filedialog import askopenfilename

file_path = askopenfilename()

df = pd.read_csv(file_path)

df.head()


# In[2]:


# BASIC DATA UNDERSTANDING

df.shape


# In[3]:


# DATA INFORMATION

df.info()


# In[4]:


# STATISTICAL SUMMARY

df.describe()


# In[5]:


# COLUMN NAMES

df.columns


# In[6]:


# MISSING VALUES CHECK

df.isnull().sum()


# In[7]:


# DUPLICATE VALUES CHECK

df.duplicated().sum()


# In[8]:


# TARGET VARIABLE DISTRIBUTION

df['Revenue'].value_counts()


# In[10]:


# EDA VISUALIZATIONS

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(x='Revenue', data=df)
plt.title("Revenue Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Revenue', y='BounceRates', data=df)
plt.title("Bounce Rate vs Revenue")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Revenue', y='ExitRates', data=df)
plt.title("Exit Rate vs Revenue")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Revenue', y='PageValues', data=df)
plt.title("Page Value vs Revenue")
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(x='VisitorType', hue='Revenue', data=df)
plt.title("Visitor Type Analysis")
plt.xticks(rotation=15)
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(x='Month', hue='Revenue', data=df)
plt.title("Month-wise Revenue Analysis")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,5))
sns.countplot(x='TrafficType', hue='Revenue', data=df)
plt.title("Traffic Type Analysis")
plt.show()


plt.figure(figsize=(15,10))

numeric_df = df.select_dtypes(include=['int64', 'float64'])

sns.heatmap(numeric_df.corr(), cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()


# In[11]:


# FEATURE ENGINEERING

df['EngagementScore'] = (
    df['ProductRelated'] * df['PageValues']
) / (df['BounceRates'] + 1)

df['ExitIntentScore'] = (
    df['BounceRates'] + df['ExitRates']
)

df = pd.get_dummies(
    df,
    columns=['Month', 'VisitorType', 'Weekend'],
    drop_first=True
)

df['Revenue'] = df['Revenue'].astype(int)

df.head()


# In[12]:


# FEATURE TARGET SPLIT AND SCALING

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Revenue', axis=1)

y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train.shape, X_test.shape


# In[13]:


# LOGISTIC REGRESSION MODEL

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# In[14]:


# XGBOOST MODEL

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred_xgb))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_xgb))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_xgb))


# In[15]:


# FEATURE IMPORTANCE VISUALIZATION

import matplotlib.pyplot as plt
import pandas as pd

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,6))

plt.barh(
    feature_importance['Feature'][:10],
    feature_importance['Importance'][:10],
    color='#6C63FF'
)

plt.gca().invert_yaxis()

plt.title(
    "Top 10 Important Features",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Importance Score", fontsize=12)

plt.grid(
    alpha=0.2,
    linestyle='--'
)

plt.tight_layout()

plt.show()


# In[16]:


# ROC CURVE VISUALIZATION

from sklearn.metrics import roc_curve, auc

y_prob = xgb_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    color='#6C63FF',
    linewidth=2,
    label=f'AUC = {roc_auc:.2f}'
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--',
    color='#B0B0B0'
)

plt.xlabel("False Positive Rate", fontsize=12)

plt.ylabel("True Positive Rate", fontsize=12)

plt.title(
    "ROC Curve - XGBoost",
    fontsize=16,
    fontweight='bold'
)

plt.legend()

plt.grid(alpha=0.2)

plt.tight_layout()

plt.show()


# In[17]:


# CONFUSION MATRIX VISUALIZATION

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='mako',
    cbar=False
)

plt.title(
    "Confusion Matrix - XGBoost",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Predicted Label", fontsize=12)

plt.ylabel("Actual Label", fontsize=12)

plt.tight_layout()

plt.show()


# In[18]:


# PURCHASE PROBABILITY DISTRIBUTION

purchase_probabilities = xgb_model.predict_proba(X_test)[:, 1]

plt.figure(figsize=(8,5))

sns.histplot(
    purchase_probabilities,
    bins=30,
    kde=True,
    color='#2A9D8F'
)

plt.title(
    "Purchase Probability Distribution",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Predicted Purchase Probability", fontsize=12)

plt.ylabel("Frequency", fontsize=12)

plt.grid(alpha=0.2)

plt.tight_layout()

plt.show()


# In[19]:


# TOP FEATURE IMPORTANCE VISUALIZATION

top_features = feature_importance.head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=top_features,
    palette='crest'
)

plt.title(
    "Most Influential Features",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Feature Importance", fontsize=12)

plt.ylabel("Features", fontsize=12)

plt.grid(alpha=0.15)

plt.tight_layout()

plt.show()


# In[20]:


# ABANDONMENT RISK SCORE GENERATION

risk_scores = (1 - purchase_probabilities) * 100

risk_df = pd.DataFrame({
    'ActualRevenue': y_test.values,
    'PurchaseProbability': purchase_probabilities,
    'AbandonmentRiskScore': risk_scores
})

risk_df.head(10)


# In[21]:


# ABANDONMENT RISK SCORE VISUALIZATION

plt.figure(figsize=(8,5))

sns.histplot(
    risk_df['AbandonmentRiskScore'],
    bins=30,
    kde=True,
    color='#E76F51'
)

plt.title(
    "Cart Abandonment Risk Distribution",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Abandonment Risk Score", fontsize=12)

plt.ylabel("Frequency", fontsize=12)

plt.grid(alpha=0.2)

plt.tight_layout()

plt.show()


# In[22]:


# HIGH RISK SESSION ANALYSIS

high_risk_sessions = risk_df[
    risk_df['AbandonmentRiskScore'] > 70
]

high_risk_sessions.head(10)


# In[23]:


# TOP HIGH-RISK SESSIONS VISUALIZATION

top_risk = high_risk_sessions.sort_values(
    by='AbandonmentRiskScore',
    ascending=False
).head(15)

plt.figure(figsize=(10,6))

sns.barplot(
    x='AbandonmentRiskScore',
    y=top_risk.index,
    data=top_risk,
    palette='flare'
)

plt.title(
    "Top High-Risk User Sessions",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Abandonment Risk Score", fontsize=12)

plt.ylabel("Session Index", fontsize=12)

plt.grid(alpha=0.15)

plt.tight_layout()

plt.show()


# In[24]:


# CONVERSION INSIGHTS BY VISITOR TYPE

visitor_conversion = df.groupby('VisitorType_Returning_Visitor')[
    'Revenue'
].mean().reset_index()

visitor_conversion['VisitorType'] = [
    'New Visitor',
    'Returning Visitor'
]

plt.figure(figsize=(7,5))

sns.barplot(
    x='VisitorType',
    y='Revenue',
    data=visitor_conversion,
    palette='viridis'
)

plt.title(
    "Conversion Rate by Visitor Type",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Visitor Type", fontsize=12)

plt.ylabel("Average Conversion Rate", fontsize=12)

plt.grid(alpha=0.15)

plt.tight_layout()

plt.show()


# In[25]:


# MONTH-WISE CONVERSION RATE ANALYSIS

month_conversion = df.groupby('Month_Nov')[
    'Revenue'
].mean().reset_index()

plt.figure(figsize=(7,5))

sns.barplot(
    x=month_conversion.index,
    y='Revenue',
    data=month_conversion,
    palette='magma'
)

plt.title(
    "Monthly Conversion Trend",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Month Category", fontsize=12)

plt.ylabel("Average Conversion Rate", fontsize=12)

plt.grid(alpha=0.15)

plt.tight_layout()

plt.show()


# In[26]:


# MODEL SAVING

import joblib

joblib.dump(xgb_model, 'optinova_xgb_model.pkl')

joblib.dump(scaler, 'optinova_scaler.pkl')

print("Model and scaler saved successfully.")


# In[27]:


# SAMPLE PREDICTION TEST

sample_data = X_test[0].reshape(1, -1)

prediction = xgb_model.predict(sample_data)

probability = xgb_model.predict_proba(sample_data)

print("Prediction:", prediction[0])

print("Purchase Probability:", probability[0][1])

print("Abandonment Risk:", (1 - probability[0][1]) * 100)


# In[29]:


import joblib

joblib.dump(xgb_model, 'optinova_xgb_model.pkl')

joblib.dump(scaler, 'optinova_scaler.pkl')


# In[30]:


import joblib

joblib.dump(xgb_model, 'optinova_xgb_model.pkl')

joblib.dump(scaler, 'optinova_scaler.pkl')


# In[31]:


import os

os.listdir()


# In[ ]:




