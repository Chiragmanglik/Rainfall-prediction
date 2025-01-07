# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler                  
from sklearn import metrics                     
from sklearn.svm import SVC  
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings 
warnings.filterwarnings('ignore')

# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Loading the dataset
file_path = '/content/drive/My Drive/Rainfall.csv'  # Adjust the path as needed
df = pd.read_csv(file_path)

# Checking for null values and cleaning data
df.isnull().sum()
df.columns
df.rename(str.strip, axis='columns', inplace=True)  # Stripping whitespace from column names
df.columns

# Fill missing values with mean
for col in df.columns:
    if df[col].isnull().sum() > 0:
        val = df[col].mean()
        df[col] = df[col].fillna(val)

# Verify that all null values are handled
print("Total missing values:", df.isnull().sum().sum())

# Adding 'month' column based on the assumption of 30 days per month (367 rows)
df['month'] = ((df.index // 30) + 1)  # +1 to start month count from 1

# If you want the months to cycle through the year (1 for Jan, 2 for Feb, ..., 12 for Dec):
df['month'] = (df.index // 30) % 12 + 1

# Check the first few rows to ensure the month column is added correctly
print(df.head())

# Replacing 'yes' and 'no' in the 'rainfall' column with 1 and 0, respectively
df['rainfall'] = df['rainfall'].replace({'yes': 1, 'no': 0})

# Visualizing rainfall distribution
plt.pie(df['rainfall'].value_counts().values,
        labels=df['rainfall'].value_counts().index,
        autopct='%1.1f%%')
plt.show()

# Grouping by 'rainfall' and calculating means
print(df.groupby('rainfall').mean())

# Selecting numerical features
features = list(df.select_dtypes(include=np.number).columns)
features.remove('day')  # Fixed syntax error here
print("Features:", features)

# Distribution plots for numerical features
plt.subplots(figsize=(15, 8))
for i, col in enumerate(features):
    plt.subplot(3, 4, i + 1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()

# Box plots for numerical features
plt.subplots(figsize=(15, 8))
for i, col in enumerate(features):
    plt.subplot(3, 4, i + 1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()

# Heatmap of correlations
plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

# Dropping unnecessary columns
df.drop(['maxtemp', 'mintemp'], axis=1, inplace=True)

# Splitting features and target
features = df.drop(['day', 'rainfall'], axis=1)
target = df['rainfall']

# Splitting the dataset into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                  test_size=0.2,
                                                  stratify=target,
                                                  random_state=2)

# Balancing the dataset
ros = RandomOverSampler(sampling_strategy='minority', random_state=22)
X, Y = ros.fit_resample(X_train, Y_train)

# Normalizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

# Initializing models
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]

# Training models and evaluating performance
for i in range(3):
    models[i].fit(X, Y)

    print(f'{models[i]}:')

    train_preds = models[i].predict_proba(X)
    print('Training Accuracy:', metrics.roc_auc_score(Y, train_preds[:, 1]))

    val_preds = models[i].predict_proba(X_val)
    print('Validation Accuracy:', metrics.roc_auc_score(Y_val, val_preds[:, 1]))
    print()

# Confusion Matrix for SVC model
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(models[2], X_val, Y_val)
plt.show()

# Grouping by 'month' and calculating the mean for numeric columns
monthly_grouped = df.groupby('month').mean()

# Print the aggregated monthly data
print(monthly_grouped)
