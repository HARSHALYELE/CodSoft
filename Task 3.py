import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('/home/samit/Downloads/Tasks/task 03/creditcard.csv')


print(df.head())
print(df.info())
print(df.describe())

# Normalize the amount column (other columns are already scaled)
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])


X = df.drop('Class', axis=1)
y = df['Class']


oversample = SMOTE()
undersample = RandomUnderSampler()
steps = [('o', oversample), ('u', undersample)]
pipeline = Pipeline(steps=steps)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train, y_train = pipeline.fit_resample(X_train, y_train)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}


for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluation metrics
    print(f'--- {name} ---')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print(f'ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.2f}')
    print('----------------------------------------')



# Use the best model to make predictions on new data (example data)
best_model = models['Random Forest']
new_transaction = pd.DataFrame({
    'V1': [0.1], 'V2': [-0.2], 'V3': [0.1], 'V4': [0.3], 'V5': [-0.4], 'V6': [0.2], 'V7': [-0.1],
    'V8': [0.2], 'V9': [-0.3], 'V10': [0.1], 'V11': [0.2], 'V12': [-0.2], 'V13': [0.3], 'V14': [-0.1],
    'V15': [0.1], 'V16': [0.2], 'V17': [-0.3], 'V18': [0.1], 'V19': [0.2], 'V20': [-0.2], 'V21': [0.3],
    'V22': [-0.1], 'V23': [0.1], 'V24': [0.2], 'V25': [-0.3], 'V26': [0.1], 'V27': [0.2], 'V28': [-0.2],
    'Amount': scaler.transform([[100]])  # Scale the amount as we did in the training
})
new_transaction_prediction = best_model.predict(new_transaction)
print(f'Predicted class for the new transaction: {new_transaction_prediction[0]}')
