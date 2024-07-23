import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
file_path = '/home/samit/Downloads/Tasks/task01/Titanic-Dataset.csv'
data = pd.read_csv(file_path)

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data.drop(columns=['Cabin'], inplace=True)

# Convert categorical variables to numerical
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)

# Define features and target variable
X = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch','Embarked'])
y = data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

def predict_survival_by_ticket(ticket_number):
    # Load the dataset again to get the original Ticket information
    original_data = pd.read_csv(file_path)
    
    # Find the row corresponding to the ticket number
    passenger_data = original_data[original_data['Ticket'] == ticket_number]
    
    if passenger_data.empty:
        print(f"No passenger found with ticket number: {ticket_number}")
        return

    # Fill missing values
    passenger_data['Age'].fillna(data['Age'].median(), inplace=True)
    
    # Convert categorical variables to numerical
    if 'Sex' in passenger_data.columns:
        passenger_data = pd.get_dummies(passenger_data, columns=['Sex'], drop_first=True)
    
    # Ensure all required columns are present
    passenger_data = passenger_data.drop(columns=['PassengerId', 'Name', 'Cabin', 'SibSp', 'Parch','Embarked'], errors='ignore')
    
    # Align columns with the model input
    for col in X.columns:
        if col not in passenger_data.columns:
            passenger_data[col] = 0
        
    passenger_data = passenger_data[X.columns]

    # Make prediction
    prediction = model.predict(passenger_data)

    # Show the result
    result = "Passenger Survived" if prediction[0] == 1 else "Passenger Did not survive"
    print(f"Prediction Result: {result}")

if __name__ == "__main__":
    # Take user input for ticket number
    ticket_number = input("Enter Ticket Number: ")

    # Predict survival
    predict_survival_by_ticket(ticket_number)
