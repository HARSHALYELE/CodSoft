
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("C:\Users\Harshal\Downloads\task02\task02\IMDb Movies India.csv", encoding='latin1')

# Explore the dataset
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Handle missing values for non-numeric columns (if any)
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')

# Convert categorical data into numerical representations
le = LabelEncoder()
le_dict = {}
for col in ['Genre', 'Director', 'Actor 1']:
    le_dict[col] = le.fit(df[col])
    df[col] = le_dict[col].transform(df[col])

# Split the dataset into training and testing sets
X = df.drop('Rating', axis=1)
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert X_train to float

# Ensure the data is in the correct format

X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

y_train = y_train.values.reshape(-1, 1)  # reshape y_train
y_test = y_test.values.reshape(-1, 1)  # reshape y_test

# Define the models
models = [
    LinearRegression(),
    Ridge(alpha=1.0),
    Lasso(alpha=0.1),
    DecisionTreeRegressor(),
    RandomForestRegressor(n_estimators=100),
    GradientBoostingRegressor()
]

# Train and evaluate each model
for model in models:
    model_name = model.__class__.__name__
    print(f'Training {model_name}...')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{model_name} - MSE: {mse:.2f}, R²: {r2:.2f}')
    print('----------------------------------------')

# Select the best model based on R² score
best_model = models[0]
best_r2 = 0
for model in models:
    model_name = model.__class__.__name__
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    if r2 > best_r2:
        best_model = model
        best_r2 = r2
print(f'Best model: {best_model.__class__.__name__} with R²: {best_r2:.2f}')

# Use the best model to make predictions on new data
new_movie = pd.DataFrame({
    'Genre': le_dict['Genre'].transform(['Action']),
    'Director': le_dict['Director'].transform(['Christopher Nolan']),
    'Actor 1': le_dict['Actor 1'].transform(['Christian Bale']),
})
new_movie = new_movie.values.reshape(-1, new_movie.shape[1])  # reshape new_movie
new_movie_rating = best_model.predict(new_movie)
print(f'Predicted rating for the new movie: {new_movie_rating[0]:.2f}')