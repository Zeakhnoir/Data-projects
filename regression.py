import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

# ----------------------------
# 1. Load the Transformed Data
# ----------------------------
df = pd.read_csv('train_transformed.csv')

# Separate features and target variable (SalePrice as output)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# ----------------------------
# 2. Build a Pipeline with Imputation, Scaling, Polynomial Feature Expansion, and Ridge Regression
# ----------------------------
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean value
    ('scaler', StandardScaler()),                 # Scale the features
    ('poly', PolynomialFeatures(include_bias=False)),  # Create polynomial features (exclude bias)
    ('ridge', Ridge())                             # Ridge regression model
])

# ----------------------------
# 3. Set Up Grid Search with Cross Validation
# ----------------------------
param_grid = {
    'poly__degree': [1, 2, 3],                  # Test polynomial degrees: linear, quadratic, and cubic
    'ridge__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]  # Different regularization strengths
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# ----------------------------
# 4. Fit the Model Using Grid Search
# ----------------------------
grid_search.fit(X, y)

# ----------------------------
# 5. Output the Best Parameters and CV Score
# ----------------------------
print("Best parameters:", grid_search.best_params_)
print("Best CV score (negative MSE):", grid_search.best_score_)

# Optionally, use the best estimator for predictions:
best_model = grid_search.best_estimator_
predictions = best_model.predict(X)
