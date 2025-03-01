House Price Prediction Workflow

This repository contains two Python scripts for a house price prediction project:

Preprocessing Script (preprocess.py)
This script reads raw training and testing CSV files, applies one-hot encoding to categorical features, replaces boolean values with 0s and 1s, handles missing values, aligns feature columns between the datasets, and outputs preprocessed files.
Regression Script (regression.py)
This script loads the preprocessed data, applies a log transformation to the target variable (SalePrice) for training, trains a linear regression model, makes predictions on the test data, converts the predictions back to the original scale, and outputs a CSV file with the predictions.
Files in the Repository

train.csv: Raw training data (must include the target variable SalePrice and an Id column).
test.csv: Raw testing data (includes an Id column; does not include SalePrice).
transformed_train.csv: Preprocessed training data (output from preprocess.py).
transformed_test.csv: Preprocessed testing data (output from preprocess.py).
testing_data_with_predictions.csv: Final output containing the test data with predicted sale prices (output from regression.py).
preprocess.py: Script for data preprocessing.
regression.py: Script for training the regression model and generating predictions.
README.md: This file.
Preprocessing Workflow (preprocess.py)

Load Data:
Read train.csv and test.csv into Pandas DataFrames.
One-Hot Encoding:
Convert categorical variables (e.g., MSSubClass, MSZoning, etc.) into dummy variables using pd.get_dummies with drop_first=True to avoid multicollinearity.
Boolean Replacement:
Replace any boolean values with 0 and 1.
Drop Unnecessary Columns:
Remove the Id column from both datasets.
Fill Missing Values:
Fill missing values in all numeric columns with their column means.
Feature Alignment:
Align the test DataFrame’s columns with the training DataFrame’s features so that both datasets share the same columns (missing columns are filled with 0).
Output Files:
Save the preprocessed training and testing data as transformed_train.csv and transformed_test.csv.
Regression Workflow (regression.py)

Load Preprocessed Data:
Read transformed_train.csv and transformed_test.csv.
Prepare Data:
Remove the target variable (SalePrice) from the test dataset.
Log Transformation of the Target:
Since SalePrice is typically right-skewed, apply np.log1p to the target variable in the training data to stabilize variance and improve linearity:
y = np.log1p(df_train['SalePrice'])
Model Training:
Train a LinearRegression model using the training features (X) and the log-transformed target (y).
Prediction:
Generate predictions on the test set (which will be in log scale).
Inverse Transformation:
Convert the log-scaled predictions back to the original scale using np.expm1:
predictions = np.expm1(log_predictions)
Output Predictions:
Append the final predictions to the test DataFrame and save as testing_data_with_predictions.csv.
How to Run

Install Dependencies:
Ensure you have Python 3.x installed along with the necessary libraries. You can install required packages via pip:
pip install pandas numpy scikit-learn matplotlib
Run Preprocessing:
Place train.csv and test.csv in the project directory and run:
python preprocess.py
This creates transformed_train.csv and transformed_test.csv.
Run Regression:
After preprocessing is complete, run:
python regression.py
This will generate testing_data_with_predictions.csv containing your predictions.
Notes

Log Transformation:
The log transformation (np.log1p) helps manage skewed data distributions common in price predictions. After model prediction, the inverse transformation (np.expm1) returns predictions to the original scale.
Feature Consistency:
The preprocessing step ensures that both training and testing datasets have identical features, which is crucial for accurate model predictions.
Model Limitations:
Linear Regression may produce negative predictions. Using the log transformation often helps reduce this issue, but you may explore other models (like tree-based models) if further improvement is needed.
Conclusion

This project workflow provides a structured approach to data preprocessing and regression modeling for house price prediction. The steps include robust handling of categorical variables, missing data, and target variable transformations, ensuring that your model training and predictions are reliable.

Feel free to modify the scripts as needed for your specific data and requirements.
