import pandas as pd

# Read the CSV file into a DataFrame

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# List of categorical variables we want to convert
categorical_variables = [
     "MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", 
    "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
    "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC",
    "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive",
    "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"
    ]

# Create dummy variables while dropping the first category for each feature
df_decodedtrain = pd.get_dummies(df_train, columns=categorical_variables, drop_first=True)
df_decodedtest =pd.get_dummies(df_test, columns=categorical_variables, drop_first=True)

# Replace boolean values with 0s and 1s in the entire DataFrame
df_decodedtrain = df_decodedtrain.replace({True: 1, False: 0})
df_decodedtest = df_decodedtest.replace({True: 1, False: 0})


# Drop the 'Id' column from the DataFrame
df_decodedtrain.drop('Id', axis=1, inplace=True)
df_decodedtest.drop('Id', axis=1, inplace=True)

# Fill missing values in all numeric columns with the mean
df_decodedtrain.fillna(df_decodedtrain.mean(), inplace=True)
df_decodedtest.fillna(df_decodedtest.mean(), inplace=True)

# Assume df_decodedtrain and df_decodedtest are your preprocessed training and testing DataFrames
# Get the full list of columns from the training DataFrame (including the target variable)
train_features = df_decodedtrain.columns

# Reindex the test DataFrame so it has exactly the same columns as the training data.
# Missing columns (e.g. the target column "SalePrice") will be added with a default value of 0.
df_decodedtest = df_decodedtest.reindex(columns=train_features, fill_value=0)


# Save the transformed DataFrame to a new CSV file
df_decodedtrain.to_csv('transformed_train.csv', index=False)
df_decodedtest.to_csv('transformed_test.csv', index=False)






