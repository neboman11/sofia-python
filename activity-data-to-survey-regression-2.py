import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb

# Step 1: Load Data
# Assuming your CSV files are named file1.csv, file2.csv, ...
# Load each CSV file separately
list_of_csv_files = glob.glob("./user_activity_data/*.csv")
data_frames = [pd.read_csv(file) for file in list_of_csv_files]

# Read survey data
survey_df = pd.read_csv(
    "survey_data/user_survey_data.csv"
)  # Assuming the survey data is in a file named survey_data.csv

# Step 2: Preprocess Data
# Handle missing values, encode categorical variables, scale numerical features, etc.
y = survey_df["Survey Answer"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    data_frames, y, test_size=0.2, random_state=42
)
# X_train = np.array(X_train)
# X_test = np.array(X_test)
# y_train_categorical = pd.cut(
#     y_train, bins=5, labels=["None", "Low", "Medium", "High", "Very High"]
# )

# n_samples_train, n_rows_train, n_columns_train = X_train.shape
# X_train_2d = X_train.reshape(n_samples_train, n_rows_train * n_columns_train)

# n_samples_test, n_rows_test, n_columns_test = X_test.shape
# X_test_2d = X_test.reshape(n_samples_test, n_rows_test * n_columns_test)

# Step 4: Model Selection
# Choose a regression model suitable for your problem
# model = LogisticRegression(max_iter=1000)
model = xgb.XGBRegressor(objective="reg:squarederror")  # for regression

# Step 5: Train Model
model.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
# plt.show()

# Visualize Feature Importance (if applicable)
if hasattr(model, "coef_"):
    feature_importance = model.coef_[0]  # Get feature importance coefficients
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xticks(
        range(len(feature_importance)), range(len(feature_importance))
    )  # Assuming feature names are not available
    plt.xlabel("Feature Index")
    plt.ylabel("Coefficient")
    plt.title("Feature Importance")
    # plt.show()

# Step 6: Make predictions
# Assuming newX contains new user activity data
for i in range(1, 6):
    newX = [pd.read_csv(f"new_user_activity/{i}.csv")]
    # n_samples_test, n_rows_test, n_columns_test = newX.shape
    # newX_2d = newX.reshape(n_samples_test, n_rows_test * n_columns_test)
    # newX_scaled = scaler.transform([newX])
    new_predictions = model.predict(newX)
    print(new_predictions)
