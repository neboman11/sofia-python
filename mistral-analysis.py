# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load user activity data
user_activity_data = {}
for user_id in set(activity_data["user_id"]):
    user_data = pd.read_csv(f"activity_data_{user_id}.csv")
    user_activity_data[user_id] = user_data
# Merge user activity data
activity_data_all = pd.concat(user_activity_data.values(), ignore_index=True)
# Merge the activity data and survey data
merged_data = pd.merge(activity_data_all, survey_data, on="user_id")
# Select relevant features
X = merged_data[["feature1", "feature2"]]  # replace with actual feature names
y = merged_data["survey_response"]
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Create the model
model = LinearRegression()
# Fit the model
model.fit(X_train, y_train)
# Evaluate the model
print("R-squared:", model.score(X_test, y_test))
print("Mean Squared Error (test):", np.mean(np.square(model.predict(X_test) - y_test)))
# Predict survey responses for new data
new_user_data = pd.read_csv("new_user_activity_data.csv")
user_activity_data_new = pd.DataFrame(
    new_user_data.values
)  # assumes the new data is in a NumPy array
predictions = model.predict(pd.concat([user_activity_data_new, X.iloc[0]], axis=1))
print("Predicted survey response:", predictions)
