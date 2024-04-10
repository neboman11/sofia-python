import glob
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def column_collapsing_function(column):
    # return np.average(column)
    return np.median(column)


def value_collapsing_function(dataframe):
    # collapsed_activity_scores = []
    # for column in dataframe:
    #     collapsed_activity_scores.append(column_collapsing_function(dataframe[column]))

    # return collapsed_activity_scores

    # flattened_data = np.reshape(dataframe, -1)
    # return flattened_data

    pca = PCA(n_components=1)
    pcs = pca.fit_transform(dataframe)

    one_dimensional_data = pcs.ravel()
    return one_dimensional_data


# Initialize an empty list to store data for each user
X = []

list_of_csv_files = glob.glob("./user_activity_data/*.csv")
print(len(list_of_csv_files))

# Read survey data
survey_df = pd.read_csv(
    "survey_data/user_survey_data.csv"
)  # Assuming the survey data is in a file named survey_data.csv

y = survey_df["Survey Answer"]

# Assuming each CSV file name corresponds to the user ID
for csv_file in list_of_csv_files:
    user_id = os.path.splitext(os.path.basename(csv_file))[0]

    # Read activity data for each user
    activity_df = pd.read_csv(csv_file)

    collapsed_activity_scores = value_collapsing_function(activity_df)

    # Append user data as a tuple (activity_data, survey_response) to the list
    X.append(collapsed_activity_scores)

model = LogisticRegression(multi_class="multinomial", solver="lbfgs")

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = preprocessing.StandardScaler().fit(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the model
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

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
    plt.show()

# Step 6: Make predictions
# Assuming newX contains new user activity data
for i in range(1, 6):
    newX = pd.read_csv(f"new_user_activity/{i}.csv")
    collapsed_activity_scores = value_collapsing_function(newX)
    newX_scaled = scaler.transform([collapsed_activity_scores])
    new_predictions = model.predict(newX_scaled)
    print(new_predictions)
