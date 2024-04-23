from datetime import datetime, timedelta
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


# Generate list of dates from February 2, 2024, to April 15, 2024
start_date = datetime(2024, 2, 2)
end_date = datetime(2024, 4, 15)
all_dates = [
    start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
]
one_week = timedelta(weeks=1)

# Initialize an empty list to store data for each user
preprocessed_user_data = []

list_of_csv_files = glob.glob("./user_activity_data/*.csv")
# print(len(list_of_csv_files))

# Read survey data
survey_df = pd.read_csv(
    "survey_data/user_survey_data.csv"
)  # Assuming the survey data is in a file named survey_data.csv

y = survey_df["Survey Answer"]

# Assuming each CSV file name corresponds to the user ID
for csv_file in list_of_csv_files:
    week_num = 1
    current_week = []
    current_week_start = datetime(2024, 2, 2)
    user_id = os.path.splitext(os.path.basename(csv_file))[0]

    # Read activity data for each user
    activity_df = pd.read_csv(csv_file)

    for index, row in activity_df.iterrows():
        date = row.Date
        date = datetime.strptime(date, "%m-%d-%Y")

        current_week.append(row.drop("Date"))

        if (date - current_week_start).days >= 7:
            collapsed_activity_scores = value_collapsing_function(current_week)

            # Append user data as a tuple (activity_data, survey_response) to the list
            preprocessed_user_data.append(
                (user_id, f"Week {week_num}", collapsed_activity_scores)
            )
            current_week = []
            current_week_start += one_week
            week_num += 1

    preprocessed_user_data.append(
        (user_id, f"Week {week_num}", value_collapsing_function(current_week))
    )

preprocessed_user_data = pd.DataFrame(
    preprocessed_user_data, columns=["user_id", "week_num", "activity_data"]
)

# print(len(preprocessed_user_data))
# print(len(survey_df.index))

# print(preprocessed_user_data.head())
# print(survey_df.head())

preprocessed_user_data = preprocessed_user_data.sort_values(by=["user_id", "week_num"])
survey_df = survey_df.sort_values(by=["User ID", "Week"])

# print(len(preprocessed_user_data))
# print(len(survey_df.index))

# print(preprocessed_user_data.head())
# print(survey_df.head())

X = (
    preprocessed_user_data.drop("user_id", axis=1)
    .drop("week_num", axis=1)
    .values.tolist()
)
X = [row[0] for row in X]
y = survey_df.drop("User ID", axis=1).drop("Week", axis=1)

print(len(X[1]))

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
