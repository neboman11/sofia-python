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
    pca.fit(dataframe)
    pcs = pca.transform(dataframe)

    one_dimensional_data = pcs.ravel()
    print(dataframe)
    print(pcs)
    print(pca.explained_variance_ratio_)
    print(one_dimensional_data)
    exit(1)
    return one_dimensional_data


# Generate list of dates from February 2, 2024, to April 15, 2024
start_date = datetime(2024, 2, 2)
end_date = datetime(2024, 4, 15)
all_dates = [
    start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
]
one_week = timedelta(weeks=1)

list_of_csv_files = glob.glob("./user_activity_data/*.csv")
# print(len(list_of_csv_files))

# Read survey data
survey_df = pd.read_csv(
    "survey_data/user_survey_data.csv"
)  # Assuming the survey data is in a file named survey_data.csv

unprocessed_activity_data = []

# Assuming each CSV file name corresponds to the user ID
for csv_file in list_of_csv_files:
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
            week_num = (date - current_week_start).days // 7
            # collapsed_activity_scores = value_collapsing_function(current_week)

            # if len(collapsed_activity_scores) > max_X_width:
            #     max_X_width = len(collapsed_activity_scores)

            # Append user data as a tuple (activity_data, survey_response) to the list
            unprocessed_activity_data.append(
                (user_id, f"Week {week_num}", current_week)
            )
            current_week = []
            current_week_start += one_week

    if len(current_week) > 0:
        unprocessed_activity_data.append((user_id, f"Week {week_num}", current_week))

raw_concatenated_data = [data for row in unprocessed_activity_data for data in row[2]]
# print(raw_concatenated_data[0])
# print(raw_concatenated_data)
pca = PCA(n_components=1)
pca.fit(raw_concatenated_data)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.components_)

max_X_width = 0
raw_preprocessed_user_data = []
for week in unprocessed_activity_data:
    transformed_week = pca.transform(week[2]).ravel()

    if len(transformed_week) > max_X_width:
        max_X_width = len(transformed_week)

    raw_preprocessed_user_data.append((week[0], week[1], transformed_week))

preprocessed_user_data = []
for row in raw_preprocessed_user_data:
    preprocessed_user_data.append(
        (row[0], row[1], np.pad(row[2], (0, max_X_width - len(row[2])), "constant"))
    )

preprocessed_user_data = pd.DataFrame(
    preprocessed_user_data, columns=["User ID", "Week", "activity_data"]
)

# print(len(preprocessed_user_data))
# print(len(survey_df.index))

# print(preprocessed_user_data.head())
# print(survey_df.head())

# Merge the two dataframes
merged_df = pd.merge(
    preprocessed_user_data,
    survey_df,
    how="inner",
    on=["User ID", "Week"],
)
print(merged_df.head())

# preprocessed_user_data = preprocessed_user_data.sort_values(by=["user_id", "week_num"])
# survey_df = survey_df.sort_values(by=["User ID", "Week"])

# print(len(preprocessed_user_data))
# print(len(survey_df.index))

# print(preprocessed_user_data.head())
# print(survey_df.head())

X = merged_df["activity_data"].values.tolist()
# X = [row[0] for row in X]
y = merged_df["Survey Answer"].values.tolist()
# y = [row[0] for row in y]

# [print(len(row)) for row in X]
# print(X[0])
# print(y)

model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)

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
    x_ticks = [i for i in range(10)]
    feature_names = [f"Column {i+1}" for i in range(10)]
    print(np.transpose([model.coef_[0]]))
    coefficients_original = pca.inverse_transform(np.transpose([model.coef_[0]]))
    feature_importance = coefficients_original[0]  # Get feature importance coefficients
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xticks(
        range(len(feature_importance)), range(len(feature_importance))
    )  # Assuming feature names are not available
    plt.xlabel("Feature Index")
    plt.ylabel("Coefficient")
    plt.title("Feature Importance")
    plt.xticks(x_ticks, labels=feature_names, rotation=45, ha="right")
    plt.show()

# Step 6: Make predictions
# Assuming newX contains new user activity data
for i in range(1, 6):
    newX = pd.read_csv(f"new_user_activity/{i}.csv")
    collapsed_activity_scores = value_collapsing_function(newX)
    newX_scaled = scaler.transform([collapsed_activity_scores])
    new_predictions = model.predict(newX_scaled)
    print(new_predictions)
