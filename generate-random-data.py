import csv
from datetime import datetime, timedelta
import random
import uuid


# Function to generate random integer with weighted probabilities
def generate_random_integer(weight=1):
    choices = list(range(101))
    # Adjust weights based on input scale (1 to 5), with more skew towards higher numbers
    weights = [1 + (weight - 1) * (i**2 / 10000) for i in range(101)]
    return random.choices(choices, weights=weights)[0]


# Function to generate a row of random integers
def generate_random_row(num_columns, weight=1):
    return [generate_random_integer(weight) for _ in range(num_columns)]


# Function to generate random weights for each week
def generate_random_weights(num_weeks):
    return [random.randint(1, 5) for _ in range(num_weeks)]


def main():
    num_columns = 10
    num_files = 100
    num_days_in_week = 7

    # Generate list of dates from February 2, 2024, to April 15, 2024
    start_date = datetime(2024, 2, 2)
    end_date = datetime(2024, 4, 15)
    all_dates = [
        start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
    ]

    num_weeks = (datetime(2024, 4, 15) - datetime(2024, 2, 2)).days // 7 + 1

    user_survey_responses = []

    for _ in range(num_files):
        user_uuid = uuid.uuid4()
        weights = generate_random_weights(num_weeks)

        # Generate a list of booleans for deciding which dates to keep and which to omit
        selected_dates = [random.random() < 0.8 for _ in range(len(all_dates))]

        # Group dates into weeks
        weeks = [
            all_dates[i : i + num_days_in_week]
            for i in range(0, len(all_dates), num_days_in_week)
        ]

        data = []
        for week_num, week in enumerate(weeks):
            for date_num, date in enumerate(week):
                if selected_dates[date_num + week_num * num_days_in_week]:
                    data.append(
                        [date.strftime("%m-%d-%Y")]
                        + generate_random_row(num_columns, weights[week_num])
                    )

        survey_data_columns = ["Date"]
        [survey_data_columns.append(f"Column {i + 1}") for i in range(num_columns)]

        with open(f"user_activity_data/{str(user_uuid)}.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(survey_data_columns)
            writer.writerows(data)

        for index, survey_response in enumerate(weights):
            user_survey_responses.append(
                (user_uuid, f"Week {index + 1}", survey_response)
            )

    with open(f"survey_data/user_survey_data.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["User ID", "Week", "Survey Answer"])
        writer.writerows(user_survey_responses)


if __name__ == "__main__":
    main()
