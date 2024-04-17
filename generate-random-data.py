import csv
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


# Function to generate random number of rows between 50 and 100
def generate_random_num_rows():
    return random.randint(50, 100)


def main():
    num_columns = 10
    num_files = 50

    user_survey_responses = []

    for _ in range(num_files):
        user_uuid = uuid.uuid4()
        survey_response = random.randint(1, 5)
        data = [
            generate_random_row(num_columns, survey_response)
            for _ in range(generate_random_num_rows())
        ]

        with open(f"user_activity_data/{str(user_uuid)}.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["Column 1", "Column 2", "Column 3", "Column 4", "Column 5"]
            )
            writer.writerows(data)

        user_survey_responses.append((user_uuid, survey_response))

    with open(f"survey_data/user_survey_data.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["User ID", "Survey Answer"])
        writer.writerows(user_survey_responses)


if __name__ == "__main__":
    main()
