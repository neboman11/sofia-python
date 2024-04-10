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


def main():
    num_rows = 100
    num_columns = 10

    survey_response = random.randint(1, 5)
    data = [generate_random_row(num_columns, survey_response) for _ in range(num_rows)]

    with open(f"new_user_activity/{survey_response}.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Column 1", "Column 2", "Column 3", "Column 4", "Column 5"])
        writer.writerows(data)


if __name__ == "__main__":
    main()
