import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Load the CSV file
file_path = 'Electric_Production_tm.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['DATE'] = pd.to_datetime(data['DATE'])  # Convert DATE to datetime format
data = data.set_index('DATE')  # Set DATE as an index

# Keep only the relevant series (IPG2211A2N) and split the data into train and test sets
series = data['IPG2211A2N']

# Function to calculate fitness (using simple linear trend assumption)
def calculate_fitness(candidate, expected_value):
    return 1 / np.abs(candidate - expected_value)

# Genetic algorithm function for forecasting the next N months
def genetic_algorithm(train_data, generations=100, mutation_rate=0.3, n_months=3):
    population = train_data[-(n_months + 4):].tolist()  # Initialize population with last n_months + 5 values
    expected_values = [train_data[-1] + i for i in range(1, n_months + 1)]  # Expecting linear increment for N months

    forecasted_values = []

    for i in range(n_months):
        expected_value = expected_values[i]
        for generation in range(generations):
            # Step 2: Calculate fitness for each candidate in population
            fitness_scores = [calculate_fitness(x, expected_value) for x in population]

            # Step 3: Selection - pick two candidates with the highest fitness
            selected_indices = np.argsort(fitness_scores)[-2:]  # Select top 2 candidates
            parents = [population[i] for i in selected_indices]

            # Step 4: Crossover - create offspring by averaging parents
            offspring = (parents[0] + parents[1]) / 2

            # Step 5: Mutation - add small random changes
            mutated_offspring = offspring + mutation_rate * np.random.randn()

            # Update population with the new offspring
            population.append(mutated_offspring)
            population.pop(0)  # Remove the oldest candidate

        # Store forecasted value for this month
        forecasted_values.append(population[-1])

    return forecasted_values

# Function to forecast the next N months using traditional GA
def forecast_next_n_months(train_data, test_data, n_months=3):
    # Forecast the next N months using genetic algorithm
    forecasted_values = genetic_algorithm(train_data, n_months=n_months)

    # Calculate the average of the actual last N months (holdout set)
    actual_avg_last_n_months = test_data[-n_months:].mean()

    # Calculate performance metrics (MAPE, RMSE)
    mape = mean_absolute_percentage_error([actual_avg_last_n_months], [np.mean(forecasted_values)])
    rmse = np.sqrt(mean_squared_error([actual_avg_last_n_months], [np.mean(forecasted_values)]))

    # Display results
    print(f"Forecasted Values for next {n_months} months: {forecasted_values}")
    print(f"Actual Average of Last {n_months} Months: {actual_avg_last_n_months}")
    print(f"MAPE: {mape}")
    print(f"RMSE: {rmse}")

    return forecasted_values, mape, rmse

###traditional genetics for TM forecasting
# mape  0.09843
forecasted_values, mape, rmse = forecast_next_n_months(series[:-1], series[-1:], n_months=1)
# Forecast the next N months (e.g., 3 months or 6 months)
# mape  0.1594917045498796
forecasted_values, mape, rmse = forecast_next_n_months(series[:-3], series[-3:], n_months=3)
# mape  0.049440
forecasted_values, mape, rmse = forecast_next_n_months(series[:-5], series[-5:], n_months=5)


