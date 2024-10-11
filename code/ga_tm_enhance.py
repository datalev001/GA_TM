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

# Function to calculate fitness using weighted mean squared error
def calculate_fitness(actual, predicted, weights):
    error = actual - predicted
    weighted_error = weights * (error ** 2)
    wmse = weighted_error.sum()
    fitness = 1 / (np.sqrt(wmse) + 1e-6)  # Add small value to avoid division by zero
    return fitness

# Improved Genetic Algorithm function for time series forecasting
def genetic_algorithm_improved(train_data, generations=150, population_size=100, mutation_rate=0.03, 
                               forecast_window=5, lag=5, alpha=0.9, elitism=True, elite_size=3):
    # Initialize population with random coefficients for lags
    population = [np.random.uniform(-1, 1, lag) for _ in range(population_size)]
    
    train_values = train_data.values
    N = len(train_values)
    
    # Define recency weights using exponential decay
    recency_weights = np.exp(-alpha * np.linspace(0, 1, N))
    recency_weights /= recency_weights.sum()  # Normalize weights
    
    forecasted_values = []
    history = list(train_values)
    
    for f in range(forecast_window):
        # Prepare data for current forecast
        current_train = np.array(history)
        current_N = len(current_train)
        
        # Define weights for current training data
        weights = np.exp(-alpha * np.linspace(0, 1, current_N))
        weights /= weights.sum()
        
        # Evolve population
        for generation in range(generations):
            fitness_scores = []
            for chromosome in population:
                # Generate in-sample predictions
                predictions = []
                for t in range(lag, current_N):
                    window = current_train[t - lag:t]
                    pred = np.dot(chromosome, window[::-1])
                    predictions.append(pred)
                
                actual = current_train[lag:]
                fitness = calculate_fitness(actual, np.array(predictions), weights[lag:])
                fitness_scores.append(fitness)
            
            fitness_scores = np.array(fitness_scores)
            
            # Handle case where all fitness scores are zero to avoid division by zero
            if fitness_scores.sum() == 0:
                fitness_probs = np.ones(population_size) / population_size
            else:
                # Selection: Fitness-proportionate selection (roulette wheel)
                fitness_probs = fitness_scores / fitness_scores.sum()
            
            # Select parents based on fitness probabilities
            selected_indices = np.random.choice(range(population_size), size=population_size, p=fitness_probs)
            parents = [population[i] for i in selected_indices]
            
            # Crossover and Mutation to produce new population
            new_population = []
            if elitism:
                # Elitism: Carry over the top elite_size chromosomes to the new population
                elite_indices = fitness_scores.argsort()[-elite_size:]
                elites = [population[i] for i in elite_indices]
                new_population.extend(elites)
            
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(len(parents), 2, replace=False)
                
                # Single-point crossover
                crossover_point = np.random.randint(1, lag)
                offspring1 = np.concatenate([parents[parent1][:crossover_point], parents[parent2][crossover_point:]])
                offspring2 = np.concatenate([parents[parent2][:crossover_point], parents[parent1][crossover_point:]])
                
                # Mutation
                offspring1 += mutation_rate * np.random.randn(lag)
                offspring2 += mutation_rate * np.random.randn(lag)
                
                # Ensure values stay within a reasonable range
                offspring1 = np.clip(offspring1, -5, 5)
                offspring2 = np.clip(offspring2, -5, 5)
                
                new_population.extend([offspring1, offspring2])
            
            population = new_population[:population_size]
        
        # After generations, select the best chromosome
        fitness_scores = []
        for chromosome in population:
            predictions = []
            for t in range(lag, current_N):
                window = current_train[t - lag:t]
                pred = np.dot(chromosome, window[::-1])
                predictions.append(pred)
            
            actual = current_train[lag:]
            fitness = calculate_fitness(actual, np.array(predictions), weights[lag:])
            fitness_scores.append(fitness)
        
        fitness_scores = np.array(fitness_scores)
        best_index = fitness_scores.argmax()
        best_chromosome = population[best_index]
        
        # Forecast the next value
        last_lag = np.array(history[-lag:])[::-1]
        next_pred = np.dot(best_chromosome, last_lag)
        forecasted_values.append(next_pred)
        history.append(next_pred)
    
    return forecasted_values

# Function to forecast the next N months using improved GA
def forecast_next_n_months_improved(train_data, test_data, n_months=5, 
                                    generations=150, population_size=100, mutation_rate=0.03, 
                                    forecast_window=5, lag=5, alpha=0.9, elitism=True, elite_size=3):
    # Forecast the next N months using improved GA
    forecasted_values = genetic_algorithm_improved(train_data, generations, population_size, 
                                                   mutation_rate, forecast_window, lag, 
                                                   alpha, elitism, elite_size)
    
    # Actual values for comparison
    actual_values = test_data[:n_months].values
    
    # Calculate performance metrics (MAPE, RMSE)
    mape = mean_absolute_percentage_error(actual_values, forecasted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, forecasted_values))
    
    # Display results
    print(f"Forecasted Values for next {n_months} months: {forecasted_values}")
    print(f"Actual Values of Last {n_months} Months: {actual_values}")
    print(f"MAPE: {mape}")
    print(f"RMSE: {rmse}")
    
    return forecasted_values, mape, rmse


print("Forecasting 1 months with Improved GA:")
forecasted_values, mape, rmse = forecast_next_n_months_improved(
    train_data=series[:-1], 
    test_data=series[-1:], 
    n_months=1, 
    generations=280, 
    population_size=90, 
    mutation_rate=0.03, 
    forecast_window=1, 
    lag=3, 
    alpha=0.8, 
    elitism=True, 
    elite_size=3
)
'''
Forecasting 1 months with Improved GA:
Forecasted Values for next 1 months: [123.0827714213442]
Actual Values of Last 1 Months: [129.4048]
MAPE: 0.04885466828630615
RMSE: 6.3220285786557895
'''

# Example usage
print("Forecasting 3 months with Improved GA:")
forecasted_values, mape, rmse = forecast_next_n_months_improved(
    train_data=series[:-3], 
    test_data=series[-3:], 
    n_months=3, 
    generations=400, 
    population_size=120, 
    mutation_rate=0.03, 
    forecast_window=3, 
    lag=6, 
    alpha=0.9, 
    elitism=True, 
    elite_size=3
)
'''
Forecasting 3 months with Improved GA:
Forecasted Values for next 3 months: [95.92765046069349, 104.49102699564476, 110.55805978928841]
Actual Values of Last 3 Months: [ 97.3359 114.7212 129.4048]
MAPE: 0.0830946303030689
RMSE: 12.407514161910767
'''

print("\nForecasting 5 months with Improved GA:")
forecasted_values, mape, rmse = forecast_next_n_months_improved(
    train_data=series[:-5], 
    test_data=series[-5:], 
    n_months=5, 
    generations=300, 
    population_size=150, 
    mutation_rate=0.03, 
    forecast_window=5, 
    lag=4, 
    alpha=1, 
    elitism=True, 
    elite_size=3
)

'''
Forecasting 5 months with Improved GA:
Forecasted Values for next 5 months: [99.4725344827074, 92.52487567604749, 95.76790713274853, 104.28661424133446, 109.53688001531333]
Actual Values of Last 5 Months: [ 98.6154  93.6137  97.3359 114.7212 129.4048]
MAPE: 0.056184190710886206
RMSE: 10.079619477286686
'''