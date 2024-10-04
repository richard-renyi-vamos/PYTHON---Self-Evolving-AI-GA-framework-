import random
import numpy as np

# Define the fitness function (we'll optimize for the maximum sum of a binary array)
def fitness_function(solution):
    return sum(solution)

# Generate the initial population
def generate_population(size, length):
    return [np.random.randint(2, size=length).tolist() for _ in range(size)]

# Select individuals based on their fitness scores (roulette wheel selection)
def select_individuals(population, fitness_scores, num_select):
    selected = random.choices(population, weights=fitness_scores, k=num_select)
    return selected

# Perform crossover between two individuals
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutate individuals (randomly flip one bit)
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

# Main Genetic Algorithm loop
def genetic_algorithm(population_size, chromosome_length, generations, mutation_rate):
    # Generate the initial population
    population = generate_population(population_size, chromosome_length)
    
    for generation in range(generations):
        # Evaluate the fitness of each individual
        fitness_scores = [fitness_function(ind) for ind in population]
        print(f"Generation {generation}, Best fitness: {max(fitness_scores)}")
        
        # Selection
        selected_individuals = select_individuals(population, fitness_scores, population_size // 2)
        
        # Crossover
        new_population = []
