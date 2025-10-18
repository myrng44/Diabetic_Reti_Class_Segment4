"""
Enhanced Self-Adaptive Northern Goshawk Optimization (SANGO) Algorithm
Extended version with:
- Multi-dimensional search space (hidden_dim1, hidden_dim2, dropout, learning_rate)
- F1-score based fitness function
- Better hyperparameter optimization
"""

import numpy as np
from typing import Callable, Tuple, List, Dict
import logging
from sklearn.metrics import f1_score
import torch
from sango import SANGO


class EnhancedSANGO:
    """
    Enhanced SANGO with extended hyperparameter space.

    Optimizes:
    - hidden_dim1: GRU first layer size
    - hidden_dim2: GRU second layer size
    - dropout_rate: Dropout probability
    - learning_rate: Adam learning rate
    """

    def __init__(
            self,
            fitness_function: Callable,
            dim: int = 4,
            population_size: int = 10,
            max_iterations: int = 50,
            bounds: List[List[float]] = None,
            learning_rate: float = 0.02,
            prey_capture_df: float = 0.4,
            prey_identification_r: float = 0.02,
            verbose: bool = True
    ):
        # Default bounds if none provided
        if bounds is None:
            bounds = [
                [128, 512],  # hidden_dim1
                [64, 256],   # hidden_dim2
                [0.1, 0.5],  # dropout
                [1e-5, 1e-3] # learning rate
            ]

        self.fitness_function = fitness_function
        self.dim = dim
        self.N = population_size
        self.T = max_iterations
        self.lr = learning_rate
        self.df_coef = prey_capture_df
        self.r_coef = prey_identification_r
        self.verbose = verbose

        # Setup bounds as numpy arrays
        self.bounds = bounds
        self.L_bound = np.array([b[0] for b in bounds], dtype=np.float64)
        self.U_bound = np.array([b[1] for b in bounds], dtype=np.float64)

        # Track parameter types
        self.is_log_scale = [False, False, False, True]  # Use log scale for learning rate
        self.discrete_params = [True, True, False, False]  # First two params are discrete

        # Initialize population
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = float('inf')

    def _transform_parameters(self, x, inverse=False):
        """Transform parameters to/from optimization space"""
        x = np.array(x, dtype=np.float64)
        result = np.zeros_like(x)

        for i in range(self.dim):
            if self.is_log_scale[i]:
                if inverse:
                    result[i] = np.exp(x[i])
                else:
                    result[i] = np.log(np.maximum(x[i], 1e-10))
            else:
                result[i] = x[i]

        return result

    def _clip_and_round(self, x):
        """Clip to bounds and round discrete parameters"""
        x = np.clip(x, self.L_bound, self.U_bound)

        # Round discrete parameters
        for i in range(self.dim):
            if self.discrete_params[i]:
                x[i] = np.round(x[i])

        return x

    def initialize_population(self):
        """Initialize population with proper parameter handling"""
        self.population = np.zeros((self.N, self.dim))
        self.fitness = np.full(self.N, float('inf'))

        for i in range(self.N):
            # Initialize each parameter within its bounds
            for j in range(self.dim):
                if self.is_log_scale[j]:
                    # Log-uniform sampling for learning rate
                    log_min = np.log(self.L_bound[j])
                    log_max = np.log(self.U_bound[j])
                    self.population[i, j] = np.exp(log_min + np.random.rand() * (log_max - log_min))
                else:
                    # Uniform sampling for other parameters
                    self.population[i, j] = self.L_bound[j] + np.random.rand() * (self.U_bound[j] - self.L_bound[j])

            # Clip and round
            self.population[i] = self._clip_and_round(self.population[i])

            # Evaluate fitness
            try:
                self.fitness[i] = self.fitness_function(self.population[i])
            except Exception as e:
                print(f"Error evaluating initial individual {i}: {e}")
                self.fitness[i] = float('inf')

        # Update best solution
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] != float('inf'):
            self.best_individual = self.population[best_idx].copy()
            self.best_fitness = self.fitness[best_idx]

    def run(self):
        """Run optimization with improved stability"""
        self.initialize_population()

        for t in range(self.T):
            try:
                # Transform to optimization space
                transformed_pop = self._transform_parameters(self.population)

                # Prey identification phase
                new_pop = transformed_pop.copy()
                for i in range(self.N):
                    # Select random prey
                    prey_idx = np.random.choice([j for j in range(self.N) if j != i])
                    prey = transformed_pop[prey_idx]

                    # Update position
                    r = np.random.rand(self.dim)
                    if self.fitness[prey_idx] < self.fitness[i]:
                        new_pop[i] = transformed_pop[i] + r * (prey - transformed_pop[i])
                    else:
                        new_pop[i] = transformed_pop[i] + r * (transformed_pop[i] - prey)

                # Transform back and evaluate
                candidate_pop = self._transform_parameters(new_pop, inverse=True)
                candidate_pop = self._clip_and_round(candidate_pop)

                # Evaluate new positions
                for i in range(self.N):
                    try:
                        new_fitness = self.fitness_function(candidate_pop[i])
                        if new_fitness < self.fitness[i]:
                            self.population[i] = candidate_pop[i]
                            self.fitness[i] = new_fitness
                    except Exception as e:
                        print(f"Error in prey identification phase: {e}")
                        continue

                # Prey capture phase
                R = self.r_coef * (1 - t/self.T)
                DF = self.df_coef * (2 * np.random.rand() - 1) * np.exp(-(t/self.T)**2)

                transformed_pop = self._transform_parameters(self.population)
                new_pop = transformed_pop.copy()

                for i in range(self.N):
                    r = 2 * np.random.rand(self.dim) - 1
                    new_pop[i] = transformed_pop[i] + R * r * transformed_pop[i] * DF

                # Transform back and evaluate
                candidate_pop = self._transform_parameters(new_pop, inverse=True)
                candidate_pop = self._clip_and_round(candidate_pop)

                # Evaluate new positions
                for i in range(self.N):
                    try:
                        new_fitness = self.fitness_function(candidate_pop[i])
                        if new_fitness < self.fitness[i]:
                            self.population[i] = candidate_pop[i]
                            self.fitness[i] = new_fitness
                    except Exception as e:
                        print(f"Error in prey capture phase: {e}")
                        continue

                # Update best solution
                best_idx = np.argmin(self.fitness)
                if self.fitness[best_idx] < self.best_fitness:
                    self.best_individual = self.population[best_idx].copy()
                    self.best_fitness = self.fitness[best_idx]

                    if self.verbose:
                        print(f"Iteration {t+1}/{self.T}: Best fitness = {self.best_fitness}")

            except Exception as e:
                print(f"Error in iteration {t+1}: {e}")
                continue

        return self.best_individual
