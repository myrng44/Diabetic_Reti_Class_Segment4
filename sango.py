"""
Self-Adaptive Northern Goshawk Optimization (SANGO) Algorithm
Based on the paper: "A multi model deep net with an explainable AI based framework
for diabetic retinopathy segmentation and classification"

SANGO is used to optimize GRU hyperparameters (number of units in layers).
"""

import numpy as np
from typing import Callable, Tuple, List
import logging


class SANGO:
    """
    Self-Adaptive Northern Goshawk Optimization Algorithm.

    Used for hyperparameter optimization of GRU network.
    Optimizes discrete parameters like number of neurons in hidden layers.
    """

    def __init__(
            self,
            fitness_function: Callable,
            dim: int = 2,
            population_size: int = 10,
            max_iterations: int = 100,
            lower_bound: List[int] = [16, 16],
            upper_bound: List[int] = [128, 128],
            learning_rate: float = 0.02,
            prey_capture_df: float = 0.4,
            prey_identification_r: float = 0.02,
            verbose: bool = True
    ):
        """
        Initialize SANGO optimizer.

        Args:
            fitness_function: Function to minimize (returns fitness score)
            dim: Dimension of search space (number of hyperparameters)
            population_size: Number of individuals in population
            max_iterations: Maximum iterations
            lower_bound: Lower bounds for each dimension
            upper_bound: Upper bounds for each dimension
            learning_rate: Learning rate for updates
            prey_capture_df: Dynamic factor coefficient for prey capture
            prey_identification_r: Factor for prey identification phase
            verbose: Print progress
        """
        self.fitness_function = fitness_function
        self.dim = dim
        self.N = population_size
        self.T = max_iterations
        self.L_bound = np.array(lower_bound)
        self.U_bound = np.array(upper_bound)
        self.lr = learning_rate
        self.df_coef = prey_capture_df
        self.r_coef = prey_identification_r
        self.verbose = verbose

        # Initialize population
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = float('inf')
        self.convergence_curve = []

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def initialize_population(self):
        """Initialize population within bounds."""
        self.population = np.zeros((self.N, self.dim))

        for i in range(self.N):
            for j in range(self.dim):
                # Random initialization between bounds
                self.population[i, j] = self.L_bound[j] + np.random.rand() * (
                        self.U_bound[j] - self.L_bound[j]
                )

        # Round to integers for discrete hyperparameters
        self.population = np.round(self.population).astype(int)

        # Ensure within bounds
        self.population = np.clip(self.population, self.L_bound, self.U_bound)

        # Evaluate initial fitness
        self.fitness = np.array([self.fitness_function(ind) for ind in self.population])

        # Track best
        best_idx = np.argmin(self.fitness)
        self.best_individual = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

        if self.verbose:
            self.logger.info(f"Initial best fitness: {self.best_fitness:.6f}")
            self.logger.info(f"Initial best individual: {self.best_individual}")

    def prey_identification(self, t: int):
        """
        Phase 1: Prey Identification.
        Northern goshawk identifies and approaches prey.

        Args:
            t: Current iteration
        """
        new_population = np.zeros_like(self.population)

        for i in range(self.N):
            # Randomly select prey (another individual)
            prey_idx = np.random.choice([j for j in range(self.N) if j != i])
            prey = self.population[prey_idx]

            # Random vector r in [0, 1]
            r = np.random.rand(self.dim)

            # Random vector I containing 1 or 2
            I = np.random.choice([1, 2], size=self.dim)

            # Update position based on prey fitness comparison
            if self.fitness[prey_idx] < self.fitness[i]:
                # Prey is better, move towards it
                new_population[i] = self.population[i] + r * (prey - I * self.population[i])
            else:
                # Current position is better, move away from prey
                new_population[i] = self.population[i] + r * (self.population[i] - prey)

            # Round to integers and clip to bounds
            new_population[i] = np.round(new_population[i]).astype(int)
            new_population[i] = np.clip(new_population[i], self.L_bound, self.U_bound)

        # Evaluate new positions
        new_fitness = np.array([self.fitness_function(ind) for ind in new_population])

        # Update if improved
        for i in range(self.N):
            if new_fitness[i] < self.fitness[i]:
                self.population[i] = new_population[i]
                self.fitness[i] = new_fitness[i]

    def prey_capture(self, t: int):
        """
        Phase 2: Prey Capture with Dynamic Factor.
        Northern goshawk chases and captures prey.

        Args:
            t: Current iteration
        """
        new_population = np.zeros_like(self.population)

        # Calculate radius R (decreases over time)
        R = self.r_coef * (1 - t / self.T)

        # Calculate Dynamic Factor (DF) - key innovation of SANGO
        # DF = 0.4 * (2 * rand - 1) * exp(-(t/T)^2)
        DF = self.df_coef * (2 * np.random.rand() - 1) * np.exp(-((t / self.T) ** 2))

        for i in range(self.N):
            # Random vector r in [-1, 1]
            r = 2 * np.random.rand(self.dim) - 1

            # Update position within radius R with dynamic factor
            new_population[i] = self.population[i] + R * r * self.population[i] * DF

            # Round to integers and clip to bounds
            new_population[i] = np.round(new_population[i]).astype(int)
            new_population[i] = np.clip(new_population[i], self.L_bound, self.U_bound)

        # Evaluate new positions
        new_fitness = np.array([self.fitness_function(ind) for ind in new_population])

        # Update if improved
        for i in range(self.N):
            if new_fitness[i] < self.fitness[i]:
                self.population[i] = new_population[i]
                self.fitness[i] = new_fitness[i]

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run SANGO optimization.

        Returns:
            best_individual: Best hyperparameters found
            best_fitness: Best fitness value
            convergence_curve: Fitness history
        """
        # Initialize
        self.initialize_population()
        self.convergence_curve = [self.best_fitness]

        # Main optimization loop
        for t in range(self.T):
            if self.verbose and t % 10 == 0:
                self.logger.info(f"Iteration {t}/{self.T}, Best Fitness: {self.best_fitness:.6f}")

            # Phase 1: Prey Identification
            self.prey_identification(t)

            # Phase 2: Prey Capture
            self.prey_capture(t)

            # Update global best
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_individual = self.population[best_idx].copy()
                self.best_fitness = self.fitness[best_idx]

                if self.verbose:
                    self.logger.info(f"New best at iteration {t}: {self.best_fitness:.6f}")
                    self.logger.info(f"Best individual: {self.best_individual}")

            # Record convergence
            self.convergence_curve.append(self.best_fitness)

        if self.verbose:
            self.logger.info(f"\nOptimization complete!")
            self.logger.info(f"Best fitness: {self.best_fitness:.6f}")
            self.logger.info(f"Best hyperparameters: {self.best_individual}")

        return self.best_individual, self.best_fitness, self.convergence_curve


def optimize_gru_hyperparameters(
        train_function: Callable,
        val_loader,
        device,
        population_size: int = 10,
        max_iterations: int = 50,
        lower_bounds: List[int] = [16, 16],
        upper_bounds: List[int] = [128, 128]
) -> Tuple[int, int]:
    """
    Use SANGO to find optimal GRU hyperparameters.

    Args:
        train_function: Function that trains model and returns validation loss
        val_loader: Validation data loader
        device: Device to train on
        population_size: SANGO population size
        max_iterations: SANGO iterations
        lower_bounds: Lower bounds for [hidden_dim, num_layers]
        upper_bounds: Upper bounds for [hidden_dim, num_layers]

    Returns:
        optimal_hidden_dim: Best hidden dimension
        optimal_num_layers: Best number of layers
    """

    def fitness_function(hyperparams):
        """
        Fitness function for SANGO.
        Lower is better (validation loss).
        """
        hidden_dim = int(hyperparams[0])
        num_layers = int(hyperparams[1])

        try:
            # Train model with these hyperparameters
            val_loss = train_function(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                val_loader=val_loader,
                device=device
            )

            return val_loss

        except Exception as e:
            # Return large penalty if training fails
            logging.error(f"Training failed for hyperparams {hyperparams}: {e}")
            return 1e6

    # Initialize SANGO
    sango = SANGO(
        fitness_function=fitness_function,
        dim=2,  # [hidden_dim, num_layers]
        population_size=population_size,
        max_iterations=max_iterations,
        lower_bound=lower_bounds,
        upper_bound=upper_bounds,
        verbose=True
    )

    # Run optimization
    best_hyperparams, best_fitness, convergence = sango.optimize()

    optimal_hidden_dim = int(best_hyperparams[0])
    optimal_num_layers = int(best_hyperparams[1])

    return optimal_hidden_dim, optimal_num_layers


# Example usage
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)


    # Test with simple quadratic function
    def test_fitness(x):
        """Simple quadratic function for testing."""
        return np.sum((x - np.array([64, 2])) ** 2)


    sango = SANGO(
        fitness_function=test_fitness,
        dim=2,
        population_size=10,
        max_iterations=30,
        lower_bound=[16, 1],
        upper_bound=[128, 4],
        verbose=True
    )

    best_x, best_f, curve = sango.optimize()

    print(f"\nOptimal solution: {best_x}")
    print(f"Optimal fitness: {best_f}")
    print(f"Expected: [64, 2]")