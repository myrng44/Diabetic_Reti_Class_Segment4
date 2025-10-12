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
            bounds: Dict[str, Tuple[float, float]] = None,
            learning_rate: float = 0.02,
            prey_capture_df: float = 0.4,
            prey_identification_r: float = 0.02,
            verbose: bool = True
    ):
        """
        Initialize Enhanced SANGO.

        Args:
            fitness_function: Function to minimize (1 - F1_score)
            dim: Dimension of search space
            population_size: Number of individuals
            max_iterations: Maximum iterations
            bounds: Dictionary with parameter bounds
                    {'hidden_dim1': (16, 256), 'hidden_dim2': (16, 256),
                     'dropout': (0.1, 0.5), 'lr': (1e-5, 1e-3)}
            learning_rate: Algorithm learning rate
            prey_capture_df: Dynamic factor coefficient
            prey_identification_r: Identification factor
            verbose: Print progress
        """
        self.fitness_function = fitness_function
        self.dim = dim
        self.N = population_size
        self.T = max_iterations
        self.lr = learning_rate
        self.df_coef = prey_capture_df
        self.r_coef = prey_identification_r
        self.verbose = verbose

        # Setup bounds
        if bounds is None:
            bounds = {
                'hidden_dim1': (32, 256),
                'hidden_dim2': (32, 256),
                'dropout': (0.1, 0.5),
                'lr': (1e-5, 1e-3)
            }

        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.L_bound = np.array([bounds[k][0] for k in self.param_names])
        self.U_bound = np.array([bounds[k][1] for k in self.param_names])

        # Track which params are discrete (int) vs continuous (float)
        self.discrete_params = ['hidden_dim1', 'hidden_dim2']
        self.discrete_indices = [i for i, name in enumerate(self.param_names)
                                 if name in self.discrete_params]

        # Initialize
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = float('inf')
        self.convergence_curve = []
        self.fitness_history = []

        self.logger = logging.getLogger(__name__)

    def initialize_population(self):
        """Initialize population with proper bounds for each parameter."""
        self.population = np.zeros((self.N, self.dim))

        for i in range(self.N):
            for j in range(self.dim):
                param_name = self.param_names[j]

                if param_name in ['lr']:
                    # Log-uniform for learning rate
                    log_min = np.log10(self.L_bound[j])
                    log_max = np.log10(self.U_bound[j])
                    self.population[i, j] = 10 ** (log_min + np.random.rand() * (log_max - log_min))
                else:
                    # Uniform for others
                    self.population[i, j] = self.L_bound[j] + np.random.rand() * (
                            self.U_bound[j] - self.L_bound[j]
                    )

        # Round discrete parameters
        for idx in self.discrete_indices:
            self.population[:, idx] = np.round(self.population[:, idx])

        # Clip to bounds
        self.population = np.clip(self.population, self.L_bound, self.U_bound)

        # Evaluate fitness
        self.fitness = np.array([self.fitness_function(self._decode_individual(ind))
                                 for ind in self.population])

        # Track best
        best_idx = np.argmin(self.fitness)
        self.best_individual = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

        if self.verbose:
            params = self._decode_individual(self.best_individual)
            self.logger.info(f"Initial best fitness: {self.best_fitness:.6f}")
            self.logger.info(f"Initial best params: {params}")

    def _decode_individual(self, individual: np.ndarray) -> Dict:
        """Convert individual array to parameter dictionary."""
        params = {}
        for i, name in enumerate(self.param_names):
            if name in self.discrete_params:
                params[name] = int(individual[i])
            else:
                params[name] = float(individual[i])
        return params

    def _encode_params(self, params: Dict) -> np.ndarray:
        """Convert parameter dictionary to individual array."""
        individual = np.zeros(self.dim)
        for i, name in enumerate(self.param_names):
            individual[i] = params[name]
        return individual

    def prey_identification(self, t: int):
        """Phase 1: Prey Identification with adaptive search."""
        new_population = np.zeros_like(self.population)

        for i in range(self.N):
            # Select random prey
            prey_idx = np.random.choice([j for j in range(self.N) if j != i])
            prey = self.population[prey_idx]

            # Random vectors
            r = np.random.rand(self.dim)
            I = np.random.choice([1, 2], size=self.dim)

            # Update based on fitness comparison
            if self.fitness[prey_idx] < self.fitness[i]:
                new_population[i] = self.population[i] + r * (prey - I * self.population[i])
            else:
                new_population[i] = self.population[i] + r * (self.population[i] - prey)

            # Apply specific handling for different parameter types
            for j, param_name in enumerate(self.param_names):
                if param_name in ['lr']:
                    # Keep learning rate in log space for better exploration
                    new_population[i, j] = np.clip(new_population[i, j],
                                                   self.L_bound[j], self.U_bound[j])
                elif param_name in self.discrete_params:
                    # Round discrete parameters
                    new_population[i, j] = np.round(new_population[i, j])

                # Clip to bounds
                new_population[i, j] = np.clip(new_population[i, j],
                                               self.L_bound[j], self.U_bound[j])

        # Evaluate new positions
        new_fitness = np.array([self.fitness_function(self._decode_individual(ind))
                                for ind in new_population])

        # Update if improved
        for i in range(self.N):
            if new_fitness[i] < self.fitness[i]:
                self.population[i] = new_population[i]
                self.fitness[i] = new_fitness[i]

    def prey_capture(self, t: int):
        """Phase 2: Prey Capture with adaptive dynamic factor."""
        new_population = np.zeros_like(self.population)

        # Adaptive radius (decreases over time)
        R = self.r_coef * (1 - t / self.T)

        # Dynamic Factor with self-adaptive component
        # Increases exploration early, exploitation later
        base_df = self.df_coef * (2 * np.random.rand() - 1) * np.exp(-((t / self.T) ** 2))

        # Add diversity maintenance
        diversity = np.std(self.fitness) / (np.mean(self.fitness) + 1e-8)
        adaptive_df = base_df * (1 + diversity)

        for i in range(self.N):
            # Random perturbation
            r = 2 * np.random.rand(self.dim) - 1

            # Update with adaptive DF
            new_population[i] = self.population[i] + R * r * self.population[i] * adaptive_df

            # Handle parameter types
            for j, param_name in enumerate(self.param_names):
                if param_name in self.discrete_params:
                    new_population[i, j] = np.round(new_population[i, j])

                new_population[i, j] = np.clip(new_population[i, j],
                                               self.L_bound[j], self.U_bound[j])

        # Evaluate
        new_fitness = np.array([self.fitness_function(self._decode_individual(ind))
                                for ind in new_population])

        # Update if improved
        for i in range(self.N):
            if new_fitness[i] < self.fitness[i]:
                self.population[i] = new_population[i]
                self.fitness[i] = new_fitness[i]

    def optimize(self) -> Tuple[Dict, float, List[float]]:
        """
        Run enhanced SANGO optimization.

        Returns:
            best_params: Best hyperparameters (as dict)
            best_fitness: Best fitness (1 - F1_score)
            convergence_curve: Fitness history
        """
        # Initialize
        self.initialize_population()
        self.convergence_curve = [self.best_fitness]

        # Main loop
        for t in range(self.T):
            if self.verbose and (t % 5 == 0 or t == self.T - 1):
                params = self._decode_individual(self.best_individual)
                f1_score = 1 - self.best_fitness
                self.logger.info(f"Iter {t}/{self.T}, F1: {f1_score:.4f}, Params: {params}")

            # SANGO phases
            self.prey_identification(t)
            self.prey_capture(t)

            # Update global best
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_individual = self.population[best_idx].copy()
                self.best_fitness = self.fitness[best_idx]

                if self.verbose:
                    params = self._decode_individual(self.best_individual)
                    f1 = 1 - self.best_fitness
                    self.logger.info(f"→ New best! F1: {f1:.4f}, Params: {params}")

            # Record convergence
            self.convergence_curve.append(self.best_fitness)
            self.fitness_history.append({
                'iteration': t,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(self.fitness),
                'std_fitness': np.std(self.fitness),
                'best_params': self._decode_individual(self.best_individual)
            })

        best_params = self._decode_individual(self.best_individual)

        if self.verbose:
            f1_final = 1 - self.best_fitness
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Optimization Complete!")
            self.logger.info(f"Best F1-Score: {f1_final:.4f}")
            self.logger.info(f"Best Parameters:")
            for k, v in best_params.items():
                self.logger.info(f"  {k}: {v}")
            self.logger.info(f"{'=' * 60}\n")

        return best_params, self.best_fitness, self.convergence_curve


def create_fitness_function_f1(model_class, train_loader, val_loader, device,
                               num_classes=5, max_epochs=10):
    """
    Create F1-score based fitness function for SANGO.

    Args:
        model_class: Model class to instantiate
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_classes: Number of classes
        max_epochs: Max training epochs per evaluation

    Returns:
        fitness_function: Function that takes params dict and returns 1 - F1_score
    """

    def fitness_function(params: Dict) -> float:
        """
        Train model with given hyperparameters and return 1 - F1_score.
        Lower is better.
        """
        try:
            # Extract parameters
            hidden_dim1 = params['hidden_dim1']
            hidden_dim2 = params['hidden_dim2']
            dropout_rate = params['dropout']
            learning_rate = params['lr']

            # Create model
            model = model_class(
                num_classes=num_classes,
                lstm_hidden_dim=hidden_dim1,
                lstm_layers=2,
                dropout=dropout_rate
            ).to(device)

            # Setup optimizer with given LR
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-5
            )

            # Simple training loop
            model.train()
            for epoch in range(max_epochs):
                for batch_idx, (images, labels, _) in enumerate(train_loader):
                    if batch_idx >= 10:  # Limit batches for speed
                        break

                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)

                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Evaluate on validation set
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            # Calculate F1 score (macro average for multi-class)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            # Return 1 - F1 (lower is better for minimization)
            fitness = 1.0 - f1

            # Cleanup
            del model, optimizer
            torch.cuda.empty_cache()

            return fitness

        except Exception as e:
            logging.error(f"Fitness evaluation failed for params {params}: {e}")
            return 1.0  # Return worst possible fitness

    return fitness_function


# Example usage
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)


    # Test with simple function
    def test_fitness(params):
        """Test function: minimize distance from target."""
        target = {'hidden_dim1': 128, 'hidden_dim2': 64, 'dropout': 0.3, 'lr': 1e-4}

        distance = sum((params[k] - target[k]) ** 2 for k in target.keys())
        return distance / 10000  # Normalize


    sango = EnhancedSANGO(
        fitness_function=test_fitness,
        dim=4,
        population_size=8,
        max_iterations=20,
        verbose=True
    )

    best_params, best_fitness, curve = sango.optimize()

    print(f"\nFound: {best_params}")
    print(f"Expected: hidden_dim1=128, hidden_dim2=64, dropout=0.3, lr=1e-4")
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
        print(f"🔸 Evaluating params: {hyperparams}")
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