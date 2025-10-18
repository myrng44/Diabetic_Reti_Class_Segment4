"""
Self-Adaptive Northern Goshawk Optimization (SANGO) Algorithm
Based on the paper: "A multi model deep net with an explainable AI based framework
for diabetic retinopathy segmentation and classification"

This file contains an improved SANGO implementation tuned for hyperparameter
search of discrete GRU hyperparameters. Key improvements over the original
version:
 - Keep internal population as floats for smooth updates; round only when
   performing a fitness evaluation.
 - Fitness caching to avoid repeated expensive training runs for identical
   hyperparameter tuples.
 - Elitism to preserve best individuals between generations.
 - Levy-flight occasional jumps to improve exploration.
 - Adaptive dynamic factor (DF) and radius (R) scheduling.
 - Small integer mutation to inject diversity for discrete search spaces.

The file also updates `optimize_gru_hyperparameters` to accept an optional
`sango_eval_epochs` argument which is passed to the training function (if
supported) to speed up SANGO evaluations during search.
"""

import math
import logging
from typing import Callable, List, Tuple, Dict

import numpy as np


class SANGO:
    """
    Self-Adaptive Northern Goshawk Optimization Algorithm (improved).

    Use this class to optimize integer-valued hyperparameters (e.g. [hidden_dim, num_layers]).
    The fitness function is expected to accept a 1-D numpy array (or list/tuple)
    of integer hyperparameters and return a scalar fitness (lower is better).
    """

    def __init__(
        self,
        fitness_function: Callable,
        dim: int = 2,
        population_size: int = 10,
        max_iterations: int = 50,
        lower_bound: List[int] = None,
        upper_bound: List[int] = None,
        learning_rate: float = 0.03,
        prey_capture_df: float = 0.4,
        prey_identification_r: float = 0.8,
        elitism_count: int = 2,
        mutation_rate: float = 0.1,
        verbose: bool = True,
    ):
        self.fitness_function = fitness_function
        self.dim = dim
        self.N = int(population_size)
        self.T = int(max_iterations)
        self.L_bound = np.array(lower_bound if lower_bound is not None else [16] * dim, dtype=float)
        self.U_bound = np.array(upper_bound if upper_bound is not None else [128] * dim, dtype=float)
        self.lr = float(learning_rate)
        self.df_coef = float(prey_capture_df)
        self.r_coef = float(prey_identification_r)
        self.elitism_count = max(1, int(elitism_count))
        self.mutation_rate = float(mutation_rate)
        self.verbose = bool(verbose)

        # internal state
        self.population = None  # float positions
        self.fitness = None
        self.best_individual = None  # stored as integer array
        self.best_fitness = float('inf')
        self.convergence_curve: List[float] = []

        # caching evaluations: key = tuple(ints) -> fitness
        self._fitness_cache: Dict[Tuple[int, ...], float] = {}

        # logger
        self.logger = logging.getLogger(__name__)

    # ---------- utility helpers ----------
    def _round_and_clip(self, arr: np.ndarray) -> np.ndarray:
        """Round to int and clip within bounds (returns int array).
        Does not modify internal float population.
        """
        a = np.round(arr).astype(int)
        a = np.clip(a, self.L_bound.astype(int), self.U_bound.astype(int))
        return a

    def _evaluate_individual(self, indiv: np.ndarray) -> float:
        """Evaluate an individual with caching. Accepts float array; rounds to ints
        for evaluation key and for passing to fitness_function.
        """
        key = tuple(int(x) for x in np.round(indiv))
        if key in self._fitness_cache:
            return self._fitness_cache[key]
        try:
            # call fitness function with a numpy array of ints
            val = float(self.fitness_function(np.array(key)))
        except Exception as e:
            self.logger.error(f"Fitness evaluation failed for {key}: {e}")
            val = 1e6
        self._fitness_cache[key] = val
        return val

    def _levy_flight(self, dim: int) -> np.ndarray:
        """Generate a Levy flight step vector."""
        beta = 1.5
        # sigma calculation for Levy distribution
        sigma = (
            (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)) /
            (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    # ---------- initialization ----------
    def initialize_population(self):
        """Initialize float population uniformly in bounds and evaluate."""
        self.population = np.zeros((self.N, self.dim), dtype=float)
        for i in range(self.N):
            self.population[i] = self.L_bound + np.random.rand(self.dim) * (self.U_bound - self.L_bound)

        # evaluate all individuals (rounded when evaluating)
        self.fitness = np.array([self._evaluate_individual(ind) for ind in self.population], dtype=float)

        best_idx = int(np.argmin(self.fitness))
        self.best_fitness = float(self.fitness[best_idx])
        self.best_individual = self._round_and_clip(self.population[best_idx]).copy()

        if self.verbose:
            self.logger.info(f"SANGO init best fitness: {self.best_fitness:.6f}, best: {self.best_individual}")

    # ---------- main phases ----------
    def prey_identification(self, t: int):
        """Phase 1: identifies prey and makes directed moves (exploration-focused)."""
        new_pop = self.population.copy()
        for i in range(self.N):
            # choose a random prey distinct from i
            candidates = [j for j in range(self.N) if j != i]
            prey_idx = np.random.choice(candidates)
            prey = self.population[prey_idx]

            r = np.random.rand(self.dim)
            I = np.random.choice([1, 2], size=self.dim)

            if self.fitness[prey_idx] < self.fitness[i]:
                # move toward prey
                new_pop[i] = self.population[i] + r * (prey - I * self.population[i]) * (1 - t / max(1, self.T))
            else:
                # move away from prey
                new_pop[i] = self.population[i] + r * (self.population[i] - prey) * (1 - t / max(1, self.T))

            # occasional Levy jump for exploration (more likely early)
            if np.random.rand() < max(0.03, 0.4 * (1 - t / max(1, self.T))):
                levy = self._levy_flight(self.dim)
                new_pop[i] += 0.05 * levy * (self.U_bound - self.L_bound)

            # small gaussian jitter
            new_pop[i] += 0.005 * np.random.randn(self.dim) * (self.U_bound - self.L_bound)

            # repair to bounds
            new_pop[i] = np.clip(new_pop[i], self.L_bound, self.U_bound)

        # evaluate and accept improvements
        new_f = np.array([self._evaluate_individual(ind) for ind in new_pop], dtype=float)
        for i in range(self.N):
            if new_f[i] < self.fitness[i]:
                self.population[i] = new_pop[i]
                self.fitness[i] = new_f[i]

    def prey_capture(self, t: int):
        """Phase 2: prey capture - local exploitation with adaptive DF and radius."""
        new_pop = self.population.copy()

        # radius R decays over time (stronger exploration early)
        R = self.r_coef * (1 - (t / max(1, self.T)) ** 1.2)

        # dynamic factor (DF) with small stochastic element
        DF = self.df_coef * (2 * np.random.rand(self.N, self.dim) - 1) * np.exp(-((t / max(1, self.T)) ** 2))

        for i in range(self.N):
            r = 2 * np.random.rand(self.dim) - 1
            step = R * r * self.population[i] * DF[i]

            # small integer mutation on discrete dims
            if np.random.rand() < self.mutation_rate:
                dim_idx = np.random.randint(0, self.dim)
                scale = max(1, int(0.03 * (self.U_bound[dim_idx] - self.L_bound[dim_idx])))
                change = np.random.randint(-scale, scale + 1)
                step[dim_idx] += change

            # occasional small levy for escape
            if np.random.rand() < max(0.01, 0.2 * (1 - t / max(1, self.T))):
                new_pop[i] += 0.03 * self._levy_flight(self.dim) * (self.U_bound - self.L_bound)

            new_pop[i] = self.population[i] + step
            new_pop[i] = np.clip(new_pop[i], self.L_bound, self.U_bound)

        # evaluate and accept
        new_f = np.array([self._evaluate_individual(ind) for ind in new_pop], dtype=float)
        for i in range(self.N):
            if new_f[i] < self.fitness[i]:
                self.population[i] = new_pop[i]
                self.fitness[i] = new_f[i]

    def _apply_elitism(self):
        """Preserve top `elitism_count` individuals by replacing worst ones."""
        idx_sorted = np.argsort(self.fitness)
        elites = idx_sorted[: self.elitism_count]
        worst = idx_sorted[-self.elitism_count :]
        for e, w in zip(elites, worst):
            if self.fitness[e] < self.fitness[w]:
                self.population[w] = self.population[e].copy()
                self.fitness[w] = self.fitness[e]

    # ---------- main optimize ----------
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """Run optimization and return best integer hyperparams, fitness, and curve."""
        self.initialize_population()
        self.convergence_curve = [self.best_fitness]

        for t in range(self.T):
            if self.verbose and (t % max(1, self.T // 10) == 0):
                self.logger.info(f"Iter {t}/{self.T}, best fitness {self.best_fitness:.6f}")

            # exploration then exploitation
            self.prey_identification(t)
            self.prey_capture(t)

            # elitism
            self._apply_elitism()

            # update best
            best_idx = int(np.argmin(self.fitness))
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = float(self.fitness[best_idx])
                self.best_individual = self._round_and_clip(self.population[best_idx]).copy()
                if self.verbose:
                    self.logger.info(f"New best @iter {t}: {self.best_fitness:.6f} -> {self.best_individual}")

            self.convergence_curve.append(self.best_fitness)

        if self.verbose:
            self.logger.info(f"Optimization complete. Best: {self.best_individual}, fitness: {self.best_fitness:.6f}")

        return self.best_individual, float(self.best_fitness), self.convergence_curve


# Wrapper for GRU hyperparameter optimization
def optimize_gru_hyperparameters(
    train_function: Callable,
    val_loader,
    device,
    population_size: int = 12,
    max_iterations: int = 40,
    lower_bounds: List[int] = None,
    upper_bounds: List[int] = None,
    sango_eval_epochs: int = None,
) -> Tuple[int, int]:
    """Use SANGO to find optimal GRU hyperparameters (hidden_dim, num_layers).

    train_function should accept keyword args at least: hidden_dim, num_layers,
    val_loader, device. If `sango_eval_epochs` is provided and the train_function
    supports an `eval_epochs` or `epochs` kwarg, it will be passed to speed up
    SANGO evaluations.
    """

    if lower_bounds is None:
        lower_bounds = [16, 1]
    if upper_bounds is None:
        upper_bounds = [128, 4]

    def fitness_function(hyperparams):
        hidden_dim = int(np.round(hyperparams[0]))
        num_layers = int(np.round(hyperparams[1]))

        kwargs = {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'val_loader': val_loader,
            'device': device,
        }
        # allow optional shorter eval during SANGO search if train_function accepts it
        if sango_eval_epochs is not None:
            # common possible kwarg names
            kwargs['eval_epochs'] = sango_eval_epochs
            kwargs['epochs'] = sango_eval_epochs

        try:
            val_loss = train_function(**kwargs)
            return float(val_loss)
        except TypeError:
            # try without eval_epochs/epochs if train_function doesn't accept them
            try:
                kwargs.pop('eval_epochs', None)
                kwargs.pop('epochs', None)
                val_loss = train_function(**kwargs)
                return float(val_loss)
            except Exception as e:
                logging.error(f"Train function failed for {hidden_dim},{num_layers}: {e}")
                return 1e6
        except Exception as e:
            logging.error(f"Train function failed for {hidden_dim},{num_layers}: {e}")
            return 1e6

    sango = SANGO(
        fitness_function=fitness_function,
        dim=2,
        population_size=population_size,
        max_iterations=max_iterations,
        lower_bound=lower_bounds,
        upper_bound=upper_bounds,
        learning_rate=0.03,
        prey_capture_df=0.5,
        prey_identification_r=0.9,
        elitism_count=2,
        mutation_rate=0.12,
        verbose=True,
    )

    best_hyperparams, best_fitness, curve = sango.optimize()
    optimal_hidden_dim = int(best_hyperparams[0])
    optimal_num_layers = int(best_hyperparams[1])

    return optimal_hidden_dim, optimal_num_layers


# --- simple local test when running this file directly ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    def test_fitness(x):
        # target is [64, 2]
        arr = np.array(x)
        return float(np.sum((arr - np.array([64, 2])) ** 2))

    s = SANGO(
        fitness_function=test_fitness,
        dim=2,
        population_size=12,
        max_iterations=30,
        lower_bound=[16, 1],
        upper_bound=[128, 4],
        verbose=True,
    )

    best_x, best_f, curve = s.optimize()
    print(f"\nOptimal solution: {best_x}")
    print(f"Optimal fitness: {best_f}")
    print(f"Expected: [64, 2]")