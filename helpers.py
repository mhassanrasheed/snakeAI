from snake import Snake
import numpy as np
import random


def select_top(previous: list[Snake], limit: int) -> list:
    """
    Selects the top performing snakes from the previous generation.

    The selection is based on the fitness score of each snake. The top `top_limit` snakes are selected
    to be the parents for the next generation.

    Args:
        previous (list): A list of snakes from the previous generation.

    Returns:
        list: A list of indexes of the top performing snakes from the previous generation.
    """
    # Create a list of rewards for each snake in the previous generation
    rewards = [snake.fitness for snake in previous]

    # Sort the rewards in descending order and select the top `top_limit` snakes
    sorted_parent_indexes = np.argsort(rewards)[::-1][:limit]

    return sorted_parent_indexes


def weighted_random_choice(chromosomes: list[Snake]) -> Snake:
    max = sum(chromosome.fitness for chromosome in chromosomes)
    pick = random.uniform(0, max)
    current = 0
    for chromosome in chromosomes:
        current += chromosome.fitness
        if current > pick:
            return chromosome
