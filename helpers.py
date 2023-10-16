from snake import Snake
from snakeAI import SnakeAI
import numpy as np
import random
from config import prev_folder
import os


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


def pick_some_trained_brains(number_of_files: int) -> list[SnakeAI]:
    """
    Selects the already trained snakes from the previous training.

    Args:
        number_of_files (int): A number of trained snakes to be selected

    Returns:
        list: A list of trained snakes
    """
    # Path to the directory containing the saved brains
    brain_dir = prev_folder

    # List of all the saved brains in the directory
    brain_files = os.listdir(brain_dir)

    # Extracting the numerical part from the file names and converting it to an integer
    # Using a list comprehension to create a list of tuples containing the numerical part and the file name
    # For example, if the file name is "10.pth", then the tuple would be (10, "10.pth")
    brain_files_int = [(int(file_name.split('.')[0]), file_name)
                       for file_name in brain_files]

    # Sort the file names based on the numerical part
    # Use the first element of each tuple (i.e., the numerical part) for sorting
    # Use the "reverse=True" argument to sort in descending order
    brain_files_int.sort(reverse=True)

    # Get the last "number_of_files" brain file names
    # Use a list comprehension to extract only the file names from the sorted list of tuples
    # Use slicing to get only the desired number of files
    return [file_name for _, file_name in brain_files_int[:number_of_files]]
