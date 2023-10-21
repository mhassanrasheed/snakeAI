from helpers import weighted_random_choice
from snake import Snake
from snakeAI import SnakeAI
import random
from config import population
import copy
import numpy as np


class GeneticAlgorithm:

    def __init__(self) -> None:
        self.mutation_power = 0.00

    def mate(self, parents: list[Snake]) -> list[SnakeAI]:
        """
        Performs crossover and mutation on a population of parent individuals to generate a set of child individuals.

        Parameters:
        - parents (list): A list of parent individuals.

        Returns:
        - babies (list): A list of child individuals created through crossover and mutation of the parent individuals.
        """
        # Create a new instance of the SnakeAI class as the DNA for the child individuals
        DNA = SnakeAI()

        # Iterate through the parameters of the DNA and perform crossover
        for param in DNA.parameters():

            # Select two random parent individuals to perform crossover with
            x = weighted_random_choice(parents)
            y = weighted_random_choice(parents)

            # Ensure that x and y are distinct parent individuals
            while x is y:
                y = random.choice(parents)

            for param in DNA.parameters():
                if len(param.shape) == 2:  # weights of linear layer
                    for i0 in range(param.shape[0]):
                        for i1 in range(param.shape[1]):
                            if i0 < (param.shape[0] / 2) and i1 < (param.shape[1]):
                                for p in x.brain.parameters():
                                    if len(p.shape) == 2 and param.shape == p.shape:
                                        param[i0][i1] = p[i0][i1]
                            else:
                                for p in y.brain.parameters():
                                    if len(p.shape) == 2 and param.shape == p.shape:
                                        param[i0][i1] = p[i0][i1]

                elif len(param.shape) == 1:  # biases of linear layer or conv layer
                    for i0 in range(param.shape[0]):
                        if i0 < (param.shape[0] / 2):
                            for p in x.brain.parameters():
                                if len(p.shape) == 1 and param.shape == p.shape:
                                    param[i0] = p[i0]
                        else:
                            for p in y.brain.parameters():
                                if len(p.shape) == 1 and param.shape == p.shape:
                                    param[i0] = p[i0]
        babies = self.make_babies(DNA)
        return babies

    def make_babies(self, DNA: SnakeAI) -> list[SnakeAI]:
        """
        Creates a list of mutated copies of the input neural network's parameters.

        Args:
        - DNA (torch.nn.Module): The neural network to be mutated.

        Returns:
        - babies (List[torch.nn.Module]): A list of mutated neural networks.
        """
        babies = []
        # create population - 1 mutated copies of the DNA
        for i in range(population - 1):
            babies.append(self.mutate(DNA))
        return babies

    def mutate(self, DNA: SnakeAI) -> SnakeAI:
        """
        Returns a mutated copy of the provided DNA neural network.

        Args:
        - DNA: the neural network to be mutated, represented as a PyTorch module.

        Returns:
        - A mutated copy of the provided DNA neural network.
        """
        # Create a deep copy of the provided DNA neural network.
        baby = copy.deepcopy(DNA)

        # Iterate through all the parameters in the neural network.
        for param in baby.parameters():

            # If the parameter is a weight matrix (i.e., a 2D tensor),
            # apply Gaussian noise to each of its elements.
            if len(param.shape) == 2:
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        param[i0][i1] += self.mutation_power * \
                            np.random.randn()

            # If the parameter is a bias vector (i.e., a 1D tensor),
            # apply Gaussian noise to each of its elements.
            elif len(param.shape) == 1:
                for i0 in range(param.shape[0]):
                    param[i0] += self.mutation_power * np.random.randn()

        # Return the mutated copy of the DNA neural network.
        return baby


class SnakeLearning(GeneticAlgorithm):
    def calculate_fitness(self, snake: Snake) -> float:
        """
        Calculates the fitness score of a given snake.

        Args:
            snake (Snake): The snake object for which the fitness score needs to be calculated.

        Returns:
            float: The fitness score of the given snake.
        """
        # Encourage the snake to stay alive for as long as possible
        return snake.life_time

    def food_reward(self, snake: Snake) -> float:
        """
        Calculates the food reward for a given snake.

        Args:
            snake (Snake): The snake object for which the food reward needs to be calculated.

        Returns:
            float: The food reward for the given snake.
        """
        reward = 0
        # Give more emphasis to length and number of foods eaten as the game progresses
        if snake.score >= 2:
            reward += snake.length * 50
            reward += snake.score * 80

        # Once the snake has eaten 10 foods, encourage it to take the shortest path to the food
        if snake.score >= 10:

            # Increase the fitness score based on the inverse of the number of steps taken to reach the food,
            # encouraging the snake to take the shortest path
            reward += (1 / snake.steps_taken) * 1000

        return reward
