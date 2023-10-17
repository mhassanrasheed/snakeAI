from helpers import weighted_random_choice
from snake import Snake
from snakeAI import SnakeAI
import random
from config import population
import copy
import numpy as np


class GeneticAlgorithm:

    def __init__(self) -> None:
        """
        Initializes the GeneticAlgorithm object.
        """
        self.mutation_power = 0.0

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

        avg_score = []
        for agent in agents:
            avg_scored = return_average_score(agent, runs)
            avg_score.append(avg_scored)
            print("score : ", agent.score, "fitness : ",
                  agent.fitness, "average fitness : ", avg_scored)
        return avg_score
