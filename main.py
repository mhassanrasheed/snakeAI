import copy
import os
import random
from snakeAI import SnakeAI
from food import Food
from snake import Snake
from helpers import select_top, weighted_random_choice, pick_some_trained_brains
from display import GameDisplay
import numpy as np
import pygame
import torch
import torch.nn as nn
from config import red, width, height, speed, \
    population, gen, is_human, is_learning, resume_learning, \
    number_of_brains_from_previous_run, code_debugging,\
    prev_folder, to_save_folder

if is_human:
    is_learning = False

snakes = []


visual = GameDisplay(width=width, height=height)


def mutate(DNA: SnakeAI) -> SnakeAI:
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
                    param[i0][i1] += mutation_power * np.random.randn()

        # If the parameter is a bias vector (i.e., a 1D tensor),
        # apply Gaussian noise to each of its elements.
        elif len(param.shape) == 1:
            for i0 in range(param.shape[0]):
                param[i0] += mutation_power * np.random.randn()

    # Return the mutated copy of the DNA neural network.
    return baby


def make_babies(DNA: SnakeAI) -> list[SnakeAI]:
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
        babies.append(mutate(DNA))
    return babies


def run_for_youtube(brain, display: bool):
    global gen
    dead = False
    top_snake = Snake(width=width, height=height, color=red, brain=SnakeAI())
    top_snake.brain = brain

    # Run the game until the snake dies
    while not dead:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                dead = False
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if is_human:
                    if event.key == pygame.K_LEFT:
                        top_snake.direction = 2
                    elif event.key == pygame.K_RIGHT:
                        top_snake.direction = 3
                    elif event.key == pygame.K_UP:
                        top_snake.direction = 0
                    elif event.key == pygame.K_DOWN:
                        top_snake.direction = 1
        # Make the snake think about what move to make
        top_snake.think()

        top_snake.fitness += calculate_fitness(top_snake)
        # Draw the game on the screen
        visual.draw(top_snake, top_snake.food, gen)

        # Check if the snake is dead
        if top_snake.steps_allowed <= 0:
            dead = True
        if top_snake.x >= width or top_snake.x <= 0 or top_snake.y >= height or top_snake.y <= 0:
            dead = True
        for y in top_snake.full[:-1]:
            if y == top_snake.head:
                dead = True

        # Check if the snake ate the food
        if top_snake.x == top_snake.food.x and top_snake.y == top_snake.food.y:
            top_snake.score += 1
            top_snake.steps_allowed += 200
            top_snake.fitness += food_reward(top_snake)
            top_snake.steps_taken = 0
            top_snake.food = Food(width=width, height=height)
            top_snake.grow()
        if display:
            pygame.time.delay(speed)
        else:
            pygame.time.delay(1)
    if not display:
        return top_snake.fitness


def mate(parents: list[Snake]) -> list[SnakeAI]:
    """
    Performs crossover and mutation on a population of parent individuals to generate a set of child individuals.

    Parameters:
    parents (list): A list of parent individuals.

    Returns:
    babies (list): A list of child individuals created through crossover and mutation of the parent individuals.

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
    babies = make_babies(DNA)
    return babies


def return_average_score(agent: Snake, runs: int) -> float:
    score = 0
    for i in range(runs):
        score += run_for_youtube(agent.brain, False)
    return score / runs


def run_agents_n_times(agents: list[Snake], runs: int) -> float:
    avg_score = []
    for agent in agents:
        avg_scored = return_average_score(agent, runs)
        avg_score.append(avg_scored)
        print("score : ", agent.score, "fitness : ",
              agent.fitness, "average fitness : ", avg_scored)
    return avg_score


def next_generation(previous: list[Snake]) -> list[SnakeAI]:
    global mutation_power
    global gen
    gen += 1
    print(
        f"*************************** GENERATION {gen} ***************************")
    tp_10 = []
    tp_2 = []
    top_10_idx = select_top(previous, 10)
    for x in top_10_idx:
        tp_10.append(previous[x])
    averages = run_agents_n_times(tp_10, 3)
    total = 0
    for average in averages:
        total += average
    mutation_power = max(averages)/total

    if mutation_power < 0.025:
        mutation_power = 0.025
    babies = mate(tp_10)
    babies.append(tp_10[0].brain)
    if not code_debugging:
        torch.save(tp_10[0].brain.state_dict(), f"{to_save_folder}/{gen}.pth")
    print("mutation rate : ", mutation_power)
    return babies


def calculate_fitness(snake: Snake) -> float:
    """
    Calculates the fitness score of a given snake.

    The fitness score is a dynamic function that takes into account the length of the snake, the time it has been alive,
    and the number of foods it has eaten.

    At the beginning of the game, the emphasis is more on the time the snake stays alive. As the game progresses,
    more emphasis is given to the length and the number of foods eaten.

    Once the snake has eaten 10 foods, the emphasis is given to the shortest path to the food.

    Args:
        snake (Snake): The snake object for which the fitness score needs to be calculated.

    Returns:
        float: The fitness score of the given snake.
    """
    # Encourage the snake to stay alive for as long as possible
    return snake.life_time


def food_reward(snake: Snake) -> float:
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


def run(snakes: list[Snake]) -> None:
    """
    Run a genetic algorithm to evolve the given list of Snake objects. This function updates each Snake's fitness score
    and removes those that have gone out of bounds or collided with themselves. It also handles Snake-object specific
    logic such as making the Snake grow when it eats the Food object, and updating its step count and position based on
    its current direction.

    Args:
    - snakes (List[Snake]): The list of Snake objects to evolve.

    Returns:
    - None
    """
    global for_next
    for_next = []

    while len(snakes) > 0:
        # Iterate through each snake in the snakes list
        for x, snake in enumerate(snakes):
            # Evaluate the snake's neural network
            snake.brain.eval()

            # If learning is enabled, use the neural network to determine the snake's next move
            if is_learning:
                snake.think()

            # Move the snake
            snake.move()

            # Update the snake's its fitness score
            snake.fitness += calculate_fitness(snake)

            # If the snake has run out of steps, mark it for removal from the snakes list
            if snake.steps_allowed <= 0:
                for_next.append(snake)
                if x is len(snakes):
                    snakes.pop()
                else:
                    snakes.pop(x)

            # If the snake has collided with a wall or with itself, mark it for removal from the snakes list
            elif snake.x >= width or snake.x <= 0 or snake.y >= height or snake.y <= 0:
                for_next.append(snake)
                if x is len(snakes):
                    snakes.pop()
                else:
                    snakes.pop(x)
            else:
                for y in snake.full[:-1]:
                    if y == snake.head:
                        for_next.append(snake)
                        if x is len(snakes):
                            snakes.pop()
                        else:
                            snakes.pop(x)

            # If the snake has eaten the food, update its score and spawn a new food at a random location
            if snake.x == snake.food.x and snake.y == snake.food.y:
                snake.score += 1
                snake.steps_allowed += 200
                snake.fitness += food_reward(snake)
                snake.steps_taken = 0
                snake.food = Food(width=width, height=height)
                snake.grow()


torch.set_grad_enabled(False)

if is_learning:
    if resume_learning:
        for snake in range(population-number_of_brains_from_previous_run):
            snake = Snake(width=width, height=height,
                          color=red, brain=SnakeAI())
            snakes.append(snake)

        for file in pick_some_trained_brains(number_of_brains_from_previous_run):
            snake = Snake(width=width, height=height,
                          color=red, brain=SnakeAI())
            snake.brain.load_state_dict(
                torch.load(os.path.join(prev_folder, file)))
            snakes.append(snake)
    else:
        for snake in range(population):
            snake = Snake(width=width, height=height,
                          color=red, brain=SnakeAI())
            snakes.append(snake)


else:
    for brain_file in pick_some_trained_brains(100):
        snake = Snake(width=width, height=height, color=red, brain=SnakeAI())
        snake.brain.load_state_dict(
            torch.load(os.path.join(prev_folder, brain_file)))
        run_for_youtube(snake.brain, True)

while gen < 10000:
    run(snakes)
    for new_brain in next_generation(for_next):
        new_snake = Snake(width=width, height=height,
                          color=red, brain=SnakeAI())
        new_snake.brain = new_brain
        snakes.append(new_snake)
