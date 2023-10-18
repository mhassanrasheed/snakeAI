import os
from snakeAI import SnakeAI
from food import Food
from snake import Snake
from helpers import select_top, weighted_random_choice, pick_some_trained_brains
from display import GameDisplay
from genetic import SnakeLearning
import pygame
import torch


from config import red, width, height, speed, \
    population, gen, is_human, is_learning, resume_learning, \
    number_of_brains_from_previous_run, code_debugging,\
    prev_folder, to_save_folder

if is_human:
    is_learning = False

snakes = []


visual = GameDisplay(width=width, height=height)
geneticAlgorithm = SnakeLearning()


def return_average_score(agent: Snake, runs: int) -> float:
    return sum([run(Snake(width=width, height=height,
                color=red, brain=agent.brain), True, False).fitness for _ in range(runs)]) / runs


def run_agents_n_times(agents: list[Snake], runs: int) -> float:
    avg_score = []
    for agent in agents:
        avg_scored = return_average_score(agent, runs)
        avg_score.append(avg_scored)
        print("score : ", agent.score, "fitness : ",
              agent.fitness, "average fitness : ", avg_scored)
    return avg_score


def next_generation(previous: list[Snake]) -> list[SnakeAI]:
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
    geneticAlgorithm.mutation_power = max(averages)/total

    if geneticAlgorithm.mutation_power < 0.025:
        geneticAlgorithm.mutation_power = 0.025
    babies = geneticAlgorithm.mate(tp_10)
    babies.append(tp_10[0].brain)
    if not code_debugging:
        torch.save(tp_10[0].brain.state_dict(), f"{to_save_folder}/{gen}.pth")
    print("mutation rate : ", geneticAlgorithm.mutation_power)
    return babies


def run(snake: Snake, display: bool, training: bool):
    """
    Run the game loop for a single snake.

    Args:
    - snake (Snake): The snake object to run the game loop for.
    - display (bool): A flag indicating whether to display the game.
    - training (bool): A flag indicating whether the snake is in training mode.

    Returns:
    - Snake or None: If `Training` is False, returns the snake object after the game loop;
      otherwise, returns None.
    """
    if display:
        # Needed for pygame window to appear
        for event in pygame.event.get():
            pass

    if training:
        # Evaluate the snake's brain in training mode
        snake.brain.eval()

    while snake.isAlive:
        # Make the snake think about its next move
        snake.think()

        # Draw the game if in display mode
        if display:
            visual.draw(snake, snake.food, gen)

        # Move the snake
        snake.move()

        # Update the snake's fitness score
        snake.fitness += geneticAlgorithm.calculate_fitness(snake)

        # Check if the snake has reached the end of its allowed steps or has collided with the wall or itself
        if (
            snake.steps_allowed <= 0 or
            snake.x >= width or snake.x <= 0 or snake.y >= height or snake.y <= 0 or
            any(y == snake.head for y in snake.full[:-1])
        ):
            snake.isAlive = False

        # Check if the snake has eaten the food
        if snake.x == snake.food.x and snake.y == snake.food.y:
            snake.score += 1
            snake.steps_allowed += 200
            snake.fitness += geneticAlgorithm.food_reward(snake)
            snake.steps_taken = 0
            snake.food = Food(width=width, height=height)
            snake.grow()

    return snake if not training else None


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
        run(snake.brain, True, False)

while gen < 10000:
    snakes = [Snake(width=width, height=height,
                    color=red, brain=new_brain) for new_brain in next_generation([run(snake, False, True) for snake in snakes])]
