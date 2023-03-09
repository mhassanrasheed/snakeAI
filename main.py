import copy
import math
import os
import random
from datetime import datetime

import numpy as np
import pygame
import torch
import torch.nn as nn

clock = pygame.time.Clock()
clock = pygame.time.Clock()
pygame.init()
red = (255, 0, 0)
green = (0, 255, 0)
yellow = (255, 255, 102)
width = 800
height = 600
speed = 10
screen = pygame.display.set_mode((width, height))
yellow = (255, 255, 102)
population = 500
snakes = []
top_limit = 3
gen = 0
is_human = False
is_learning = True
resume_learning = False
number_of_brains_from_previous_run = 10
if is_human:
    is_learning = False
prev_folder = "day4"
to_save_folder = "day6"
font_style = pygame.font.SysFont("bahnschrift", 15)
score_font = pygame.font.SysFont("comicsansms", 15)


def snake_fitness(fitness):
    value = score_font.render("fitness: " + str(fitness), True, yellow)
    screen.blit(value, [5, 30])


def your_score(score):
    value = score_font.render("Score: " + str(score), True, yellow)
    screen.blit(value, [5, 0])


def message(msg, color, x, y):
    mesg = font_style.render(msg, True, color)
    screen.blit(mesg, [x, y])


class SnakeAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(24, 16, bias=True),
            nn.ReLU(),
            # nn.Linear(16, 8, bias=True),
            # nn.ReLU(),
            nn.Linear(16, 4, bias=True),
            nn.Softmax(-1)
        )

    def forward(self, inputs):
        x = self.fc(inputs)
        return x


class Food:
    """
    A class to represent the food that the snake can eat.

    Attributes:
    -----------
    x : int
        The x-coordinate of the food on the screen.
    y : int
        The y-coordinate of the food on the screen.

    Methods:
    --------
    __init__(self):
        Initializes the Food object by randomly generating the x and y coordinates
        within the game screen.
    draw(self):
        Draws the food on the screen as a green rectangle.
    """

    def __init__(self):
        """
        Initializes the Food object by randomly generating the x and y coordinates
        within the game screen.
        """
        self.x = int(round(random.randrange(0, width - 10) / 10.0) * 10.0)
        self.y = int(round(random.randrange(0, height - 10) / 10.0) * 10.0)

    def draw(self):
        """
        Draws the food on the screen as a green rectangle.
        """
        pygame.draw.rect(screen, green, [self.x, self.y, 10, 10])


def food_collide(pos_x, pos_y, fx, fy):
    """
    Check if the position of a snake collides with the position of a food.

    Args:
    - pos_x (int): The x coordinate of the snake's head.
    - pos_y (int): The y coordinate of the snake's head.
    - fx (int): The x coordinate of the food.
    - fy (int): The y coordinate of the food.

    Returns:
    - bool: True if the snake's head is at the same position as the food, False otherwise.
    """
    if pos_x == fx and pos_y == fy:
        return True
    return False


class Snake:
    """
    A class representing a snake in a game.

    Attributes:
        block (int): The size of each block in the game board.
        food (class): Each snake will have it's own food.
        x (int): The x-coordinate of the snake's head.
        y (int): The y-coordinate of the snake's head.
        dx (int): The change in x-coordinate for the snake's movement.
        dy (int): The change in y-coordinate for the snake's movement.
        head (list): A list of the coordinates of each block in the snake's body.
        full (list): A list of the coordinates of each block in the snake's body.
        length (int): The length of the snake.
        direction (int): The direction the snake is facing (1 = up, 2 = right, 3 = down, 4 = left).
        steps_allowed (int): The maximum number of steps the snake can take.
        steps_taken (int): The number of steps taken by the snake to eat a food.
        life_time (int): The number of steps the snake has taken so far.
        score (int): The current score of the snake.
        fitness (int): The fitness score of the snake (used for evolutionary algorithms).
        brain (SnakeAI): The artificial intelligence controlling the snake.
        color (tuple): The color of the snake on the game board.

    Methods:
        __init__(): Initializes the snake's starting position, size, direction, and other attributes.
        draw(self): Draws the snake on the game screen.
        move(self): Moves the snake in the current direction by one block and updates the snake's position.
        grow(self): Increases the length of the snake by 1 block.
        is_on_tail(self): Checks if the snake has collided with its own tail.
        gonna_die(self, x, y): Checks if the snake is going to die by colliding with the wall or its own tail.
        body_collide(self, pos_x, pos_y): Checks if the given position collides with any part of the snake's body.
        look_in_direction(self, x, y, fx, fy): Looks in a particular direction (x, y) from the current position of the snake and returns information about what it sees.
        look(self, fx, fy):  Generates a 24-dimensional array that represents the snake's field of vision, looking in eight different directions (left, up-left, up, up-right, right, down-right, down, down-left), with three dimensions for each direction (0 for no food/body detected, 1 for food/body detected). The third dimension for each direction represents the inverse of the distance to the nearest object in that direction.
        think(self, fx: int, fy: int): Determines the direction of the next move based on the current state of the snake.
    """

    def __init__(self):
        # Initialize the snake's starting position, size, direction, and other attributes
        self.block = 10
        self.food = Food()
        # self.x = int(round(random.randrange(0, height - self.block) / 10.0) * 10.0)
        # self.y = int(round(random.randrange(0, height - self.block) / 10.0) * 10.0)
        self.x = 450
        self.y = 230
        self.dx = 0
        self.dy = 0
        self.head = []
        self.full = []
        self.length = 3
        self.direction = 4
        self.steps_allowed = 500
        self.life_time = 0
        self.steps_taken = 0
        self.score = 0
        self.fitness = 0
        self.brain = SnakeAI()
        self.color = red
        for param in self.brain.parameters():
            param.requires_grad = False

    def draw(self):
        """
        Draws the snake on the game screen.

        This method loops through all the blocks in the snake's body and draws them on the screen using the Pygame library's
        draw method.

        Args:
            None

        Returns:
            None
        """
        for x in self.full:
            # Pygame's draw method takes the following arguments: (screen, color, (x,y,width,height))
            # Here, we pass in the game screen object, the color of the snake, and the (x,y) coordinates and dimensions of the block.
            pygame.draw.rect(screen, self.color,
                             (x[0], x[1], self.block, self.block))

    def move(self):
        """
        Moves the snake in the current direction by one block and updates the snake's position.

        The snake's direction is indicated by a number from 0-3, where 0 is up, 1 is down, 2 is left, and 3 is right.
        The snake is moved by updating its x and y position based on its current direction and block size.
        The snake's head is added to the list of its full body parts, and if the snake's length exceeds its maximum length,
        the oldest body part is removed from the list.

        Returns:
            None
        """
        if self.direction is 0:
            self.dy = -self.block
            self.dx = 0
        if self.direction is 1:
            self.dy = self.block
            self.dx = 0
        if self.direction is 2:
            self.dy = 0
            self.dx = -self.block
        if self.direction is 3:
            self.dy = 0
            self.dx = self.block
        self.steps_allowed -= 1
        self.life_time += 0.01
        self.steps_taken += 1
        self.x += self.dx
        self.y += self.dy
        self.head = []
        self.is_on_tail()
        self.head.append(self.x)
        self.head.append(self.y)
        self.full.append(self.head)
        if len(self.full) > self.length:
            del self.full[0]

    def grow(self):
        """
        Increases the length of the snake by 1 block.

        This method is called when the snake eats a piece of food, and it adds a new block to the end of the snake's tail,
        effectively increasing the length of the snake.
        """
        self.length += 1

    def is_on_tail(self):
        """
        Checks if the snake has collided with its own tail.

        This method checks if the head of the snake has collided with any part of its own tail. If a collision is detected,
        it means the snake has collided with itself and the method returns True. Otherwise, it returns False.

        Returns:
            bool: True if the snake has collided with its own tail, False otherwise.
        """
        for x in self.full:
            if self.x == x[0] + self.block and self.y == x[1] + self.block:
                return True
        return False

    def gonna_die(self, x, y):
        """
        Checks if the snake is going to die by colliding with the wall or its own tail.

        Args:
            x (int): The x-coordinate of the snake's head.
            y (int): The y-coordinate of the snake's head.

        Returns:
            bool: True if the snake is going to die, False otherwise.
        """
        # Check if the snake is going to collide with the wall
        if x >= width or x <= 0 - self.block or y >= height or y <= 0 - self.block:
            return True

        # Check if the snake is going to collide with its own tail
        return self.is_on_tail()

    def body_collide(self, pos_x, pos_y):
        """
        Checks if the given position collides with any part of the snake's body.

        Args:
            pos_x (int): The x-coordinate of the position to be checked.
            pos_y (int): The y-coordinate of the position to be checked.

        Returns:
            bool: True if the position collides with the snake's body, False otherwise.
        """
        for x in self.full:
            if pos_x == x[0] and pos_y == x[1]:
                return True
        return False

    def look_in_direction(self, x, y):
        """
        Looks in a particular direction (x, y) from the current position of the snake and returns information about what it
        sees.

        Args:
            x (int): The x direction in which the snake should look.
            y (int): The y direction in which the snake should look.
            fx (int): The x coordinate of the food on the board.
            fy (int): The y coordinate of the food on the board.

        Returns:
            list: A list of 3 elements representing what the snake can see in the given direction:
                - The first element is a 1 if there is food in the line of sight, 0 otherwise.
                - The second element is a 1 if there is any part of the snake's body in the line of sight, 0 otherwise.
                - The third element is a normalized value representing the distance from the snake's head to the first
                    obstacle (either food or the snake's body) in the line of sight.
        """
        look = [0, 0, 0]
        pos_x, pos_y = self.x + x, self.y + y
        distance = 0
        food_found = False
        body_found = False
        distance += 1
        while not self.gonna_die(pos_x, pos_y):
            if not food_found and food_collide(pos_x, pos_y, self.food.x, self.food.y):
                food_found = True
                look[0] = 1
            if not body_found and self.body_collide(pos_x, pos_y):
                body_found = True
                look[1] = 1
            pos_x += x
            pos_y += y
            # pygame.draw.line(screen, (0, 0, 255), (self.x + x, self.y + y), (pos_x, pos_y))
            distance += 1
        look[2] = 1 / distance
        return look

    def look(self):
        """
        Generates a 24-dimensional array that represents the snake's field of vision,
        looking in eight different directions (left, up-left, up, up-right, right, down-right, down, down-left),
        with three dimensions for each direction (0 for no food/body detected, 1 for food/body detected).
        The third dimension for each direction represents the inverse of the distance to the nearest object in that direction.

        Args:
        fx (int): the x-coordinate of the food the snake is seeking.
        fy (int): the y-coordinate of the food the snake is seeking.

        Returns:
        list: a 24-dimensional array that represents the snake's field of vision.
        """
        vision = [0 for x in range(24)]
        temp = self.look_in_direction(-self.block, 0)
        vision[0] = temp[0]
        vision[1] = temp[1]
        vision[2] = temp[2]

        temp = self.look_in_direction(-self.block, -self.block)
        vision[3] = temp[0]
        vision[4] = temp[1]
        vision[5] = temp[2]

        temp = self.look_in_direction(0, -self.block)
        vision[6] = temp[0]
        vision[7] = temp[1]
        vision[8] = temp[2]

        temp = self.look_in_direction(self.block, -self.block)
        vision[9] = temp[0]
        vision[10] = temp[1]
        vision[11] = temp[2]

        temp = self.look_in_direction(self.block, 0)
        vision[12] = temp[0]
        vision[13] = temp[1]
        vision[14] = temp[2]

        temp = self.look_in_direction(self.block, self.block)
        vision[15] = temp[0]
        vision[16] = temp[1]
        vision[17] = temp[2]

        temp = self.look_in_direction(0, self.block)
        vision[18] = temp[0]
        vision[19] = temp[1]
        vision[20] = temp[2]

        temp = self.look_in_direction(-self.block, self.block)
        vision[21] = temp[0]
        vision[22] = temp[1]
        vision[23] = temp[2]

        return vision

    def think(self):
        """
        Determines the direction of the next move based on the current state of the snake.

        Args:
        - fx (int): The x-coordinate of the food.
        - fy (int): The y-coordinate of the food.

        Returns:
        - None

        This method looks at the environment from the snake's perspective using the `look` method, and converts
        it into an input tensor to the neural network `brain`. The neural network outputs a probability distribution
        over the possible actions (i.e., move up, down, left or right). The direction of the next move is then determined
        by randomly sampling an action from the distribution.

        """
        observation = self.look()
        inp = torch.tensor(observation).type('torch.FloatTensor').view(1, -1)
        output_probabilities = self.brain(inp).detach().numpy()[0]
        self.direction = np.random.choice(
            range(4), 1, p=output_probabilities).item()


def select_top(previous: list[Snake], limit: int):
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


def mutate(DNA):
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


def make_babies(DNA):
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


def draw(snake: Snake, food: Food):
    """
    Draws the game screen.

    Args:
        snake (Snake): An instance of the Snake class representing the game's snake.
        food (Food): An instance of the Food class representing the game's food.

    Returns:
        None
    """
    # Set the background color to black
    screen.fill((0, 0, 0))

    # Display the player's score
    your_score(snake.score)

    # Display the generation number
    message("Gen : " + str(gen), yellow, 5, 30)

    # Draw the food
    food.draw()

    # Move the snake
    snake.move()

    # Draw the snake
    snake.draw()

    # Set the game clock to tick at 20 frames per second
    # clock.tick(20)

    # Update the display to show the current screen
    pygame.display.update()


def run_for_youtube(brain, display: bool):
    global gen
    dead = False
    top_snake = Snake()
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
        draw(top_snake, top_snake.food)

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
            top_snake.food = Food()
            top_snake.grow()
        if display:
            pygame.time.delay(speed)
        else:
            pygame.time.delay(1)
    if not display:
        return top_snake.fitness


def weighted_random_choice(chromosomes: list[Snake]) -> Snake:
    max = sum(chromosome.fitness for chromosome in chromosomes)
    pick = random.uniform(0, max)
    current = 0
    for chromosome in chromosomes:
        current += chromosome.fitness
        if current > pick:
            return chromosome


def mate(parents: list[Snake]) -> list[Snake]:
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


def next_generation(previous: list[Snake]) -> list[Snake]:
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
        reward += snake.length * 10
        reward += snake.score * 30

    # Once the snake has eaten 10 foods, encourage it to take the shortest path to the food
    if snake.score >= 10:

        # Increase the fitness score based on the inverse of the distance to the food,
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
    - s (List[Snake]): The list of Snake objects to evolve.

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
                snake.food = Food()
                snake.grow()


def pick_some_trained_brains(number_of_files: int) -> list[SnakeAI]:
    # Get the path to the directory containing the saved brains
    brain_dir = prev_folder

    # Get a list of all the saved brains in the directory
    brain_files = os.listdir(brain_dir)

    # Extract the numerical part from the file names and convert it to an integer
    # Use a list comprehension to create a list of tuples containing the numerical part and the file name
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


torch.set_grad_enabled(False)
default_time = datetime.min

if is_learning:
    if resume_learning:
        for snake in range(population-number_of_brains_from_previous_run):
            snake = Snake()
            snakes.append(snake)

        for file in pick_some_trained_brains(number_of_brains_from_previous_run):
            snake = Snake()
            snake.brain.load_state_dict(
                torch.load(os.path.join(prev_folder, file)))
            snakes.append(snake)
    else:
        for snake in range(population):
            snake = Snake()
            snakes.append(snake)


else:
    for brain_file in pick_some_trained_brains(100):
        snake = Snake()
        snake.brain.load_state_dict(
            torch.load(os.path.join(prev_folder, brain_file)))
        run_for_youtube(snake.brain, True)

while gen < 10000:
    run(snakes)
    for new_brain in next_generation(for_next):
        new_snake = Snake()
        new_snake.brain = new_brain
        snakes.append(new_snake)
