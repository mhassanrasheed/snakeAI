from food import Food
from config import width, height
import numpy as np
import torch


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
        move(self): Moves the snake in the current direction by one block and updates the snake's position.
        grow(self): Increases the length of the snake by 1 block.
        is_on_tail(self): Checks if the snake has collided with its own tail.
        gonna_die(self, x, y): Checks if the snake is going to die by colliding with the wall or its own tail.
        body_collide(self, pos_x, pos_y): Checks if the given position collides with any part of the snake's body.
        look_in_direction(self, x, y, fx, fy): Looks in a particular direction (x, y) from the current position of the snake and returns information about what it sees.
        look(self, fx, fy):  Generates a 24-dimensional array that represents the snake's field of vision, looking in eight different directions (left, up-left, up, up-right, right, down-right, down, down-left), with three dimensions for each direction (0 for no food/body detected, 1 for food/body detected). The third dimension for each direction represents the inverse of the distance to the nearest object in that direction.
        think(self, fx: int, fy: int): Determines the direction of the next move based on the current state of the snake.
    """

    def __init__(self, width, height, color, brain):
        # Initialize the snake's starting position, size, direction, and other attributes
        self.block = 10
        self.food = Food(width=width, height=height)
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
        self.brain = brain
        self.color = color
        for param in self.brain.parameters():
            param.requires_grad = False

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

    def look_in_direction(self, x: int, y: int) -> list[int]:
        """
        Looks in a particular direction (x, y) from the current position of the snake and returns information about what it
        sees.

        Args:
            x (int): The x direction in which the snake should look.
            y (int): The y direction in which the snake should look.

        Returns:
            list: A list of 3 elements representing what the snake can see in the given direction:
                - The first element is a 1 if there is food in the line of sight, 0 otherwise.
                - The second element is a 1 if there is any part of the snake's body in the line of sight, 0 otherwise.
                - The third element is the inverse of the distance from the snake's head to the first
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
