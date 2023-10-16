import random


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
    __init__(self, width, height):
        Initializes the Food object by randomly generating the x and y coordinates
        within the game screen.
    """

    def __init__(self, width, height):
        """
        Initializes the Food object by randomly generating the x and y coordinates
        within the game screen.
        """
        self.x = int(round(random.randrange(0, width - 20) / 10.0) * 10.0)
        self.y = int(round(random.randrange(0, height - 20) / 10.0) * 10.0)
