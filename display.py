from config import yellow, green, display_speed, food_size
import pygame
from food import Food
from snake import Snake


class Display:
    def __init__(self, width, height):
        """
        Initializes the Display object.

        Args:
            width (int): Width of the game screen.
            height (int): Height of the game screen.
        """

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.font_style = pygame.font.SysFont("bahnschrift", 15)
        self.score_font = pygame.font.SysFont("comicsansms", 15)
        self.clock = pygame.time.Clock()

    def draw_food(self, food: Food) -> None:
        """
        Draws the food on the screen as a green rectangle.

        Args:
            food (Food): An instance of the Food class representing the game's food.

        Returns:
            None
        """
        pygame.draw.rect(self.screen, green, [
                         food.x, food.y, food_size, food_size])

    def draw_snake(self, snake: Snake) -> None:
        """
        Draws the snake on the game screen.

        This method loops through all the blocks in the snake's body and draws them on the screen.

        Args:
            snake (Snake): An instance of the Snake class representing the game's snake.

        Returns:
            None
        """
        for x in snake.full:
            pygame.draw.rect(self.screen, snake.color,
                             (x[0], x[1], snake.block, snake.block))

    def message(self, msg, color, x, y):
        """
        Displays a message on the screen.

        Args:
            msg (str): The message to display.
            color: The color of the message.
            x (int): X-coordinate of the message.
            y (int): Y-coordinate of the message.

        Returns:
            None
        """
        message = self.font_style.render(msg, True, color)
        self.screen.blit(message, [x, y])

    def your_score(self, score):
        """
        Displays the player's score on the screen.

        Args:
            score: The player's score.

        Returns:
            None
        """
        value = self.score_font.render("Score: " + str(score), True, yellow)
        self.screen.blit(value, [5, 0])

    def snake_fitness(self, fitness):
        """
        Displays the snake's fitness on the screen.

        Args:
            fitness: The snake's fitness.

        Returns:
            None
        """
        value = self.score_font.render(
            "Fitness: " + str(fitness), True, yellow)
        self.screen.blit(value, [5, 30])

    def update_display(self):
        """
        Update the display to show the current screen.

        Returns:
            None
        """
        pygame.display.update()

    def control_frame_rate(self, speed: int) -> None:
        """
        Control the frame rate of the game.

        Args:
            speed (int): The desired frame rate.

        Returns:
            None
        """
        self.clock.tick(speed)


class GameDisplay(Display):
    def draw(self, snake: Snake, food: Food, gen: int) -> None:
        """
        Draws the game screen.

        Args:
            snake (Snake): An instance of the Snake class representing the game's snake.
            food (Food): An instance of the Food class representing the game's food.
            gen (int): The generation number.

        Returns:
            None
        """
        # Set the background color to black
        self.screen.fill((0, 0, 0))

        # Display the player's score
        self.your_score(snake.score)

        # Display the generation number
        self.message("Gen : " + str(gen), yellow, 5, 30)

        # Draw the food
        self.draw_food(food)

        # Move the snake
        snake.move()

        # Draw the snake
        self.draw_snake(snake)

        self.control_frame_rate(display_speed)
        # Update the display to show the current screen
        self.update_display()
