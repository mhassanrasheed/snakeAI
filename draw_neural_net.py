from snake import Snake
from food import Food
from genetic import SnakeLearning
from config import width, height, red
from snakeAI import SnakeAI
import pygame
from display import Display
geneticAlgorithm = SnakeLearning()


snake = Snake(width, height, red, SnakeAI())

display = Display(width, height)

display.screen.fill((0, 0, 0))

for event in pygame.event.get():
    pass
