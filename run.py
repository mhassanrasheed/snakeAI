from snake import Snake
from food import Food
from genetic import SnakeLearning
from config import width, height, red
from snakeAI import SnakeAI

geneticAlgorithm = SnakeLearning()


snake = Snake(width, height, red, SnakeAI())


def length(func):
    def wrapper(message, array):
        print(f"{message}, array : {array} ,  len: {len(array)}\n")
        return func(message, array)
    return wrapper


@length
def one_node(message: str, array: list):
    pass
    # print(f"{message} single node, len: {len(array[0])}, node: {array[0]}\n")


# one_node("input layer", [x for x in snake.brain.parameters()][0])

one_node("2nd layer", [x for x in snake.brain.parameters()][1])
one_node("2nd layer mutated", [
         x for x in geneticAlgorithm.mutate(snake.brain).parameters()][1])

print(snake == Snake(width, height, red, SnakeAI()))
