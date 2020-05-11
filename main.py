import pygame
import random
import torch
import torch.nn as nn
import os
import numpy as np
import math
import copy

clock = pygame.time.Clock()
clock = pygame.time.Clock()
pygame.init()
red = (255, 0, 0)
green = (0, 255, 0)
yellow = (255, 255, 102)
width = 800
height = 600
speed = 2000000
screen = pygame.display.set_mode((width, height))
yellow = (255, 255, 102)
population = 200
snakes = []
top_limit = 3
gen = 0
human = False
learning = True
if human:
    learning = False
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
    def __init__(self):
        self.x = int(round(random.randrange(0, width - 10) / 10.0) * 10.0)
        self.y = int(round(random.randrange(0, height - 10) / 10.0) * 10.0)

    def draw(self):
        pygame.draw.rect(screen, green, [self.x, self.y, 10, 10])


def food_collide(pos_x, pos_y, fx, fy):
    if pos_x == fx and pos_y == fy:
        return True
    return False


class Snake:
    def __init__(self):
        self.block = 10
        # self.x = int(round(random.randrange(0, height - self.block) / 10.0) * 10.0)
        # self.y = int(round(random.randrange(0, height - self.block) / 10.0) * 10.0)
        self.x = 450
        self.y = 230
        self.dx = 0
        self.dy = 0
        self.head = []
        self.full = []
        self.length = 8
        self.direction = 4
        self.steps_allowed = 500
        self.life_time = 0
        self.score = 0
        self.fitness = 0
        self.brain = SnakeAI()
        self.color = red
        for param in self.brain.parameters():
            param.requires_grad = False

    def draw(self, color):
        for x in self.full:
            pygame.draw.rect(screen, self.color, (x[0], x[1], self.block, self.block))

    def move(self):
        # print(self.direction)
        # print(type(self.direction))
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
        self.life_time += 1
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
        self.length += 1

    def is_on_tail(self):
        for x in self.full:
            if self.x == x[0] + self.block and self.y == x[1] + self.block:
                return True
        return False

    def gonna_die(self, x, y):
        if x >= width or x <= 0 - self.block or y >= height or y <= 0 - self.block:
            return True
        return self.is_on_tail()

    def body_collide(self, pos_x, pos_y):
        for x in self.full:
            if pos_x == x[0] and pos_y == x[1]:
                return True
        return False

    def look_in_direction(self, x, y, fx, fy):
        look = [0, 0, 0]
        pos_x, pos_y = self.x + x, self.y + y
        distance = 0
        food_found = False
        body_found = False
        distance += 1
        while not self.gonna_die(pos_x, pos_y):
            if not food_found and food_collide(pos_x, pos_y, fx, fy):
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

    def look(self, fx, fy):
        vision = [0 for x in range(24)]
        temp = self.look_in_direction(-self.block, 0, fx, fy)
        vision[0] = temp[0]
        vision[1] = temp[1]
        vision[2] = temp[2]

        temp = self.look_in_direction(-self.block, -self.block, fx, fy)
        vision[3] = temp[0]
        vision[4] = temp[1]
        vision[5] = temp[2]

        temp = self.look_in_direction(0, -self.block, fx, fy)
        vision[6] = temp[0]
        vision[7] = temp[1]
        vision[8] = temp[2]

        temp = self.look_in_direction(self.block, -self.block, fx, fy)
        vision[9] = temp[0]
        vision[10] = temp[1]
        vision[11] = temp[2]

        temp = self.look_in_direction(self.block, 0, fx, fy)
        vision[12] = temp[0]
        vision[13] = temp[1]
        vision[14] = temp[2]

        temp = self.look_in_direction(self.block, self.block, fx, fy)
        vision[15] = temp[0]
        vision[16] = temp[1]
        vision[17] = temp[2]

        temp = self.look_in_direction(0, self.block, fx, fy)
        vision[18] = temp[0]
        vision[19] = temp[1]
        vision[20] = temp[2]

        temp = self.look_in_direction(-self.block, self.block, fx, fy)
        vision[21] = temp[0]
        vision[22] = temp[1]
        vision[23] = temp[2]

        return vision

    def think(self, fx, fy):
        observation = self.look(fx, fy)
        inp = torch.tensor(observation).type('torch.FloatTensor').view(1, -1)
        output_probabilities = self.brain(inp).detach().numpy()[0]
        self.direction = np.random.choice(range(4), 1, p=output_probabilities).item()


def select_top(previous):
    rewards = [snake.fitness for snake in previous]
    sorted_parent_indexes = np.argsort(rewards)[::-1][
                            :top_limit]
    return sorted_parent_indexes


def mutate(DNA):
    baby = copy.deepcopy(DNA)
    mutation_power = 0.15  # hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf

    for param in baby.parameters():

        if len(param.shape) == 2:  # weights of linear layer
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    param[i0][i1] += mutation_power * np.random.randn()

        elif len(param.shape) == 1:  # biases of linear layer or conv layer
            for i0 in range(param.shape[0]):
                param[i0] += mutation_power * np.random.randn()

    return baby


def make_babies(DNA):
    babies = []
    for i in range(population - 1):
        babies.append(mutate(DNA))
    return babies


def draw(snake, food):
    screen.fill((0, 0, 0))
    your_score(snake.score)
    message("Gen : " + str(gen), yellow, 5, 30)
    food.draw()
    snake.move()
    snake.draw(red)
    # clock.tick(20)
    pygame.display.update()


def run_for_youtube(brain):
    global gen
    dead = False
    top_snake = Snake()
    top_snake.brain = brain
    food = Food()
    gen += 1
    while not dead:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                dead = False
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if human:
                    if event.key == pygame.K_LEFT:
                        snake.direction = 2
                    elif event.key == pygame.K_RIGHT:
                        snake.direction = 3
                    elif event.key == pygame.K_UP:
                        snake.direction = 0
                    elif event.key == pygame.K_DOWN:
                        snake.direction = 1
        top_snake.think(food.x, food.y)
        draw(top_snake, food)
        if top_snake.steps_allowed <= 0:
            print("out of steps")
            dead = True
        if top_snake.x >= width or top_snake.x <= 0 or top_snake.y >= height or top_snake.y <= 0:
            dead = True
            print("out")
        for y in top_snake.full[:-1]:
            if y == top_snake.head:
                dead = True
                print("eaten it self")
        if top_snake.x == food.x and top_snake.y == food.y:
            top_snake.score += 1
            top_snake.steps_allowed += 200
            food = Food()
            top_snake.grow()


def mate(parents):
    DNA = SnakeAI()
    for param in DNA.parameters():
        x = random.choice(parents)
        y = random.choice(parents)
        while x is y:
            y = random.choice(parents)
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
    # draw(parents[0], Food())
    for parent in parents:
        print(parent.fitness)
    babies.append(parents[0].brain)
    run_for_youtube(parents[0].brain)
    return babies


def next_generation(previous):
    parents = []
    sum = 0
    top = select_top(previous)
    for x in top:
        sum += previous[x].fitness
        parents.append(previous[x])
    babies = mate(parents)
    print("average is ", sum / top_limit)
    return babies


def run():
    is_game_in_process = True
    for_next = []
    food = Food()
    while is_game_in_process and len(snakes) > -1:
        for x, snake in enumerate(snakes):
            snake.brain.eval()
            if learning:
                snake.think(food.x, food.y)
            snake.move()
            snake.life_time += 1
            snake.fitness = 20 * snake.length + 5 * snake.life_time
            if snake.steps_allowed <= 0:
                for_next.append(snake)
                snakes.pop(x)
            if snake.x >= width or snake.x <= 0 or snake.y >= height or snake.y <= 0:
                for_next.append(snake)
                snakes.pop(x)
            for y in snake.full[:-1]:
                if y == snake.head:
                    for_next.append(snake)
                    snakes.pop(x)
            if snake.x == food.x and snake.y == food.y:
                snake.score += 1
                snake.steps_allowed += 200
                food = Food()
                snake.grow()
            if len(snakes) is 0:
                for new_brain in next_generation(for_next):
                    new_snake = Snake()
                    new_snake.brain = new_brain
                    snakes.append(new_snake)
                for x, snake in enumerate(snakes):
                    print(x, snake)
                food = Food()
                for_next = []


torch.set_grad_enabled(False)
if learning:
    for snake in range(population):
        snake = Snake()
        snakes.append(snake)
run()
