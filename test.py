import random

import pygame

from perceptron import Perceptron

WINDOW_SIZE = (800, 800)
RED = (200, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 50, 200)
GREY = (30, 30, 30)
WHITE = (255, 255, 255)

running = True


def activation(x):
    if x > 0:
        return 1
    else:
        return -1


class Point:
    def __init__(self):
        self.outline_color = WHITE
        self.x = random.randint(0, WINDOW_SIZE[0])
        self.y = random.randint(0, WINDOW_SIZE[1])
        if self.x > self.y:
            self.label = 1
            self.color = RED
        else:
            self.label = -1
            self.color = BLUE

    def show(self):
        pygame.draw.circle(screen, self.outline_color, (self.x, self.y), 10)
        if self.label == 1:
            color = RED
        else:
            color = BLUE
        pygame.draw.circle(screen, color, (self.x, self.y), 5)

    def get_coord(self):
        return [self.x, self.y]


if __name__ == '__main__':
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 24, True)
    pygame.display.set_caption("Handwriting Classifier")
    pygame.display.set_caption("Linear Classifier")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()

    # SETUP
    points = []
    for i in range(150):
        points.append(Point())
    test = Perceptron(2, 0.01, 1, activation)
    # SETUP ENDS

    while running:
        # UPDATE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for i in range(len(points)):
            points[i].outline_color = WHITE
            if test.feed_forward(points[i].get_coord()) == points[i].label:
                points[i].outline_color = GREEN

        for i in range(10):
            p = Point()
            test.train(p.get_coord(), p.label)


        # print(test.weights)
        w0, w1, w2 = test.get_weights()
        w0_text = font.render('W1 = {:.4f}'.format(w0), True, WHITE)
        w1_text = font.render('W2 = {:.4f}'.format(w1), True, WHITE)
        # UPDATE ENDS

        screen.fill(GREY)
        # DRAW
        pygame.draw.line(screen, WHITE, (0, 0), (WINDOW_SIZE[0], WINDOW_SIZE[1]), 3)

        if w1 != 0:
            pygame.draw.line(screen, GREEN, (0, -((w0 / w1 * 0) - w2 / w1)), (WINDOW_SIZE[0], -((w0 / w1 * WINDOW_SIZE[0]) - w2 / w1)), 3)

        for i in range(len(points)):
            points[i].show()

        screen.blit(w0_text, (0, 0))
        screen.blit(w1_text, (0, 25))

        # DRAW ENDS
        pygame.display.flip()

    pygame.quit()
