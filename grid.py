import pygame
import sys
import os
import math
import numpy as np
from pygame.locals import *
from pygame import gfxdraw
import sys
import common
from common import circle
# Параметры по умолчанию
WIDTH, HEIGHT = 1920, 1080
BG_COLOR = (0, 0, 0)
PENDULUM_COLOR = (255, 255, 255)  # Все маятники красные
PENDULUM_RADIUS = 1
SEGMENT_WIDTH = 1
NHOR = 50  # количество маятников по горизонтали
HVER = 28  # количество маятников по вертикали
IS_SAVE = False  # сохранять кадры
# Параметры маятника


if len(sys.argv) > 1:
  IS_SAVE = True
else:
  IS_SAVE = False


L1, L2 = 1.0, 1.0  # длины сегментов
M1, M2 = 1.0, 1.0  # массы
G = 9.8             # ускорение свободного падения
DT = 0.1           # шаг времени
MAX_ITERATIONS = 2000



class DoublePendulum:
    def __init__(self, theta1, theta2):
        self.theta1 = theta1  # угол первого сегмента
        self.theta2 = theta2  # угол второго сегмента
        self.omega1 = 0.0     # угловая скорость первого сегмента
        self.omega2 = 0.0     # угловая скорость второго сегмента
        self.stopped = False  # флаг остановки анимации
        self.iteration = 0    # счетчик итераций
        
    def update(self):
        if self.stopped:
            return
            
        # Уравнения движения двойного маятника
        delta = self.theta2 - self.theta1
        den1 = (M1 + M2) * L1 - M2 * L1 * math.cos(delta) * math.cos(delta)
        den2 = (L2 / L1) * den1
        
        # Производные угловых скоростей
        domega1 = (M2 * L2 * self.omega2 * self.omega2 * math.sin(delta) * math.cos(delta) +
                   M2 * G * math.sin(self.theta2) * math.cos(delta) +
                   M2 * L2 * self.omega2 * self.omega2 * math.sin(delta) -
                   (M1 + M2) * G * math.sin(self.theta1)) / den1
        
        domega2 = ((-M2 * L2 * self.omega2 * self.omega2 * math.sin(delta) * math.cos(delta) +
                   (M1 + M2) * G * math.sin(self.theta1) * math.cos(delta) -
                   (M1 + M2) * L1 * self.omega1 * self.omega1 * math.sin(delta) -
                   (M1 + M2) * G * math.sin(self.theta2))) / den2
        
        # Обновление углов и скоростей
        self.omega1 += domega1 * DT
        self.omega2 += domega2 * DT
        self.theta1 += self.omega1 * DT
        self.theta2 += self.omega2 * DT
        
        # Проверка на остановку
        if abs(self.theta1) > 2 * math.pi or self.iteration >= MAX_ITERATIONS:
            self.stopped = True
        self.iteration += 1
    
    def get_positions(self, scale, offset_x, offset_y):
        x1 = L1 * math.sin(self.theta1) * scale + offset_x
        y1 = L1 * math.cos(self.theta1) * scale + offset_y
        x2 = x1 + L2 * math.sin(self.theta2) * scale
        y2 = y1 + L2 * math.cos(self.theta2) * scale
        return (offset_x, offset_y), (x1, y1), (x2, y2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Двойные маятники")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    
    # Настройки интерфейса
    n_horizontal = NHOR  # маятников по горизонтали
    n_vertical = HVER    # маятников по вертикали
    save_frames = IS_SAVE
    running = True
    paused = True
    
    modx = 1.78
    mody = modx * (HEIGHT / WIDTH)
    
    minx = -math.pi*modx
    maxx = math.pi*modx
    miny = -math.pi*mody
    maxy = math.pi*mody

    # Создание маятников
    pendulums = []
    def create_pendulums():
        nonlocal pendulums
        pendulums = []
        for i in range(n_vertical):
            row = []
            for j in range(n_horizontal):
                theta1 = minx + (2 * maxx) * j / (n_horizontal - 1) if n_horizontal > 1 else 0
                theta2 = -miny + (2 * maxy) * i / (n_vertical - 1) if n_vertical > 1 else 0
                row.append(DoublePendulum(theta1, theta2))
            pendulums.append(row)
    
    create_pendulums()
    
    # Создание папки для сохранения кадров
    if save_frames and not os.path.exists('pend_anim'):
        os.makedirs('pend_anim')
    frame_count = 0
    
    # Основной цикл
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_SPACE:
                    paused = not paused
                elif event.key == K_s:
                    save_frames = not save_frames
                    if save_frames and not os.path.exists('pend_anim'):
                        os.makedirs('pend_anim')
                elif event.key == K_r:
                    create_pendulums()
                    frame_count = 0
                elif event.key == K_UP and n_vertical < 500:
                    n_vertical *= 2
                    n_horizontal *= 2
                    create_pendulums()
                elif event.key == K_DOWN and n_vertical > 1:
                    n_vertical = round(n_vertical/2)
                    n_horizontal = round(n_horizontal/2)
                    create_pendulums()

        
        if not paused:
            # Обновление маятников
            for row in pendulums:
                for pendulum in row:
                    pendulum.update()
        
        # Отрисовка
        screen.fill(BG_COLOR)
        

        
        # Расчет размера ячейки для маятника
        cell_width = WIDTH // n_horizontal
        cell_height = HEIGHT // n_vertical
        scale = min(cell_width, cell_height) * 0.2
        
        # Отрисовка маятников
        for i, row in enumerate(pendulums):
            for j, pendulum in enumerate(row):
                offset_x = j * cell_width + cell_width // 2
                offset_y = i * cell_height + cell_height // 2
                
                # Цвет фона ячейки в зависимости от количества итераций
                bg_intensity = min(255, round(pendulum.iteration * 255 / (MAX_ITERATIONS/4)))
                cell_bg_color = (bg_intensity, bg_intensity, bg_intensity)
                #cell_rect = pygame.Rect(j * cell_width, i * cell_height + 50, cell_width, cell_height)
                
                circle(screen,cell_bg_color, offset_x, offset_y, round(cell_width / 2))

                # pygame.draw.rect(screen, cell_bg_color, cell_rect)
                
                # Получение позиций сегментов
                pivot, bob1, bob2 = pendulum.get_positions(scale, offset_x, offset_y)
                
                if not pendulum.stopped:    
                    # Отрисовка сегментов
                    pygame.draw.aaline(screen, PENDULUM_COLOR, pivot, bob1, SEGMENT_WIDTH)
                    pygame.draw.aaline(screen, PENDULUM_COLOR, bob1, bob2, SEGMENT_WIDTH)
                    # Отрисовка сегментов
                    pygame.draw.aaline(screen, PENDULUM_COLOR, pivot, bob1, SEGMENT_WIDTH)
                    pygame.draw.aaline(screen, PENDULUM_COLOR, bob1, bob2, SEGMENT_WIDTH)
                
                    # Отрисовка точек
                    circle(screen, PENDULUM_COLOR, int(pivot[0]), int(pivot[1]), PENDULUM_RADIUS)
                    circle(screen, PENDULUM_COLOR, int(bob1[0]), int(bob1[1]), PENDULUM_RADIUS)
                    circle(screen, PENDULUM_COLOR, int(bob2[0]), int(bob2[1]), PENDULUM_RADIUS*2)
                
                # Отображение начальных углов
                if False and pendulum.iteration == 0:
                    theta1_text = f"θ1: {pendulum.theta1:.2f}"
                    theta2_text = f"θ2: {pendulum.theta2:.2f}"
                    text1 = font.render(theta1_text, True, (200, 200, 200))
                    text2 = font.render(theta2_text, True, (200, 200, 200))
                    screen.blit(text1, (offset_x - 50, offset_y - 30))
                    screen.blit(text2, (offset_x - 50, offset_y - 10))
        
        pygame.display.flip()
        clock.tick(60)
        
        # Сохранение кадра
        if save_frames and not paused:
            pygame.image.save(screen, f"frames/grid/frame_{frame_count:05d}.png")
            frame_count += 1
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()