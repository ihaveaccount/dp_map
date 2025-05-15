import pygame
import sys
import os
import math
import datetime
import argparse
from typing import List, Tuple

# Конфигурационные константы
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
BACKGROUND_COLOR = (0, 0, 0)
PENDULUM_COLOR = (255, 255, 255)

# Параметры маятника
PENDULUM_COUNT = 5  # Количество маятников
L1 = 200.0  # Длина первого стержня (пиксели)
L2 = 200.0  # Длина второго стержня (пиксели)
M1 = 20.0  # Масса первого шара
M2 = 20.0  # Масса второго шара
G = 9.8  # Гравитация

# Начальные углы (в градусах) и скорости для каждого маятника
# Каждый элемент - это (angle1, angle2, velocity1, velocity2)
INITIAL_CONDITIONS = [
    (math.radians(90 + i * 0.0005), math.radians(45 + i * 0.0005), 0.0, 0.0)
    for i in range(PENDULUM_COUNT)
]

# Настройки записи кадров
RECORD_FRAMES = False  # По умолчанию не записываем кадры
FRAME_RATE = 60  # Частота кадров
FRAME_SKIP = 1  # Записывать каждый N-й кадр (для экономии места)

class DoublePendulum:
    def __init__(self, l1: float, l2: float, m1: float, m2: float, 
                 initial_angles: Tuple[float, float, float, float]):
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        
        # Углы в радианах и угловые скорости
        self.angle1, self.angle2, self.velocity1, self.velocity2 = initial_angles
        
        # Позиции шаров
        self.x1, self.y1 = 0.0, 0.0
        self.x2, self.y2 = 0.0, 0.0
        
        self.update_positions()
    
    def update(self, dt: float):
        # Вычисляем ускорения с помощью уравнений Лагранжа
        angle_diff = self.angle1 - self.angle2
        
        # Для оптимизации вычислений
        sin_diff = math.sin(angle_diff)
        cos_diff = math.cos(angle_diff)
        sin1 = math.sin(self.angle1)
        sin2 = math.sin(self.angle2)
        den1 = (2 * self.m1 + self.m2 - self.m2 * math.cos(2 * angle_diff))
        
        # Угловое ускорение первого стержня
        acceleration1 = (
            -G * (2 * self.m1 + self.m2) * sin1 
            - self.m2 * G * sin1 
            - 2 * sin_diff * self.m2 
            * (self.velocity2**2 * self.l2 + self.velocity1**2 * self.l1 * cos_diff)
        ) / (self.l1 * den1)
        
        # Угловое ускорение второго стержня
        acceleration2 = (
            2 * sin_diff * (
                self.velocity1**2 * self.l1 * (self.m1 + self.m2) 
                + G * (self.m1 + self.m2) * math.cos(self.angle1) 
                + self.velocity2**2 * self.l2 * self.m2 * cos_diff
            )
        ) / (self.l2 * den1)
        
        # Обновляем скорости и углы
        self.velocity1 += acceleration1 * dt
        self.velocity2 += acceleration2 * dt
        self.angle1 += self.velocity1 * dt
        self.angle2 += self.velocity2 * dt
        
        self.update_positions()
    
    def update_positions(self):
        # Вычисляем позиции шаров
        self.x1 = self.l1 * math.sin(self.angle1)
        self.y1 = self.l1 * math.cos(self.angle1)
        self.x2 = self.x1 + self.l2 * math.sin(self.angle2)
        self.y2 = self.y1 + self.l2 * math.cos(self.angle2)
    
    def get_positions(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # Возвращаем позиции в экранных координатах
        return (
            (CENTER[0] + self.x1, CENTER[1] + self.y1),
            (CENTER[0] + self.x2, CENTER[1] + self.y2)
        )

def create_output_dir() -> str:
    """Создает папку для сохранения кадров с текущей датой и временем"""
    now = datetime.datetime.now()
    dir_name = f"frames/single_{now.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def main():
    parser = argparse.ArgumentParser(description='Double Pendulum Simulation')
    parser.add_argument('--record', action='store_true', 
                       help='Enable frame recording to frames/ directory')
    args = parser.parse_args()
    
    global RECORD_FRAMES
    RECORD_FRAMES = args.record
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Double Pendulum Simulation")
    clock = pygame.time.Clock()
    
    # Создаем маятники
    pendulums = [
        DoublePendulum(L1, L2, M1, M2, init_cond) 
        for init_cond in INITIAL_CONDITIONS
    ]
    
    output_dir = None
    frame_count = 0
    
    if RECORD_FRAMES:
        output_dir = create_output_dir()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Обновляем маятники
        dt = 10 / FRAME_RATE
        for pendulum in pendulums:
            pendulum.update(dt)
        
        # Отрисовка
        screen.fill(BACKGROUND_COLOR)
        
        hand_width = 5
        weight_radius = 10

        # Рисуем все маятники
        for pendulum in pendulums:
            pos1, pos2 = pendulum.get_positions()
            
            # Рисуем стержни
            pygame.draw.line(screen, PENDULUM_COLOR, CENTER, pos1, hand_width)
            pygame.draw.line(screen, PENDULUM_COLOR, pos1, pos2, hand_width)
            
            # Рисуем шары
            pygame.draw.circle(screen, PENDULUM_COLOR, (int(pos1[0]), int(pos1[1])), weight_radius)
            pygame.draw.circle(screen, PENDULUM_COLOR, (int(pos2[0]), int(pos2[1])), weight_radius)
        
        pygame.display.flip()
        
        # Сохранение кадра, если нужно
        if RECORD_FRAMES and frame_count % FRAME_SKIP == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
            pygame.image.save(screen, frame_path)
        
        frame_count += 1
        clock.tick(FRAME_RATE)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()