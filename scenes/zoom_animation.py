import pygame
import sys
import os
import glob
import time
import math
import argparse

def main():
    parser = argparse.ArgumentParser(description='Zoom animation on image click')
    parser.add_argument('input', help='Input image or folder containing frame*.png images')
    parser.add_argument('--pixel-threshold', type=float, default=5.0, 
                       help='Pixel size threshold for switching to grid mode (default: 5)')
    parser.add_argument('--save', nargs='?', const='', default=None,
                       help='Enable saving frames. For folder input, saves in input folder. '
                       'For image input, saves in specified directory or image directory if not specified.')
    args = parser.parse_args()

    # Определение пути к изображению
    if os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, 'frame*.[pP][nN][gG]')) + \
                glob.glob(os.path.join(args.input, 'frame*.[jJ][pP][gG]')) + \
                glob.glob(os.path.join(args.input, 'frame*.[jJ][pP][eE][gG]'))
        if not files:
            print(f'No frame* images found in directory: {args.input}')
            return
        files.sort()
        image_path = files[-1]
        output_dir = args.input
    else:
        image_path = args.input
        if args.save is not None:
            output_dir = os.path.dirname(image_path) if args.save == '' else args.save
        else:
            output_dir = None
    save_frames = args.save is not None

    # Инициализация Pygame
    pygame.init()
    source_image = pygame.image.load(image_path)
    img_width, img_height = source_image.get_size()
    screen = pygame.display.set_mode((img_width, img_height))
    pygame.display.set_caption('Zoom Animation - Click to zoom or press SPACE for center zoom')
    clock = pygame.time.Clock()

    # Отображение исходного изображения
    screen.blit(source_image, (0, 0))
    pygame.display.flip()

    # Ожидание клика или нажатия пробела
    waiting = True
    click_pos = None
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                click_pos = event.pos
                waiting = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                elif event.key == pygame.K_SPACE:
                    # Запуск анимации в центр изображения
                    click_pos = (img_width // 2, img_height // 2)
                    waiting = False

    # Параметры анимации
    t0 = time.time()
    growth_rate = 0.17  # Скорость увеличения
    frame_count = 0
    pixel_threshold = args.pixel_threshold
    # Центр зума - точка клика (дробные координаты)
    zoom_center = (click_pos[0] + 0.5, click_pos[1] + 0.5)
    screen_center = (img_width / 2, img_height / 2)

    # Основной цикл анимации
    running = True
    while running:
        current_time = time.time() - t0
        current_scale = math.exp(growth_rate * current_time)
        
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        if not running:
            break
        
        screen.fill((0, 0, 0))  # Черный фон
        
        if current_scale < pixel_threshold:
            # Этап 1: Плавное масштабирование с фиксированным центром
            
            # Размер видимой области в исходных пикселях
            visible_width = img_width / current_scale
            visible_height = img_height / current_scale
            
            # Координаты области для выборки (дробные)
            view_x = zoom_center[0] - visible_width / 2
            view_y = zoom_center[1] - visible_height / 2
            view_w = visible_width
            view_h = visible_height
            
            # Вычисляем целочисленные границы для выборки
            src_x = max(0, math.floor(view_x))
            src_y = max(0, math.floor(view_y))
            src_w = min(img_width - src_x, math.ceil(view_w + (view_x - src_x) + 1))
            src_h = min(img_height - src_y, math.ceil(view_h + (view_y - src_y) + 1))
            
            if src_w <= 0 or src_h <= 0:
                continue
            
            # Создаем временную поверхность для выбранной области
            temp_surf = pygame.Surface((src_w, src_h))
            temp_surf.blit(source_image, (0, 0), (src_x, src_y, src_w, src_h))
            
            # Масштабируем область
            scaled_w = int(src_w * current_scale)
            scaled_h = int(src_h * current_scale)
            if scaled_w == 0 or scaled_h == 0:
                continue
                
            try:
                scaled_surf = pygame.transform.scale(
                    temp_surf, (scaled_w, scaled_h), 
                    interpolation=pygame.NEAREST
                )
            except:
                scaled_surf = pygame.transform.scale(temp_surf, (scaled_w, scaled_h))
            
            # Вычисляем смещение для точного позиционирования
            offset_x = (view_x - src_x) * current_scale
            offset_y = (view_y - src_y) * current_scale
            
            # Рисуем масштабированное изображение с центром в точке клика
            screen.blit(scaled_surf, (-offset_x, -offset_y))
            
        else:
            # Этап 2: Сетка пикселей
            pixel_size = pixel_threshold
            spacing = current_scale - pixel_threshold
            
            # Рассчет видимой области
            visible_width = img_width / current_scale
            visible_height = img_height / current_scale
            
            # Границы для отрисовки (в исходных координатах)
            j_min = max(0, math.floor(zoom_center[0] - visible_width / 2))
            j_max = min(img_width, math.ceil(zoom_center[0] + visible_width / 2))
            i_min = max(0, math.floor(zoom_center[1] - visible_height / 2))
            i_max = min(img_height, math.ceil(zoom_center[1] + visible_height / 2))
            
            # Отрисовка пикселей
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    # Позиция центра пикселя на экране
                    pos_x = screen_center[0] + (j - zoom_center[0]) * current_scale
                    pos_y = screen_center[1] + (i - zoom_center[1]) * current_scale
                    
                    # Прямоугольник для отрисовки пикселя
                    pixel_rect = [
                        pos_x - pixel_size / 2,
                        pos_y - pixel_size / 2,
                        pixel_size,
                        pixel_size
                    ]
                    
                    # Получаем цвет пикселя и рисуем
                    color = source_image.get_at((j, i))
                    pygame.draw.rect(screen, color, pixel_rect)
        
        pygame.display.flip()
        
        # Сохранение кадров
        if save_frames and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            pygame.image.save(screen, os.path.join(output_dir, f'zoom_{frame_count:05d}.png'))
            frame_count += 1
        
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()