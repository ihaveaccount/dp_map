import pygame
import numpy as np
import os
import datetime
import math
import argparse 


import Metal as metal
from dp_lib import *
import time
from PIL import Image

parser = argparse.ArgumentParser(description='Double Pendulum Simulation')
parser.add_argument('--anim', action='store_true', 
                    help='Run predefined animation')
parser.add_argument('--height', type=int, default=1080,
                    help='Height of the window (default: 1080)')
parser.add_argument('--frames', type=int, default=3600,help='Number of frames for animation')

parser.add_argument('--start', type=int, default=0,help='Start frame for animation') 
parser.add_argument('--folder', type=str, default="",help='Folder to save frames')
parser.add_argument('--pfile', type=str, default="",help='File with target point coordinates')
parser.add_argument('--backend', type=str, default="opencl", choices=["metal", "opencl"],
                    help='Backend to use for computation (default: opencl)')


args = parser.parse_args()


if(args.backend == "metal"):
    device = metal.MTLCreateSystemDefaultDevice()

    if not device:
        print("Metal is not supported on this device")
    else:
        print(f"Default Metal device: {device.name()}")


wid = round(args.height*1.777);

if wid %2 != 0: wid += 1

# --- Конфигурация ---
WIDTH, HEIGHT = wid, args.height

ANIMATION_FRAMES = 1 # Количество кадров для анимации зума/изменения параметров
ANIMATION_FRAMES_OUT = 2   # Количество кадров для анимации выхода из зума/изменения параметров
ZOOM_FACTOR = 0.1       # Коэффициент зума при клике мыши  

modx = 1.78
mody = modx * (HEIGHT / WIDTH)

theta1_min = -math.pi*modx
theta1_max = math.pi*modx
theta2_min = -math.pi*mody
theta2_max = math.pi*mody






def save_to_file(normalized_2d_data):
    global frame_counter, frames_dir
    """ Сохраняет 2D нормализованные данные в PNG файл. """
    # Импортируем PIL вместо pygame

    # Создаем копию данных, чтобы не изменять оригинал
    # и преобразуем значения в диапазон 0-255 для изображения
    img_data = normalized_2d_data.copy()
    
    # Если данные еще не нормализованы до 0-255, нужно их преобразовать
    if img_data.max() <= 1.0:
        img_data = (img_data * 255).astype(np.uint8)
    else:
        img_data = img_data.astype(np.uint8)
    
    # Создаем RGB изображение
    height, width = img_data.shape
    rgb_data = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_data[:, :, 0] = img_data  # R
    rgb_data[:, :, 1] = img_data  # G
    rgb_data[:, :, 2] = img_data  # B
    
    
    # Создаем изображение из массива и сохраняем в файл
    img = Image.fromarray(rgb_data)

    filename = os.path.join(frames_dir, f"frame_{frame_counter:05d}.png")

    img.save(filename)
    
    frame_counter += 1
    #return output_path


def find_max_frame_number(directory):
    folder = pathlib.Path(directory)
    max_num = -1
    pattern = re.compile(r'^frame_(\d+)\.png$')
    
    for file in folder.glob('*.png'):
        match = pattern.match(file.name)
        if match:
            current_num = int(match.group(1))
            max_num = max(max_num, current_num)
    
    return max_num if max_num != -1 else 0




def write_target_point_to_file(filename, coords):
    """Записывает координаты точки в файл."""
    try:
        with open(filename, 'w') as file:
            file.write(' '.join(map(str, coords)))    
    except Exception as e:  
        print(f"Error writing target point to file: {e}")


def read_target_point_from_file(filename):
    """Читает координаты точки из файла."""
    try:        
        with open(filename, 'r') as file:
            content = file.read().split(' ') 
            
            coords = list(map(float, content))
            print(coords)
            return coords
    except Exception as e:
        print(f"Error reading target point from file: {e}")
        return None



def create_surface_from_normalized_data(normalized_2d_data):

    image_data_rgb = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    """ Создает Pygame Surface из 2D нормализованных данных (яркость). """
    image_data_rgb[:, :, 0] = normalized_2d_data
    image_data_rgb[:, :, 1] = normalized_2d_data
    image_data_rgb[:, :, 2] = normalized_2d_data
    # Pygame ожидает (width, height, channels)
    return pygame.surfarray.make_surface(image_data_rgb.transpose(1,0,2))




def process_normalized_data(normalized_2d_data):
    
    if args.anim:
        save_to_file(normalized_2d_data)
    else:
        # Создаем Pygame Surface из нормализованных данных
        surface = create_surface_from_normalized_data(normalized_2d_data)
        return surface
    
    
    




 

# --- Глобальные переменные для анимации и сохранения ---

frames_dir = ""
animation_queue = [] # [(type, data, current_step, total_steps, prev_data_norm)]
# type: 'zoom', 'params_change'
# data for zoom: (target_x_min, target_x_max, target_y_min, target_y_max)
# data for param_change: new_pendulum_params_tuple (не используется напрямую, т.к. pendulum уже обновлен)
# prev_data_norm: предыдущие нормализованные данные для блендинга при param_change

def setup_frame_saving():
    global frame_counter, frames_dir
    
    base_dir = "frames"
    
    if(args.folder == ""):
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        frames_dir = os.path.join(base_dir, f"run_{HEIGHT}_{run_timestamp}")
    else:
        frames_dir = os.path.join(base_dir, args.folder)

    if not args.pfile:
        args.pfile = os.path.join(frames_dir, "point.txt")

    if(args.start > 0):
        frame_counter = args.start
    else:
        # Ищем максимальный номер кадра в папке
        max_frame_number = find_max_frame_number(frames_dir)

        if max_frame_number > 0:
            # Если есть уже сохраненные кадры, начинаем с max_frame_number + 1
            frame_counter = max_frame_number + 1
            print(f"Starting from frame {frame_counter}")
        else:
            # Если нет сохраненных кадров, начинаем с 0
            frame_counter = 0
        
    

    os.makedirs(frames_dir, exist_ok=True)
    print(f"Saving frames to: {frames_dir}")

def save_current_frame(surface):
    global frame_counter
    if not frames_dir:
        setup_frame_saving()
    
    
    filename = os.path.join(frames_dir, f"frame_{frame_counter:05d}.png")
    pygame.image.save(surface, filename)
    frame_counter += 1




def main():
    global animation_queue,frame_counter # Разрешаем модификацию

    if not args.anim:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Double Pendulum Map (OpenCL)")
        clock = pygame.time.Clock()


    if args.anim and not args.folder:
        print("Error: --folder argument is required for animation.")
    
        return
                

    pendulum = DoublePendulum(L1=1.0, 
                              L2=1.0, 
                              M1=1.0, 
                              M2=1.0, 
                              G=9.81, 
                              DT=0.2, 
                              MAX_ITER=5000, 
                              theta1_min=theta1_min, theta1_max=theta1_max, theta2_min=theta2_min, theta2_max=theta2_max)
    
    
    
    # Use the backend selection function to create the appropriate mapper
    mapper = create_mapper(WIDTH, HEIGHT, pendulum, args.backend)
    
    setup_frame_saving() # Создаем папку для кадров при запуске

    # Для анимации изменения параметров
    current_normalized_data = None 
    (x_min, x_max), (y_min, y_max) = pendulum.get_current_view_ranges()

    if(frame_counter == 0):
        # Первоначальный расчет и отрисовка
        print("Performing initial calculation...")
        
        raw_data = mapper.compute_map_raw(x_min, x_max, y_min, y_max)
        current_normalized_data = mapper.normalize_data(raw_data)
        

        if args.anim:
            save_to_file(current_normalized_data)            
        else:
            current_surface = create_surface_from_normalized_data(current_normalized_data)


      
  

 



    if args.anim:

        if(args.pfile != ""):
                # Читаем координаты точки из файла
            coords = read_target_point_from_file(args.pfile)
            if coords:
                target_point = coords
                print(f"Target point from file: {args.pfile}")
            else:
                print("Error reading target point from file, using default.")


        animation_queue.append({'type': 'zoom','start_view': (x_min, x_max, y_min, y_max),
                            'target_view': target_point,
                            'step': frame_counter, 'total_steps': args.frames})
        


    running = True
    while running:

        # --- Логика обновления и анимации ---
        if animation_queue:
            anim = animation_queue[0]
 
            anim['step'] += 1
            t = anim['step'] / anim['total_steps']
          

            if anim['type'] == 'zoom':
 
                # Начальные и целевые границы
                
                target_vx_min, target_vx_max, target_vy_min, target_vy_max = anim['target_view']
                interp_x_min, interp_x_max, interp_y_min, interp_y_max = interpolate(anim)



                # Начинаем отсчет времени
                start_time = time.time()

 
                

                # Пересчет карты для текущего кадра анимации
                # Этот блок может быть медленным, если ANIMATION_FRAMES большое
                raw_data_anim = mapper.compute_map_raw(interp_x_min, interp_x_max, interp_y_min, interp_y_max)
                current_normalized_data = mapper.normalize_data(raw_data_anim) # Обновляем глобальные данные
                
                if not args.anim:
                    current_surface = create_surface_from_normalized_data(current_normalized_data)
                
                if anim['step'] >= anim['total_steps']:
                    pendulum.set_current_view_ranges(target_vx_min, target_vx_max, target_vy_min, target_vy_max)
                    pendulum.print_params() # Вывести финальные параметры вида
                    animation_queue.pop(0)
            
            # elif anim['type'] == 'param_change':
            #     prev_data = anim['prev_data_norm']
            #     target_data = anim['target_data_norm']
                
            #     # Блендинг между старым и новым изображением
            #     blended_data = ((1 - t) * prev_data + t * target_data).astype(np.uint8)
            #     current_surface = create_surface_from_normalized_data(blended_data)
                
            #     if anim['step'] >= anim['total_steps']:
            #         current_normalized_data = target_data.copy() # Фиксируем новое состояние
            #         current_surface = create_surface_from_normalized_data(current_normalized_data) # Финальный кадр
            #         pendulum.print_params() # Параметры маятника уже обновлены
            #         animation_queue.pop(0)

            if(args.anim):
                save_to_file(current_normalized_data)

            end_time = time.time()

            # Вычисляем прошедшее время и форматируем вывод
            elapsed_time = end_time - start_time
            estimated_time = round(elapsed_time * (anim['total_steps'] - anim['step'])/3600, 2)
            print(f"Rendered {(frame_counter-1):05d} at {elapsed_time:.3f}s, estimated: {estimated_time:.2f} hours")

        elif(args.anim and frame_counter > 0):
            # Если анимация завершена, выходим из цикла
            running = False

        


        if not args.anim:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    param_changed = pendulum.update_params(event.key)
                    if param_changed:
                        print("Parameter changed, starting animation...")
                        # Сохраняем текущие нормализованные данные для блендинга
                        prev_normalized_data_for_anim = current_normalized_data.copy()
                        
                        # Новые параметры уже в pendulum объекте
                        (x_min, x_max), (y_min, y_max) = pendulum.get_current_view_ranges()
                        new_raw_data = mapper.compute_map_raw(x_min, x_max, y_min, y_max)
                        new_normalized_data = mapper.normalize_data(new_raw_data)
                        
                        animation_queue.append({
                            'type': 'param_change',
                            'prev_data_norm': prev_normalized_data_for_anim,
                            'target_data_norm': new_normalized_data, # Это будут новые current_normalized_data
                            'step': 0,
                            'total_steps': ANIMATION_FRAMES
                        })
                        # current_normalized_data будет обновлен в конце этой анимации


                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r:
                        print("Resetting view, starting animation...")
                        # Целевые значения - это начальные диапазоны
                        (initial_x_min, initial_x_max), (initial_y_min, initial_y_max) = pendulum.get_initial_ranges()
                        
                        # Текущие значения для интерполяции
                        (current_x_min, current_x_max), (current_y_min, current_y_max) = pendulum.get_current_view_ranges()

                        animation_queue.append({
                            'type': 'zoom',
                            'start_view': (current_x_min, current_x_max, current_y_min, current_y_max),
                            'target_view': (initial_x_min, initial_x_max, initial_y_min, initial_y_max),
                            'step': 0,
                            'total_steps': ANIMATION_FRAMES_OUT
                        })
                        # pendulum.reset_view_ranges() будет вызван в конце этой анимации

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Левая кнопка мыши
                        if not animation_queue: # Не начинать новый зум во время анимации
                            mx, my = event.pos # Координаты мыши в пикселях окна
                            
                            (current_x_min, current_x_max), (current_y_min, current_y_max) = pendulum.get_current_view_ranges()
                            
                            # Преобразование пиксельных координат в координаты данных
                            data_x = current_x_min + (mx / WIDTH) * (current_x_max - current_x_min)
                            data_y = current_y_min + (my / HEIGHT) * (current_y_max - current_y_min)

                            
                            new_span_x = (current_x_max - current_x_min) * ZOOM_FACTOR
                            new_span_y = (current_y_max - current_y_min) * ZOOM_FACTOR

                            
                            target_x_min = data_x -  new_span_x
                            target_x_max = data_x +  new_span_x
                            target_y_min = data_y - new_span_y
                            target_y_max = data_y +  new_span_y
                        
                            if(args.pfile != ""):
                                # Записываем координаты точки в файл
                                write_target_point_to_file(args.pfile, (target_x_min, target_x_max, target_y_min, target_y_max))
                                print(f"Target point written to file: {args.pfile}")


                            print("Zooming, starting animation...")
                            animation_queue.append({
                                'type': 'zoom',
                                'start_view': (current_x_min, current_x_max, current_y_min, current_y_max),
                                'target_view': (target_x_min, target_x_max, target_y_min, target_y_max),
                                'step': 0,
                                'total_steps': ANIMATION_FRAMES
                            })




        if not args.anim:
          # --- Отрисовка ---
            screen.blit(current_surface, (0, 0))
            pygame.display.flip()
            clock.tick(60) # Ограничение FPS для отзывчивости интерфейса
    
    if not args.anim:
        pygame.quit()
    
    print("Exiting.")

if __name__ == '__main__':
    main()