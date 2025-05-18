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
parser.add_argument('--frames', type=int, default=1800,help='Number of frames for animation')

parser.add_argument('--start', type=int, default=0,help='Start frame for animation') 

parser.add_argument('--folder', type=str, default="",help='Folder to save frames')
parser.add_argument('--pfile', type=str, default="",help='File with target point coordinates')
parser.add_argument('--kernel', type=str, default="pendulum_kernel.c",
                    help='Backend to use for computation (default: opencl)')
parser.add_argument('--skipcalc', action='store_true',   
                    help='Skip calculation of the map, only show angles info')
parser.add_argument('--vertical', action='store_true', 
                    help='Use vertical orientation for the window')
parser.add_argument('--m1', type=float, default=1.0, help='Mass of the first pendulum (default: 1.0)')
parser.add_argument('--m2', type=float, default=1.0, help='Mass of the second pendulum (default: 1.0)')
parser.add_argument('--l1', type=float, default=1.0, help='Length of the first pendulum (default: 1.0)')
parser.add_argument('--l2', type=float, default=1.0, help='Length of the second pendulum (default: 1.0)')
parser.add_argument('--g', type=float, default=9.81, help='Gravitational acceleration (default: 9.81)')
parser.add_argument('--dt', type=float, default=0.2, help='Time step (default: 0.2)')
parser.add_argument('--iter', type=int, default=5000, help='Number of iterations (default: 5000)')

args = parser.parse_args()



 

if args.vertical:
    wid = round(args.height/1.777);
else:
    wid = round(args.height*1.777);


if wid %2 != 0: wid += 1

# --- Конфигурация ---
WIDTH, HEIGHT = wid, args.height

ANIMATION_FRAMES = 1 # Количество кадров для анимации зума/изменения параметров
ANIMATION_FRAMES_OUT = 1   # Количество кадров для анимации выхода из зума/изменения параметров
ZOOM_FACTOR = 0.1       # Коэффициент зума при клике мыши  
PARAM_CHANGE_STEPS = 10 # Количество кадров для изменения параметров
L1=args.l1
L2=args.l2
M1=args.m1
M2=args.m2
G=args.g
DT=args.dt
ITER=args.iter

if args.vertical:
    mody = 1.7777
    modx = 1 # Соотношение сторон для нормализации данных
else:
    modx = 1.78
    mody = modx * (HEIGHT / WIDTH)

x_min = -math.pi*modx
x_max = math.pi*modx
y_min = -math.pi*mody
y_max = math.pi*mody

frame_counter = 0


from scipy.ndimage import median_filter




def save_to_file(rgb_data):
    global frame_counter, frames_dir
    
    img = Image.fromarray(rgb_data)
    filename = os.path.join(frames_dir, f"frame_{frame_counter:05d}.png")
    img.save(filename)





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
        args.pfile = os.path.join(frames_dir, "point.json")

    if(args.start > 0):
        frame_counter = args.start
    elif args.anim:
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



def save_to_file(rgb_data):
    """Сохраняет обработанные данные в файл"""
    global frame_counter, frames_dir
    
    
    img = Image.fromarray(rgb_data)
    filename = os.path.join(frames_dir, f"frame_{frame_counter:05d}.png")
    img.save(filename)
   

def create_surface_from_normalized_data(rgb_data):
    """Создает поверхность Pygame из обработанных данных"""
    
    # Транспонируем оси для совместимости с Pygame (width, height, channels)
    return pygame.surfarray.make_surface(rgb_data.transpose(1, 0, 2))


def main():
    global animation_queue,frame_counter # Разрешаем модификацию
    global x_min, x_max, y_min, y_max








    if not args.anim:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Double Pendulum Map (OpenCL)")
        clock = pygame.time.Clock()


    if args.anim and not args.folder:
        print("Error: --folder argument is required for animation.")
    
        return
                
    total_render_time = 0
    render_time_history = []


    params = {
    'L1': args.l1,
    'L2': args.l2,
    'M1': args.m1,
    'M2': args.m2,
    'G': args.g,
    'DT': args.dt,
    'MAX_ITER': args.iter,
    'current_view': (x_min, x_max, y_min, y_max)  # Добавляем начальный вид
    }

    mapper = DPMapper(
        width=WIDTH,
        height=HEIGHT,
        kernel_file=args.kernel,
        params=params.copy()  # Гарантируем копию, а не ссылку
    )
    target_vx_min, target_vx_max, target_vy_min, target_vy_max = x_min, x_max, y_min, y_max
    # mapper.set_current_view(x_min, x_max, y_min, y_max)
    
    # Use the backend selection function to create the appropriate mapper
    
    
    setup_frame_saving() # Создаем папку для кадров при запуске

    # Для анимации изменения параметров
    current_normalized_data = None 
    


    if(frame_counter == 0):
        # Первоначальный расчет и отрисовка
        print("Performing initial calculation...")
        
        if not args.skipcalc:           
            
            rgb_data = mapper.calc_and_get_rgb_data()

            if args.anim:
                save_to_file(rgb_data)
                frame_counter += 1
            else:
                current_surface = create_surface_from_normalized_data(rgb_data)


      
  

 



    if args.anim:

        if args.pfile != "":
            # Сохраняем начальные параметры ДО загрузки файла
            initial_params = {
                'L1': L1,
                'L2': L2,
                'M1': M1,
                'M2': M2,
                'G': G,
                'DT': DT,
                'MAX_ITER': ITER
            }
            

            # Загружаем параметры ИЗ ФАЙЛА в маппер
            mapper, coords = Mapper.load_state(args.pfile, existing_mapper=mapper)
            
            # Получаем целевые параметры ИЗ ЗАГРУЖЕННОГО СОСТОЯНИЯ
            target_params = mapper.params.copy()
            
            # Добавляем комбинированную анимацию
            animation_queue.append({
                'type': 'combined',
                'start_view': (x_min, x_max, y_min, y_max),  # Начальный вид
                'target_view': coords,                        # Вид из файла
                'start_params': initial_params,               # Параметры до загрузки
                'target_params': target_params,               # Параметры из файла
                'step': args.start,
                'total_steps': args.frames
            })

        

    total_time = time.time()
    rendered_frames = 0
    running = True
    while running:

        # --- Логика обновления и анимации ---
        if animation_queue:
            anim = animation_queue[0]

            anim['step'] += 1
            t = anim['step'] / anim['total_steps']
          
            if anim['type'] == 'zoom':
 
                # Начальные и целевые границы
                if anim['target_view']:
                    target_vx_min, target_vx_max, target_vy_min, target_vy_max = anim['target_view']
                    interp_x_min, interp_x_max, interp_y_min, interp_y_max = mapper.interpolate_zoom(anim)
                    mapper.set_current_view(interp_x_min, interp_x_max, interp_y_min, interp_y_max)   



            if anim['type'] == 'combined':
                # Интерполяция параметров


                current_params = {}
                for key in anim['start_params']:
                    start = anim['start_params'][key]
                    target = anim['target_params'].get(key, start)  # Безопасный доступ
                    current_params[key] = start + (target - start) * t

                
                # Интерполяция зума
                interp_x_min, interp_x_max, interp_y_min, interp_y_max = mapper.interpolate_zoom(anim)
                
                # Обновляем маппер
                mapper.params.update(current_params)
                mapper.set_current_view(interp_x_min, interp_x_max, interp_y_min, interp_y_max)




            # Завершение анимации
            if anim['step'] >= anim['total_steps']:
                mapper.set_current_view(target_vx_min, target_vx_max, target_vy_min, target_vy_max)
                animation_queue.pop(0)

            start_time = time.time()
            rgb_data = mapper.calc_and_get_rgb_data()
            if  args.anim:
                save_to_file(rgb_data)
            
                frame_counter += 1
            else:
                current_surface = create_surface_from_normalized_data(rgb_data)
            end_time = time.time()



            elapsed_time = end_time - start_time
            render_time_history.append(elapsed_time)
            total_render_time += elapsed_time

            avg_frame_time = total_render_time / len(render_time_history)
            remaining_frames = anim['total_steps'] - anim['step']

            # Расчет общего времени в секундах и преобразование
            total_seconds = avg_frame_time * remaining_frames
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            estimated_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            progress_percent = (anim['step'] / anim['total_steps']) * 100
            frames_per_second = (1.0 / avg_frame_time)*60 if avg_frame_time > 0 else 0
            
            
            viewing_degree_angle = 0 # math.degrees(interp_x_max-interp_x_min)
            
            
            timestamp = frame_counter/30

            print(f"Done {(frame_counter-1):05d}/{anim['total_steps']}\t"
                f"{viewing_degree_angle}° @ TS {timestamp:.3f}s\t"
                f"rendered {elapsed_time:.3f}s @ "
                f"{frames_per_second:.2f}fpm\t"
                f"estimated {estimated_time_str} "
                f"({progress_percent:.1f}%)"
                )
            
            rendered_frames +=1

        elif args.anim and frame_counter > 0:
            # Если анимация завершена, выходим из цикла
            running = False

        


        if not args.anim:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:


                    step = 0.1
                    iter_step = 100
                    # Получаем текущие параметры из маппера
                    params = mapper.params
                    
                    # Длины
                    if event.key == pygame.K_UP:
                        params['L1'] += step
                    elif event.key == pygame.K_DOWN:
                        params['L1'] -= step
                    elif event.key == pygame.K_RIGHT:
                        params['L2'] += step
                    elif event.key == pygame.K_LEFT:
                        params['L2'] -= step
                        
                    # Массы
                    elif event.key == pygame.K_w:
                        params['M1'] += step
                    elif event.key == pygame.K_s:
                        params['M1'] -= step
                    elif event.key == pygame.K_d:
                        params['M2'] += step
                    elif event.key == pygame.K_a:
                        params['M2'] -= step
                        
                    # Гравитация
                    elif event.key == pygame.K_q:
                        params['G'] += step
                    elif event.key == pygame.K_e:
                        params['G'] -= step
                        
                    # Шаг времени
                    elif event.key == pygame.K_z:
                        params['DT'] = round(params['DT'] - 0.01, 2)
                    elif event.key == pygame.K_x:
                        params['DT'] = round(params['DT'] + 0.01, 2)
                        
                    # Итерации
                    elif event.key == pygame.K_c:
                        params['MAX_ITER'] += iter_step
                    elif event.key == pygame.K_v:
                        params['MAX_ITER'] = max(100, params['MAX_ITER'] - iter_step)
                    
                    # Обновляем и перерисовываем
                    if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT,
                                    pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d,
                                    pygame.K_q, pygame.K_e, pygame.K_z, pygame.K_x,
                                    pygame.K_c, pygame.K_v]:
                        rgb_data = mapper.calc_and_get_rgb_data()
                        current_surface = create_surface_from_normalized_data(rgb_data)

                        if args.pfile != "":
                            
                            mapper.save_state(args.pfile)
                            
                        

                
                        print(f"Params updated: {params}")

 

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Левая кнопка мыши
                        
                        mx, my = event.pos # Координаты мыши в пикселях окна
                        
                    
                        current_x_min, current_x_max, current_y_min, current_y_max = mapper.params['current_view']
                        
                        # Преобразование пиксельных координат в координаты данных
                        data_x = current_x_min + (mx / WIDTH) * (current_x_max - current_x_min)
                        data_y = current_y_min + (my / HEIGHT) * (current_y_max - current_y_min)

                        
                        new_span_x = (current_x_max - current_x_min) * ZOOM_FACTOR
                        new_span_y = (current_y_max - current_y_min) * ZOOM_FACTOR

                        
                        target_x_min = data_x -  new_span_x
                        target_x_max = data_x +  new_span_x
                        target_y_min = data_y - new_span_y
                        target_y_max = data_y +  new_span_y

                        mapper.set_current_view(target_x_min, target_x_max, target_y_min, target_y_max)
                        rgb_data = mapper.calc_and_get_rgb_data()
                        current_surface = create_surface_from_normalized_data(rgb_data)       
                        
                        if args.pfile != "":
                            
                            mapper.save_state(args.pfile)

                        print("Zooming, starting animation...")
                        




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