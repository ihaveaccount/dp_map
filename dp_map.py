# dp_map.py
import pygame
import numpy as np
import os
import datetime
import math
import argparse 
import pathlib
import re
import time
import json
from dp_lib import CLMapper, Mapper # Import both mappers and base Mapper

from PIL import Image

parser = argparse.ArgumentParser(description='Double Pendulum Simulation')
parser.add_argument('--anim', action='store_true', 
                    help='Run predefined animation')
parser.add_argument('--height', type=int, default=1080,
                    help='Height of the window (default: 1080)')
parser.add_argument('--frames', type=int, default=1800,help='Number of frames for animation')

parser.add_argument('--start', type=int, default=0,help='Start frame for animation') 
parser.add_argument('--end', type=int, default=None,help='End frame for animation (exclusive)')

parser.add_argument('--folder', type=str, default="",help='Folder to save frames')
parser.add_argument('--pfile', type=str, default="",help='File with target point coordinates')
parser.add_argument('--kernel', type=str, default="pendulum.c",
                    help='Kernel file to use for computation (default: pendulum.c)')

parser.add_argument('--skipcalc', action='store_true',   
                    help='Skip calculation of the map, only show angles info')
parser.add_argument('--vertical', action='store_true', 
                    help='Use vertical orientation for the window')
# Removed specific pendulum parameters from args, they will be handled dynamically or by CLMapper
# parser.add_argument('--m1', type=float, default=1.0, help='Mass of the first pendulum (default: 1.0)')
# ...
parser.add_argument('--min_x', type=float, default=None, help='Min X for view (overrides kernel default)')
parser.add_argument('--max_x', type=float, default=None, help='Max X for view (overrides kernel default)')
parser.add_argument('--min_y', type=float, default=None, help='Min Y for view (overrides kernel default)')
parser.add_argument('--max_y', type=float, default=None, help='Max Y for view (overrides kernel default)')
parser.add_argument('--median', type=int, default=3, help='Median filter size (0 — do not apply)')
parser.add_argument('--invert', action='store_true', help='Invert image (min=255, max=0)')

# New arguments for generic parameter passing
parser.add_argument('--param', nargs=2, action='append', metavar=('NAME', 'VALUE'),
                    help='Set a kernel parameter. Use multiple --param NAME VALUE for multiple parameters.')

parser.add_argument('--param_to', nargs=2, action='append', metavar=('NAME', 'VALUE'),
                    help='Set a target kernel parameter for param_interpolation. Use multiple --param_to NAME VALUE for multiple parameters.')

parser.add_argument('--paranim', action='store_true', 
                    help='Animate parameter interpolation from current to --param values')


args = parser.parse_args()

if args.vertical:
    wid = round(args.height/1.777)
else:
    wid = round(args.height*1.777)

if wid %2 != 0: wid += 1

# --- Configuration ---
WIDTH, HEIGHT = wid, args.height

ZOOM_FACTOR = 0.1       # Zoom factor on mouse click
SMOOTHING_FILE = "smooth.json"  # File for brightness smoothing parameters

frame_counter = 0

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

# --- Global variables for animation and saving ---
frames_dir = ""
animation_queue = [] # [(type, data, current_step, total_steps, prev_data_norm)]

def setup_frame_saving():
    global frame_counter, frames_dir
    
    base_dir = "frames"
    
    if args.folder == "":
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        frames_dir = os.path.join(base_dir, f"run_{HEIGHT}_{run_timestamp}")
    else:
        frames_dir = os.path.join(base_dir, args.folder)

    if not args.pfile:
        args.pfile = os.path.join(frames_dir, "point.json")

    if args.start > 0:
        frame_counter = args.start
    elif args.anim:
        max_frame_number = find_max_frame_number(frames_dir)
        if max_frame_number > 0:
            frame_counter = max_frame_number + 1
            print(f"Starting from frame {frame_counter}")
        else:
            frame_counter = 0
    
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Saving frames to: {frames_dir}")

def save_to_file(rgb_data):
    """Saves processed data to file"""
    global frame_counter, frames_dir
    
    img = Image.fromarray(rgb_data)
    filename = os.path.join(frames_dir, f"frame_{frame_counter:05d}.png")
    img.save(filename)
   
def create_surface_from_normalized_data(rgb_data):
    """Creates a Pygame surface from processed data"""
    return pygame.surfarray.make_surface(rgb_data.transpose(1, 0, 2))

def main():
    global animation_queue, frame_counter 
    
    view_history = []


    if args.paranim:
        args.anim = True


    if not args.anim:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pendulum Map")
        clock = pygame.time.Clock()

    if args.anim and not args.folder:
        print("Error: --folder argument is required for animation.")
        return
    



                
    total_render_time = 0
    render_time_history = []

    # Initialize params dictionary
    params = {}
 

        

    if args.param:
        for param_name, param_value in args.param:
            try:
                # Try converting to float first, then int
                if '.' in param_value:
                    params[param_name] = float(param_value)
                else:
                    params[param_name] = int(param_value)
            except ValueError:
                print(f"Warning: Could not parse parameter '{param_name}' with value '{param_value}'. Keeping as string.")
                params[param_name] = param_value


    mapper = CLMapper(
                width=WIDTH,
                height=HEIGHT,
                kernel_file=args.kernel,
                params = params.copy()
            )

    mapper.set_median_filter_size(args.median)
    mapper.set_invert(args.invert)


    # Override with command line arguments if provided
    if args.min_x is not None:
        x_min = args.min_x
    if args.max_x is not None:
        x_max = args.max_x
    if args.min_y is not None:
        y_min = args.min_y
    if args.max_y is not None:
        y_max = args.max_y

    # mapper.set_current_view(x_min, x_max, y_min, y_max)
    x_min, x_max, y_min, y_max = mapper.get_current_view()
    
    target_vx_min, target_vx_max, target_vy_min, target_vy_max = x_min, x_max, y_min, y_max # For animation logic

    setup_frame_saving() 

    # Setup brightness smoothing system
    smooth_file_path = os.path.join(frames_dir, SMOOTHING_FILE)
    is_continuing_from_frame = (args.start > 0) or (args.anim and frame_counter > 0)
    mapper.setup_brightness_smoothing(smooth_file_path, frame_counter, is_continuing_from_frame) 

    
    # Parameters for keyboard control
    # Get a sorted list of parameter names for consistent indexing
    keyboard_controllable_params = sorted([k for k in mapper.params.keys() if k not in ['current_view']])

    selected_param_index = 0
    param_step_size = 0.1
    param_iter_step_size = 100 # For integer parameters like MAX_ITERATIONS

    param_changed = False
    last_param_change_time = 0
    RECALC_DELAY = 1.0  # seconds

    if frame_counter == 0:
        print("Performing initial calculation...")
        if not args.skipcalc:
            if not args.paranim:
                new_target = (x_min, x_max, y_min, y_max)
                
                view_width_absolute = x_max - x_min
                
                mapper.set_current_view(*new_target)
                
                # Update frame counter for brightness smoothing
                mapper.update_frame_counter(frame_counter)
                
                rgb_data = mapper.calc_and_get_rgb_data()
                
                # Save brightness smoothing parameters after calculation
                mapper.save_brightness_smoothing()
                
                if args.anim:
                    save_to_file(rgb_data)
                    frame_counter += 1
                else:
                    mapper.add_keyframe(args.pfile, view_width_absolute)
                    mapper.init_point_file(args.pfile)
                    current_surface = create_surface_from_normalized_data(rgb_data)
        else:
             # Create a dummy surface if calculation is skipped and not in animation mode
            if not args.anim:
                current_surface = pygame.Surface((WIDTH, HEIGHT))
                current_surface.fill((100, 100, 100)) # Grey background
            frame_counter += 1 # Increment even if skipped for consistency in animation start

    if args.paranim:
        if not args.param or not args.param_to:
            print("Error: --paranim requires both --param (start) and --param_to (end) parameters.")
            return

        # 1. Получаем дефолтные значения из шейдера
        default_params = mapper.get_initial_params()

        # 2. Формируем стартовые параметры из --param (поверх дефолтов)
        start_params = default_params.copy()
        for param_name, param_value in args.param:
            try:
                if '.' in param_value:
                    start_params[param_name] = float(param_value)
                else:
                    start_params[param_name] = int(param_value)
            except ValueError:
                start_params[param_name] = param_value

        # 3. Формируем целевые параметры из --param_to (поверх дефолтов)
        target_params = default_params.copy()
        for param_name, param_value in args.param_to:
            try:
                if '.' in param_value:
                    target_params[param_name] = float(param_value)
                else:
                    target_params[param_name] = int(param_value)
            except ValueError:
                target_params[param_name] = param_value

        print("Start params: ", start_params)
        print("Target params: ", target_params)

        animation_queue.append({
            'type': 'param_interpolation',
            'start_params': start_params.copy(),
            'target_params': target_params.copy(),
            'step': frame_counter,
            'total_steps': args.frames
        })


    elif args.anim:
        if args.pfile != "":
            initial_params = mapper.params.copy() # Capture current params before loading
            
            # Load state might update mapper.params and keyframes
            mapper, coords = Mapper.load_state(args.pfile, existing_mapper=mapper)
            
            target_params = mapper.params.copy() # Params after loading
            
            animation_queue.append({
                'type': 'keyframe',
                'start_view': (x_min, x_max, y_min, y_max),  # This should be the view at frame 0
                'target_view': coords,                        # Target view from file
                'start_params': initial_params,               # Params before loading
                'target_params': target_params,               # Params after loading
                'step': frame_counter,
                'total_steps': args.frames
            })
    

        
    total_time = time.time()
    rendered_frames = 0
    running = True
    view_history = [mapper.get_current_view()]  # Initialize stack with the first view
    current_width = 0.0
    while running:
        if animation_queue:
            anim = animation_queue[0]
            
            anim['step'] += 1
            t = anim['step'] / anim['total_steps']
          
            if anim['type'] == 'zoom':
                if anim['target_view']:
                    target_vx_min, target_vx_max, target_vy_min, target_vy_max = anim['target_view']
                    interp_x_min, interp_x_max, interp_y_min, interp_y_max = mapper.interpolate_zoom(anim)
                    mapper.set_current_view(interp_x_min, interp_x_max, interp_y_min, interp_y_max)   

            if anim['type'] == 'keyframe':
                target_vx_min, target_vx_max, target_vy_min, target_vy_max = anim['target_view']
                interp_x_min, interp_x_max, interp_y_min, interp_y_max = mapper.interpolate_zoom(anim)
                
                current_width = interp_x_max - interp_x_min
                
                sorted_keyframes = sorted(mapper.keyframes, key=lambda k: k['target_view_width'])
                
                left_idx, right_idx = 0, 0
                # Find the two keyframes that bracket the current view width
                found_bracket = False
                if len(sorted_keyframes) > 1:
                    for i in range(len(sorted_keyframes)-1):
                        if sorted_keyframes[i]['target_view_width'] <= current_width <= sorted_keyframes[i+1]['target_view_width']:
                            left_idx = i
                            right_idx = i + 1
                            found_bracket = True
                            break
                    if not found_bracket: # If current_width is outside the range of keyframes
                        if current_width < sorted_keyframes[0]['target_view_width']:
                            left_idx = right_idx = 0
                        else: # current_width > sorted_keyframes[-1]['target_view_width']
                            left_idx = right_idx = len(sorted_keyframes) - 1
                elif len(sorted_keyframes) == 1:
                    left_idx = right_idx = 0
                else: # No keyframes, use initial params
                    # Handle case with no keyframes gracefully, perhaps use current parameters
                    mapper.set_current_view(interp_x_min, interp_x_max, interp_y_min, interp_y_max)
                    # No parameter interpolation if no keyframes
                    if anim['step'] >= anim['total_steps']:
                        animation_queue.pop(0)
                    continue

                left_kf = sorted_keyframes[left_idx]
                right_kf = sorted_keyframes[right_idx]
                
                # Correct interpolation factor
                width_range = right_kf['target_view_width'] - left_kf['target_view_width']
                if width_range == 0:
                    local_t = 0.0
                else:
                    local_t = (current_width - left_kf['target_view_width']) / width_range
                
                # Apply smooth interpolation (same as param_interpolation)
                local_t = 0.5 * (1 - math.cos(math.pi * local_t))
                
                # Interpolate parameters
                for param_name in mapper.param_order: # Iterate through recognized params
                    if param_name in left_kf['params'] and param_name in right_kf['params']:
                        start_val = left_kf['params'][param_name]
                        end_val = right_kf['params'][param_name]
                        
                        if isinstance(start_val, int): # Handle int parameters
                            mapper.params[param_name] = int(start_val + (end_val - start_val) * local_t)
                        else: # Assume float otherwise
                            mapper.params[param_name] = start_val + (end_val - start_val) * local_t
                        
                        # print (param_name, "->",mapper.params[param_name]) # Debug print
                            
                mapper.set_current_view(interp_x_min, interp_x_max, interp_y_min, interp_y_max)


            if anim['type'] == 'param_interpolation':
                t = anim['step'] / anim['total_steps']
                t = 0.5 * (1 - math.cos(math.pi * t))

                params = {}

                # Интерполировать все параметры, которые есть в start_params и target_params
                for param_name in set(anim['start_params'].keys()).union(anim['target_params'].keys()):
                    start_val = anim['start_params'].get(param_name, mapper.params.get(param_name))
                    end_val = anim['target_params'].get(param_name, start_val)
                    if start_val == end_val or not isinstance(start_val, (int, float)) or not isinstance(end_val, (int, float)):
                        params[param_name] = start_val
                        continue
                    interpolated = start_val + (end_val - start_val) * t


                    print(f"{param_name} {start_val}->{end_val} = {interpolated}")

                    if isinstance(start_val, int):
                        params[param_name] = int(round(interpolated))
                    else:
                        params[param_name] = interpolated

                
                mapper.set_params(params)


            start_time = time.time()
            if not args.skipcalc:
                # Update frame counter for brightness smoothing
                mapper.update_frame_counter(frame_counter)
                
                rgb_data = mapper.calc_and_get_rgb_data()

                # Save brightness smoothing parameters after calculation
                mapper.save_brightness_smoothing()

                if args.anim:
                    save_to_file(rgb_data)
                    frame_counter += 1
                else:
                    current_surface = create_surface_from_normalized_data(rgb_data)
            else:
                frame_counter += 1

            # End animation - check after frame calculation and saving
            if anim['step'] >= anim['total_steps'] or (args.end is not None and frame_counter > args.end):
                mapper.set_current_view(target_vx_min, target_vx_max, target_vy_min, target_vy_max)
                animation_queue.pop(0)
            
            end_time = time.time()

            elapsed_time = end_time - start_time
            render_time_history.append(elapsed_time)
            total_render_time += elapsed_time

            avg_frame_time = total_render_time / len(render_time_history)
            
            # Calculate remaining frames considering --end argument
            effective_total_steps = anim['total_steps']
            if args.end is not None and args.end < anim['total_steps']:
                effective_total_steps = args.end
            
            remaining_frames = effective_total_steps - anim['step']

            total_seconds = avg_frame_time * remaining_frames
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            estimated_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            progress_percent = (anim['step'] / effective_total_steps) * 100
            frames_per_second = (1.0 / avg_frame_time)*60 if avg_frame_time > 0 else 0 # frames per minute if multiplied by 60
            
            viewing_angle = mapper.get_current_view()[1] - mapper.get_current_view()[0]
            timestamp = frame_counter / 30  # Assuming 30 FPS for timestamp display
            
            # Show effective total for display
            display_total = effective_total_steps if args.end is not None and args.end < anim['total_steps'] else anim['total_steps']
            
            print(f"{(frame_counter-1):05d}/{display_total}\t"
                  f"{viewing_angle:.16e} - TS {timestamp:.3f}s\t"
                  f"took {elapsed_time:.3f}s @ "
                  f"{frames_per_second:.2f}fpm\t\t"
                  f"est. {estimated_time_str} "
                  f"({progress_percent:.1f}%)"
            )
            
            rendered_frames +=1

        elif args.anim and (frame_counter > 0 or (args.end is not None and frame_counter > args.end)):
            running = False
        
        if not args.anim:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    # Generic parameter control (up to 9 parameters)
                    if pygame.K_1 <= event.key <= pygame.K_9:
                        idx = event.key - pygame.K_1
                        if idx < len(keyboard_controllable_params):
                            selected_param_index = idx
                            param_name = keyboard_controllable_params[selected_param_index]
                            print(f"Selected parameter for adjustment: {keyboard_controllable_params[selected_param_index]} (Value: {mapper.params[param_name]})")
                        else:
                            print("No parameter mapped to this key.")
                    elif event.key == pygame.K_UP:
                        if keyboard_controllable_params:
                            param_name = keyboard_controllable_params[selected_param_index]
                            if isinstance(mapper.params[param_name], int):
                                mapper.params[param_name] += param_iter_step_size
                            elif isinstance(mapper.params[param_name], float):
                                mapper.params[param_name] += param_step_size
                            print(f"Increased {param_name} to {mapper.params[param_name]}")
                            param_changed = True
                            last_param_change_time = time.time()
                    elif event.key == pygame.K_DOWN:
                        if keyboard_controllable_params:
                            param_name = keyboard_controllable_params[selected_param_index]
                            if isinstance(mapper.params[param_name], int):
                                mapper.params[param_name] = max(1, mapper.params[param_name] - param_iter_step_size) # Prevent going below 1 for iterations
                            elif isinstance(mapper.params[param_name], float):
                                mapper.params[param_name] -= param_step_size
                            print(f"Decreased {param_name} to {mapper.params[param_name]}")
                            param_changed = True
                            last_param_change_time = time.time()

                    elif event.key == pygame.K_r: # Reset view
                        mapper.set_current_view(x_min, x_max, y_min, y_max)
                        rgb_data = mapper.calc_and_get_rgb_data()
                        current_surface = create_surface_from_normalized_data(rgb_data)
                        print("View reset.")

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left mouse button (zoom in)
                        mx, my = event.pos 
                        current_x_min, current_x_max, current_y_min, current_y_max = mapper.params['current_view']
                        
                        data_x = current_x_min + (mx / WIDTH) * (current_x_max - current_x_min)
                        data_y = current_y_min + (my / HEIGHT) * (current_y_max - current_y_min)
                        
                        new_span_x = (current_x_max - current_x_min) * ZOOM_FACTOR
                        new_span_y = (current_y_max - current_y_min) * ZOOM_FACTOR
                        
                        target_x_min = data_x -  new_span_x / 2 # Center zoom
                        target_x_max = data_x +  new_span_x / 2
                        target_y_min = data_y - new_span_y / 2
                        target_y_max = data_y + new_span_y / 2

                        view_width = target_x_max - target_x_min
                        view_width_deg = (view_width)

                        print("Zooming to ", f"{view_width_deg:.6e}", "view", (target_x_min, target_x_max, target_y_min, target_y_max) )
                        mapper.set_current_view(target_x_min, target_x_max, target_y_min, target_y_max)
                        rgb_data = mapper.calc_and_get_rgb_data()
                        current_surface = create_surface_from_normalized_data(rgb_data)       
                        
                        mapper.add_keyframe(args.pfile, view_width)
                        
                        current_view = mapper.get_current_view()
                        if not view_history or current_view != view_history[-1]:
                            view_history.append(current_view)
                        
                    if event.button == 3:  # Right mouse button (zoom out / return)
                        if len(view_history) > 1:
                            view_history.pop()
                            prev_view = view_history[-1]
                            mapper.set_current_view(*prev_view)
                            rgb_data = mapper.calc_and_get_rgb_data()
                            current_surface = create_surface_from_normalized_data(rgb_data)
                            print("Returned to previous view:", prev_view)
                        else:
                            print("No previous view in history.")
                            
            if param_changed and (time.time() - last_param_change_time) >= RECALC_DELAY:
                
                print("Params recalculation...")
                timer = time.time()
                rgb_data = mapper.calc_and_get_rgb_data()
                current_surface = create_surface_from_normalized_data(rgb_data)
                param_changed = False
                timer = time.time()-timer;
                print("Done in", timer)
                

            screen.blit(current_surface, (0, 0))
            pygame.display.flip()
            clock.tick(60) 
    
    if not args.anim:
        pygame.quit()
    
    print("Exiting.")

if __name__ == '__main__':
    main()