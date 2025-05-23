# dp_lib.py
import numpy as np
import pyopencl as cl
import math
import re
import time
from scipy.ndimage import median_filter
import json


class Mapper:
    def __init__(self, width, height, params=None):
        self.width = width
        self.height = height
        self.params = params or {}
        self.raw_data = {} # Changed to dictionary to store multiple raw channels
        self.normalized_data = {} # Changed to dictionary for multiple normalized channels
        self.keyframes = []  # New field for storing keyframes
        self.output_channels = ['brightness'] # Default output channel, can be 'R', 'G', 'B', 'H', 'S', 'V'
        self.initial_params = {}
        
    @staticmethod
    

    def interpolate_params(self, anim):
        t = anim['step'] / anim['total_steps']


    def interpolate_zoom(self, anim):
        t = anim['step'] / anim['total_steps']
        # Initial and target boundaries
        start_vx_min, start_vx_max, start_vy_min, start_vy_max = anim['start_view']
        target_vx_min, target_vx_max, target_vy_min, target_vy_max = anim['target_view']
        phase_cutoff = 0.05  # 5% of time for centering
        

        # Initial parameters
        start_scale_x = start_vx_max - start_vx_min
        target_scale_x = target_vx_max - target_vx_min
        start_scale_y = start_vy_max - start_vy_min
        target_scale_y = target_vy_max - target_vy_min

        start_center = ((start_vx_min + start_vx_max)/2, 
                    (start_vy_min + start_vy_max)/2)
        target_center = ((target_vx_min + target_vx_max)/2, 
                        (target_vy_min + target_vy_max)/2)

        # Calculate target scale for phase 1 (reduce by 2)
        phase1_target_scale_x = start_scale_x / 2
        phase1_target_scale_y = start_scale_y / 2

        if t <= phase_cutoff:
            # Phase 1: Panning + scale reduction
            t_phase = t / phase_cutoff
            t_eased = 0.5 * (1 - math.cos(t_phase * math.pi))  # Smooth acceleration
            
            # Center interpolation
            current_center_x = start_center[0] + (target_center[0] - start_center[0]) * t_eased
            current_center_y = start_center[1] + (target_center[1] - start_center[1]) * t_eased
            
            # Scale interpolation to reduced by 2
            current_scale_x = start_scale_x - (start_scale_x - phase1_target_scale_x) * t_eased
            current_scale_y = start_scale_y - (start_scale_y - phase1_target_scale_y) * t_eased
            
        else:
            # Phase 2: Exponential increase
            t_phase = (t - phase_cutoff) / (1 - phase_cutoff)
            
            # Initial scale - result of phase 1
            start_scale_phase2_x = phase1_target_scale_x
            start_scale_phase2_y = phase1_target_scale_y
            
            # Exponential interpolation
            current_scale_x = start_scale_phase2_x * (target_scale_x / start_scale_phase2_x) ** t_phase
            current_scale_y = start_scale_phase2_y * (target_scale_y / start_scale_phase2_y) ** t_phase
            
            # Center is fixed at the target
            current_center_x = target_center[0]
            current_center_y = target_center[1]

        # Calculate area boundaries
        interp_x_min = current_center_x - current_scale_x / 2
        interp_x_max = current_center_x + current_scale_x / 2
        interp_y_min = current_center_y - current_scale_y / 2
        interp_y_max = current_center_y + current_scale_y / 2

        # Guarantee that the target area remains in frame
        interp_x_min = max(interp_x_min, target_vx_min)
        interp_x_max = min(interp_x_max, target_vx_max)
        interp_y_min = max(interp_y_min, target_vy_min)
        interp_y_max = min(interp_y_max, target_vy_max)

    
        # Calculate boundaries
        interp_x_min = current_center_x - current_scale_x / 2
        interp_x_max = current_center_x + current_scale_x / 2
        interp_y_min = current_center_y - current_scale_y / 2
        interp_y_max = current_center_y + current_scale_y / 2


        return interp_x_min, interp_x_max, interp_y_min, interp_y_max

    def compute_map(self, x_min, x_max, y_min, y_max):
        raise NotImplementedError

    def normalize_data(self): # Modified to take no arguments, uses self.raw_data
        """ Normalizes raw data (logarithm + min-max scaling to 0-255) for each channel. """
        for channel_name, raw_channel_data in self.raw_data.items():
            # Apply natural logarithm (log1p for zeros: log(1+x))
            log_data = np.log1p(raw_channel_data.astype(np.float64))

            min_log_val = np.min(log_data)
            max_log_val = np.max(log_data)
            
            if max_log_val == min_log_val: # If all values are the same
                normalized_channel_data = np.zeros_like(log_data, dtype=np.uint8)
            else:
                normalized_channel_data = 255 * (log_data - min_log_val) / (max_log_val - min_log_val)
            
            self.normalized_data[channel_name] = np.clip(normalized_channel_data, 0, 255).astype(np.uint8)


        # ... (inside the loop for each channel)
        self.normalized_data[channel_name] = np.clip(normalized_channel_data, 0, 255).astype(np.uint8)
        # Debugging: Print some statistics of the normalized data
        # print(f"Normalized data for {channel_name}: min={np.min(self.normalized_data[channel_name])}, max={np.max(self.normalized_data[channel_name])}, mean={np.mean(self.normalized_data[channel_name])}")
        # print(f"Normalized data (first 10 elements): {self.normalized_data[channel_name][:10]}")
        return self.normalized_data         

        return self.normalized_data

    def set_output_channels(self, channels):
        """Sets the names of the output channels the kernel will produce (e.g., ['H', 'S', 'B'] or ['R', 'G', 'B'])."""
        self.output_channels = channels

    def get_output_channels(self):
        return self.output_channels

    def add_keyframe(self, filename, view_width_deg):
        """Adds a keyframe only if parameters change and updates target_view"""
        # Copy parameters, excluding current_view
        params_to_save = {k: v for k, v in self.params.items() if k not in ['current_view']}
        
        # Check if parameters differ from previous frame
        if self.keyframes:
            last_params = self.keyframes[-1]["params"]
            if params_to_save == last_params:
                # Even if parameters didn't change, update target_view
                self._save_keyframes(filename)
                return  # Parameters didn't change - skip adding, but update view
        
        new_keyframe = {
            "target_view_width": view_width_deg,
            "params": params_to_save
        }
        
        self.keyframes.append(new_keyframe)
        self._save_keyframes(filename)



    def _save_keyframes(self, filename):
        """Saves keyframes and target view separately"""
        state = {
            "keyframes": self.keyframes,
            "target_view": list(self.get_current_view())
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)


    @classmethod
    def load_state(cls, filename, existing_mapper=None):
        
        with open(filename) as f:
            state = json.load(f)
        
        if existing_mapper:
            existing_mapper.keyframes = state.get("keyframes", [])
            existing_mapper.params.update(state.get("params", {}))
            return existing_mapper, state.get("target_view")
        
        new_mapper = cls(1, 1, state.get("params", {}))
        new_mapper.keyframes = state.get("keyframes", [])
        return new_mapper, state.get("target_view")
 

    def get_current_view(self):
        return self.params["current_view"]


    def set_current_view(self, x_min, x_max, y_min, y_max):
        self.params['current_view'] = (x_min, x_max, y_min, y_max)


    def calc_and_get_rgb_data(self):
        
        self.compute_map()

        timer = time.time()


        self.normalize_data()  # Call without arguments

        median_size = self.get_median_filter_size()

 

        if set(self.output_channels) == {'R', 'G', 'B'}:
            # Apply median filter to each channel if needed
            r_data = self.normalized_data['R'].reshape((self.height, self.width))
            g_data = self.normalized_data['G'].reshape((self.height, self.width))
            b_data = self.normalized_data['B'].reshape((self.height, self.width))
            if median_size and median_size > 0:
                r_data = median_filter(r_data, size=median_size)
                g_data = median_filter(g_data, size=median_size)
                b_data = median_filter(b_data, size=median_size)

            rgb_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            rgb_data[:, :, 0] = r_data
            rgb_data[:, :, 1] = g_data
            rgb_data[:, :, 2] = b_data

        elif 'brightness' in self.output_channels:
            img_data = self.normalized_data['brightness'].copy()
            
            
            
            if img_data.max() <= 1.0:
                img_data = (img_data * 255).astype(np.uint8)
            else:
                img_data = img_data.astype(np.uint8)
            if self.get_invert():
                img_data = 255 - img_data
            img_data = img_data.reshape((self.height, self.width))  # <-- добавьте эту строку

            if median_size and median_size > 0:
                img_data = median_filter(img_data, size=median_size)


            height, width = img_data.shape
            rgb_data = np.zeros((height, width, 3), dtype=np.uint8)
            rgb_data[:, :, :] = img_data[..., np.newaxis]
        else:
            raise ValueError("Unsupported output channel configuration.")

        timer = time.time()-timer
        
        if timer > 1.0:
            print(f"Normalize timer {timer}")

        return rgb_data

    def init_point_file(self, filename):
        """Creates or overwrites a file with initial parameters and current view"""
        # Save all kernel parameters except 'current_view'
        state = {
            "start_params": {k: v for k, v in self.params.items() if k != 'current_view'},
            "start_view": list(self.get_current_view())
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)

    def set_median_filter_size(self, size):
        """Sets the median filter size (0 — do not apply)"""
        self._median_filter_size = size

    def get_median_filter_size(self):
        return getattr(self, '_median_filter_size', 0)

    def set_invert(self, invert: bool):
        self._invert = invert

    def get_invert(self):
        return getattr(self, '_invert', False)

    def get_parameters(self):
        """
        Returns a copy of all current parameters (including current_view and any user overrides).
        """
        return self.params.copy()


class CLMapper(Mapper):
    def __init__(self, width, height, kernel_file, params=None):
        super().__init__(width, height, params)
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        
        with open(kernel_file, "r") as f:
            kernel_code = f.read()
        
        self.param_order = []
        self.kernel_function_name = 'simulate'
        self.default_view_from_kernel = {}
        self._parse_kernel_parameters(kernel_code)
        self._parse_kernel_view_defaults(kernel_code)
        self._parse_kernel_output_channels(kernel_code)

        # После парсинга параметров из ядра, только теперь применяем params поверх дефолтов:
        if params:
            for k, v in params.items():
                self.params[k] = v
        # Гарантируем, что все параметры из ядра есть в self.params
        for name in self.param_order:
            if name not in self.params:
                self.params[name] = self.initial_params[name]
        
        self.prg = cl.Program(self.ctx, kernel_code).build()
        
        self.num_points = width * height
        
        # Changed: Multiple output buffers based on parsed output_channels
        self.output_bufs = {}
        for channel_name in self.output_channels:
            self.output_bufs[channel_name] = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, np.empty(self.num_points, dtype=np.float32).nbytes) # Use float32 for output
            self.raw_data[channel_name] = np.empty(self.num_points, dtype=np.float32) # Store raw data for each channel

        self.mf = cl.mem_flags
        self.input_xs_np = np.empty(self.num_points, dtype=np.double) # Renamed
        self.input_ys_np = np.empty(self.num_points, dtype=np.double) # Renamed

    def _parse_kernel_parameters(self, kernel_code):
        param_pattern = re.compile(r'//\s*PARAM:\s*(\w+)\s+(\w+)\s+([\d\.\-]+)')
        self.param_order = []
        self.initial_params.clear()  # Очищаем перед парсингом
        for line in kernel_code.split('\n'):
            line = line.strip()
            if line.startswith('// PARAM:'):
                match = param_pattern.match(line)
                if match:
                    name, type_str, default_str = match.groups()
                    if name in ['width', 'height']:
                        continue
                    # Определение типа значения
                    if type_str == 'double':
                        default = float(default_str)
                    elif type_str == 'int':
                        default = int(default_str)
                    else:
                        continue  # Пропускаем неподдерживаемые типы
                    # Сохраняем в initial_params
                    self.initial_params[name] = default
                    # Добавляем в self.params, если отсутствует
                    if name not in self.params:
                        self.params[name] = default
                    self.param_order.append(name)

    def get_initial_params(self):
        """Возвращает параметры, прочитанные из комментариев шейдера."""
        return self.initial_params.copy()  # Возвращаем копию для безопасности


    def _parse_kernel_view_defaults(self, kernel_code):
        view_pattern = re.compile(r'//\s*VIEW_DEFAULT:\s*(x_min|x_max|y_min|y_max)\s+([\d\.\-]+)')
        for line in kernel_code.split('\n'):
            line = line.strip()
            if line.startswith('// VIEW_DEFAULT:'):
                match = view_pattern.match(line)
                if match:
                    key, value_str = match.groups()
                    self.default_view_from_kernel[key] = float(value_str)

        # --- Adjust to aspect ratio ---
        # Adjust only if all 4 parameters are read and sizes are known
        if all(k in self.default_view_from_kernel for k in ('x_min', 'x_max', 'y_min', 'y_max')):
            x_min = self.default_view_from_kernel['x_min']
            x_max = self.default_view_from_kernel['x_max']
            y_min = self.default_view_from_kernel['y_min']
            y_max = self.default_view_from_kernel['y_max']
            span_x = x_max - x_min
            span_y = y_max - y_min

            # Get sizes from self.width/self.height, if present
            width = getattr(self, 'width', None)
            height = getattr(self, 'height', None)
            if width is not None and height is not None:
                aspect_kernel = span_x / span_y
                aspect_target = width / height

                if (aspect_kernel < aspect_target):
                    # Need to increase span_x
                    new_span_x = span_y * aspect_target
                    center_x = (x_min + x_max) / 2
                    x_min = center_x - new_span_x / 2
                    x_max = center_x + new_span_x / 2
                elif (aspect_kernel > aspect_target):
                    # Need to increase span_y
                    new_span_y = span_x / aspect_target
                    center_y = (y_min + y_max) / 2
                    y_min = center_y - new_span_y / 2
                    y_max = center_y + new_span_y / 2

                self.default_view_from_kernel['x_min'] = x_min
                self.default_view_from_kernel['x_max'] = x_max
                self.default_view_from_kernel['y_min'] = y_min
                self.default_view_from_kernel['y_max'] = y_max
                
                self.params['current_view'] = (x_min, x_max, y_min, y_max)

    def _parse_kernel_output_channels(self, kernel_code):
        output_pattern = re.compile(r'//\s*OUTPUT_CHANNELS:\s*(\w+(?:,\s*\w+)*)')
        for line in kernel_code.split('\n'):
            line = line.strip()
            if line.startswith('// OUTPUT_CHANNELS:'):
                match = output_pattern.match(line)
                if match:
                    channels_str = match.groups()[0]
                    self.set_output_channels([c.strip() for c in channels_str.split(',')])
                    return
        # If no OUTPUT_CHANNELS comment, default to 'brightness'
        self.set_output_channels(['brightness'])

    def get_kernel_params(self):
        params = []
        for name in self.param_order:
            value = self.params.get(name)
            # print(f"param {name}: {value} ({type(value)})")  # <-- add this for debug
            if isinstance(value, float):
                params.append(np.double(value))
            elif isinstance(value, int):
                params.append(np.int32(value))
            else:
                params.append(value)
        params.append(np.int32(self.width))
        params.append(np.int32(self.height))
        # print("Final params:", params)
        return tuple(params)

    def set_params(self, params):
        """
        Overwrites current parameters with the provided dictionary, keeping any missing parameters unchanged.
        """
        if params:
            for k, v in params.items():
                self.params[k] = v
        # Гарантируем, что все параметры из param_order присутствуют (даже если не были явно заданы)
        for name in self.param_order:
            if name not in self.params:
                self.params[name] = self.initial_params[name]

    def compute_map(self):
        # Use current parameters from get_parameters()
        params = self.get_parameters()
        x_min, x_max, y_min, y_max = params['current_view']
        xs_vals = np.linspace(x_min, x_max, self.width, dtype=np.double)
        ys_vals = np.linspace(y_min, y_max, self.height, dtype=np.double)

        for r_idx in range(self.height):
            for c_idx in range(self.width):
                idx = r_idx * self.width + c_idx
                self.input_xs_np[idx] = xs_vals[c_idx]
                self.input_ys_np[idx] = ys_vals[r_idx]

        input_xs_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.input_xs_np)
        input_ys_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.input_ys_np)

        kernel_args = [
            input_xs_buf,
            input_ys_buf,
        ]
        for channel_name in self.output_channels:
            kernel_args.append(self.output_bufs[channel_name])
        kernel_args.extend(self.get_kernel_params())

        global_size = (self.height, self.width)
        kernel_func = getattr(self.prg, self.kernel_function_name)
        kernel_func(self.queue, global_size, None, *kernel_args).wait()

        for channel_name in self.output_channels:
            cl.enqueue_copy(self.queue, self.raw_data[channel_name], self.output_bufs[channel_name]).wait()

        return self.raw_data

