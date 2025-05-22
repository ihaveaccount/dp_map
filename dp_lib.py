# dp_lib.py
import numpy as np
import pyopencl as cl
import math
import re
from scipy.ndimage import median_filter
import json


class Mapper:
    def __init__(self, width, height, params=None):
        self.width = width
        self.height = height
        self.params = params or {}
        self.raw_data = None
        self.normalized_data = None
        self.keyframes = []  # New field for storing keyframes
        
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

    def normalize_data(self, raw_data):
        self.raw_data = self.compute_map()
        """ Normalizes raw data (logarithm + min-max scaling to 0-255). """
        # Apply natural logarithm (log1p for zeros: log(1+x))
        log_data = np.log1p(self.raw_data.astype(np.float64))

        min_log_val = np.min(log_data)
        max_log_val = np.max(log_data)
        
        if max_log_val == min_log_val: # If all values are the same
            normalized_data = np.zeros_like(log_data, dtype=np.uint8)
        else:
            normalized_data = 255 * (log_data - min_log_val) / (max_log_val - min_log_val)
        
        normalized_data = np.clip(normalized_data, 0, 255).astype(np.uint8)
        
        self.normalized_data = normalized_data.reshape((self.height, self.width))
        return self.normalized_data

    


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
        self.normalize_data(self.raw_data)
        """Generates RGB data from normalized 2D data with preprocessing"""
        img_data = self.normalized_data.copy()
        
        median_size = self.get_median_filter_size()
        
        if median_size and median_size > 0:
            img_data = median_filter(img_data, size=median_size)        
        # Normalize and convert to uint8
        if img_data.max() <= 1.0:
            img_data = (img_data * 255).astype(np.uint8)
        else:
            img_data = img_data.astype(np.uint8)
        

        # Inversion, if required
        if self.get_invert():
            img_data = 255 - img_data

        # Create 3-channel image
        height, width = img_data.shape
        rgb_data = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_data[:, :, :] = img_data[..., np.newaxis]  # Copy data to all 3 channels
        
        return rgb_data

    def init_point_file(self, filename):
        """Creates or overwrites a file with initial parameters and current view"""
        state = {
            "start_params": {
                "L1": self.params.get('L1', 1.0),
                "L2": self.params.get('L2', 1.0),
                "M1": self.params.get('M1', 1.0),
                "M2": self.params.get('M2', 1.0),
                "G": self.params.get('G', 9.81),
                "DT": self.params.get('DT', 0.2),
                "MAX_ITER": self.params.get('MAX_ITER', 5000),
            },
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



class CLMapper(Mapper):
    def __init__(self, width, height, kernel_file, params=None):
        super().__init__(width, height, params)
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        
        with open(kernel_file, "r") as f:
            kernel_code = f.read()
        
        self.params = params.copy() if params else {}
        self.param_order = []
        self.kernel_function_name = 'simulate' # Default kernel function name
        self.default_view_from_kernel = {} # Store default view from kernel comments
        self._parse_kernel_parameters(kernel_code)
        self._parse_kernel_view_defaults(kernel_code)
        
        self.prg = cl.Program(self.ctx, kernel_code).build()
        
        self.num_points = width * height
        self.raw_results_np = np.empty(self.num_points, dtype=np.int32)
        self.mf = cl.mem_flags
        self.output_buf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.raw_results_np.nbytes)
        self.input_xs_np = np.empty(self.num_points, dtype=np.double) # Renamed
        self.input_ys_np = np.empty(self.num_points, dtype=np.double) # Renamed

    def _parse_kernel_parameters(self, kernel_code):
        param_pattern = re.compile(r'//\s*PARAM:\s*(\w+)\s+(\w+)\s+([\d\.\-]+)') # Handles negative defaults
        self.param_order = []
        for line in kernel_code.split('\n'):
            line = line.strip()
            if line.startswith('// PARAM:'):
                match = param_pattern.match(line)
                if match:
                    name, type_str, default_str = match.groups()
                    # Skip width and height as they are passed separately
                    if name in ['width', 'height']:
                        continue 
                    
                    if name not in self.params: # Only set if not already set by dp_map.py args
                        if type_str == 'double':
                            default = float(default_str)
                        elif type_str == 'int':
                            default = int(default_str)
                        else:
                            continue  # Unsupported type
                        self.params[name] = default
                    self.param_order.append(name)
        # Ensure required parameters are present



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

    def compute_map(self):
        x_min, x_max, y_min, y_max = self.params['current_view']
        # initial_xs and initial_ys are used for the kernel arguments
        xs_vals = np.linspace(x_min, x_max, self.width, dtype=np.double)
        ys_vals = np.linspace(y_min, y_max, self.height, dtype=np.double)

        for r_idx in range(self.height):
            for c_idx in range(self.width):
                idx = r_idx * self.width + c_idx
                self.input_xs_np[idx] = xs_vals[c_idx]
                self.input_ys_np[idx] = ys_vals[r_idx]

        input_xs_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.input_xs_np)
        input_ys_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.input_ys_np)

        kernel_args = (
            input_xs_buf,
            input_ys_buf,
            self.output_buf,
            *self.get_kernel_params()
        )
        
        # print("kernel_args types:", [type(a) for a in kernel_args])
        
        global_size = (self.height, self.width) # PyOpenCL expects (rows, cols) for 2D
        kernel_func = getattr(self.prg, self.kernel_function_name)
        kernel_func(self.queue, global_size, None, *kernel_args).wait()
        cl.enqueue_copy(self.queue, self.raw_results_np, self.output_buf).wait()
        return self.raw_results_np.copy()