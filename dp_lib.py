# dp_lib.py
import numpy as np
import pyopencl as cl
import math
import re
import time
import os
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
        
        # Brightness smoothing system
        self.smoothing_frames = 90  # Number of frames for smooth transition
        self.brightness_history = {}  # History of min/max values for each channel
        self.current_smooth_params = {}  # Current smoothing parameters for each channel
        self.target_smooth_params = {}   # Target smoothing parameters for each channel
        self.baseline_smooth_params = {}  # Baseline values for comparison (set every N frames)
        self.smooth_transition_frame = 0  # Frame when transition started
        self.smooth_file_path = ""  # Will be set in setup
        self.frame_counter = 0  # Current frame number
        self.baseline_update_interval = 15  # Update baseline every N frames for comparison
        
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

        # Acceleration factor for the end of phase 1 (adjustable parameter)
        phase1_acceleration = 1.5  # > 1.0 creates acceleration at the end, < 1.0 creates deceleration

        if t <= phase_cutoff:
            # Phase 1: Panning + scale reduction with configurable end acceleration
            t_phase = t / phase_cutoff
            
            # Different easing for center (panning) and scale
            # For scale: use power function to create acceleration at the end
            t_eased_scale = t_phase ** phase1_acceleration
            
            # For center (panning): use deceleration at the end for smooth transition
            # Use inverse power function or cosine-based easing for smooth deceleration
            t_eased_center = 0.5 * (1 - math.cos(t_phase * math.pi))  # Smooth S-curve with deceleration at end
            
            # Center interpolation with deceleration
            current_center_x = start_center[0] + (target_center[0] - start_center[0]) * t_eased_center
            current_center_y = start_center[1] + (target_center[1] - start_center[1]) * t_eased_center
            
            # Scale interpolation with acceleration (unchanged algorithm)
            current_scale_x = start_scale_x - (start_scale_x - phase1_target_scale_x) * t_eased_scale
            current_scale_y = start_scale_y - (start_scale_y - phase1_target_scale_y) * t_eased_scale
            
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
        """ Normalizes raw data with smooth brightness transitions across frames. """
        for channel_name, raw_channel_data in self.raw_data.items():
            # Apply natural logarithm (log1p for zeros: log(1+x))
            log_data = np.log1p(raw_channel_data.astype(np.float64))

            current_min_val = np.min(log_data)
            current_max_val = np.max(log_data)
            
            # Add current frame data to history
            frame_data = {
                'frame': self.frame_counter,
                'min_val': current_min_val,
                'max_val': current_max_val
            }
            
            # Initialize history for this channel if needed
            if channel_name not in self.brightness_history:
                self.brightness_history[channel_name] = []
            
            self.brightness_history[channel_name].append(frame_data)
            
            # Keep only last smoothing_frames
            if len(self.brightness_history[channel_name]) > self.smoothing_frames:
                self.brightness_history[channel_name] = self.brightness_history[channel_name][-self.smoothing_frames:]
            
            # Calculate target values based on percentiles from history
            history = self.brightness_history[channel_name]
            if len(history) >= 3:  # Need at least some history (reduced from 5 to 3)
                min_vals = [h['min_val'] for h in history]
                max_vals = [h['max_val'] for h in history]
                
                # Use more conservative percentiles for stability
                target_min = np.percentile(min_vals, 10)  # 10th percentile for min (was 5th)
                target_max = np.percentile(max_vals, 90)  # 90th percentile for max (was 95th)
                
                # print(f"  >> Percentile calculation: min {np.min(min_vals):.3f}-{np.max(min_vals):.3f} "
                    #   f"-> {target_min:.3f}, max {np.min(max_vals):.3f}-{np.max(max_vals):.3f} -> {target_max:.3f}")
            else:
                target_min = current_min_val
                target_max = current_max_val
                # print(f"  >> Using current values (insufficient history: {len(history)})")
            
            # Initialize smoothing parameters if needed
            if channel_name not in self.current_smooth_params:
                self.current_smooth_params[channel_name] = {
                    'min_val': current_min_val,
                    'max_val': current_max_val
                }
                self.target_smooth_params[channel_name] = {
                    'min_val': target_min,
                    'max_val': target_max
                }
                self.baseline_smooth_params[channel_name] = {
                    'min_val': target_min,
                    'max_val': target_max,
                    'frame': self.frame_counter
                }
                self.smooth_transition_frame = self.frame_counter
            
            # Update baseline periodically or if it doesn't exist
            baseline_params = self.baseline_smooth_params.get(channel_name, {})
            frames_since_baseline = self.frame_counter - baseline_params.get('frame', 0)
            
            if frames_since_baseline >= self.baseline_update_interval or not baseline_params:
                self.baseline_smooth_params[channel_name] = {
                    'min_val': target_min,
                    'max_val': target_max,
                    'frame': self.frame_counter
                }
                baseline_params = self.baseline_smooth_params[channel_name]
                # print(f"  >> Updated baseline at frame {self.frame_counter}: {target_min:.3f}-{target_max:.3f}")
            
            # Compare with baseline instead of previous target
            current_params = self.current_smooth_params[channel_name]
            target_params = self.target_smooth_params[channel_name]
            
            # Calculate changes from baseline (accumulated over time)
            baseline_min_change = abs(target_min - baseline_params['min_val']) / max(abs(baseline_params['min_val']), 1e-10)
            baseline_max_change = abs(target_max - baseline_params['max_val']) / max(abs(baseline_params['max_val']), 1e-10)
            
            # Also calculate immediate changes (for debugging)
            immediate_min_change = abs(target_min - target_params['min_val']) / max(abs(target_params['min_val']), 1e-10)
            immediate_max_change = abs(target_max - target_params['max_val']) / max(abs(target_params['max_val']), 1e-10)
            
            # Debug output
            # print(f"Frame {self.frame_counter:05d} [{channel_name}] "
            #       f"Raw: {current_min_val:.3f}-{current_max_val:.3f} | "
            #       f"Target: {target_min:.3f}-{target_max:.3f} | "
            #       f"Current: {current_params['min_val']:.3f}-{current_params['max_val']:.3f} | "
            #       f"Baseline: {baseline_params['min_val']:.3f}-{baseline_params['max_val']:.3f} | "
            #       f"vs Baseline: {baseline_min_change:.3%}-{baseline_max_change:.3%} | "
            #       f"vs Immediate: {immediate_min_change:.3%}-{immediate_max_change:.3%} | "
            #       f"History: {len(history)}")
            
            # Start transition based on accumulated change from baseline (reduced threshold)
            if baseline_min_change > 0.05 or baseline_max_change > 0.05:  # 5% accumulated change
                # Start new transition
                # print(f"  >> Starting new transition (baseline): {baseline_min_change:.3%} or {baseline_max_change:.3%} > 5%")
                self.target_smooth_params[channel_name] = {
                    'min_val': target_min,
                    'max_val': target_max
                }
                self.smooth_transition_frame = self.frame_counter
                # Update baseline after starting transition
                self.baseline_smooth_params[channel_name] = {
                    'min_val': target_min,
                    'max_val': target_max,
                    'frame': self.frame_counter
                }
            elif immediate_min_change > 0.001 or immediate_max_change > 0.001:  # Small immediate changes
                # Even for small changes, continuously update target slightly
                # print(f"  >> Continuous smoothing: {immediate_min_change:.3%} and {immediate_max_change:.3%} > 1%")
                alpha = 0.1  # Moderate adaptation for small changes
                self.target_smooth_params[channel_name] = {
                    'min_val': target_params['min_val'] * (1 - alpha) + target_min * alpha,
                    'max_val': target_params['max_val'] * (1 - alpha) + target_max * alpha
                }
            # else:
                # print(f"  >> No significant change detected")
            
            # Calculate smooth interpolation - always apply some smoothing
            frames_since_transition = self.frame_counter - self.smooth_transition_frame
            
            # Apply smoothing for a longer period to ensure smooth transitions
            smoothing_period = self.smoothing_frames * 2  # Double the smoothing period
            
            if frames_since_transition < smoothing_period:
                # Interpolate smoothly
                t = frames_since_transition / smoothing_period
                # Use smooth easing function
                t_smooth = 3*t*t - 2*t*t*t  # Smoothstep function
                
                smooth_min = current_params['min_val'] * (1 - t_smooth) + target_params['min_val'] * t_smooth
                smooth_max = current_params['max_val'] * (1 - t_smooth) + target_params['max_val'] * t_smooth
                
                # print(f"  >> Interpolating: t={t:.3f}, t_smooth={t_smooth:.3f}, "
                    #   f"frames_since={frames_since_transition}/{smoothing_period}")
            else:
                # Apply continuous light smoothing even after transition
                alpha = 0.002  # Very small constant smoothing factor
                smooth_min = current_params['min_val'] * (1 - alpha) + target_params['min_val'] * alpha
                smooth_max = current_params['max_val'] * (1 - alpha) + target_params['max_val'] * alpha
                # print(f"  >> Continuous smoothing: alpha={alpha}")
            
            # Update current parameters
            self.current_smooth_params[channel_name] = {
                'min_val': smooth_min,
                'max_val': smooth_max
            }
            
            # print(f"  >> Final smooth: {smooth_min:.3f}-{smooth_max:.3f}")
            
            # Normalize using smooth parameters
            if smooth_max == smooth_min:
                normalized_channel_data = np.zeros_like(log_data, dtype=np.uint8)
            else:
                normalized_channel_data = 255 * (log_data - smooth_min) / (smooth_max - smooth_min)
            
            self.normalized_data[channel_name] = np.clip(normalized_channel_data, 0, 255).astype(np.uint8)

        return self.normalized_data

    def set_output_channels(self, channels):
        """Sets the names of the output channels the kernel will produce (e.g., ['H', 'S', 'B'] or ['R', 'G', 'B'])."""
        self.output_channels = channels

    def get_output_channels(self):
        return self.output_channels

    def setup_brightness_smoothing(self, smooth_file_path, current_frame, is_continuing_from_frame):
        """Setup brightness smoothing system with file persistence."""
        self.smooth_file_path = smooth_file_path
        self.frame_counter = current_frame
        
        # Load smoothing data if continuing from a previous session
        if is_continuing_from_frame and os.path.exists(smooth_file_path):
            try:
                with open(smooth_file_path, 'r') as f:
                    smooth_data = json.load(f)
                
                # Check if the data is for the correct frame
                if smooth_data.get('frame', -1) == current_frame - 1:
                    self.current_smooth_params = smooth_data.get('current_smooth_params', {})
                    self.target_smooth_params = smooth_data.get('target_smooth_params', {})
                    self.baseline_smooth_params = smooth_data.get('baseline_smooth_params', {})
                    self.smooth_transition_frame = smooth_data.get('smooth_transition_frame', current_frame)
                    self.brightness_history = smooth_data.get('brightness_history', {})
                    print(f"Loaded smoothing parameters from {smooth_file_path} for frame {current_frame - 1}")
                else:
                    print(f"Smoothing file frame mismatch. Expected {current_frame - 1}, got {smooth_data.get('frame', -1)}. Starting fresh.")
                    self._reset_smoothing_state()
            except (json.JSONDecodeError, KeyError, IOError) as e:
                print(f"Failed to load smoothing file {smooth_file_path}: {e}. Starting fresh.")
                self._reset_smoothing_state()
        else:
            self._reset_smoothing_state()

    def _reset_smoothing_state(self):
        """Reset all smoothing state to initial values."""
        self.current_smooth_params = {}
        self.target_smooth_params = {}
        self.baseline_smooth_params = {}
        self.brightness_history = {}
        self.smooth_transition_frame = self.frame_counter

    def save_brightness_smoothing(self):
        """Save current brightness smoothing parameters to file."""
        if not self.smooth_file_path:
            return
        
        smooth_data = {
            'frame': self.frame_counter,
            'current_smooth_params': self.current_smooth_params,
            'target_smooth_params': self.target_smooth_params,
            'baseline_smooth_params': self.baseline_smooth_params,
            'smooth_transition_frame': self.smooth_transition_frame,
            'brightness_history': self.brightness_history
        }
        
        try:
            with open(self.smooth_file_path, 'w') as f:
                json.dump(smooth_data, f, indent=2)
        except IOError as e:
            print(f"Failed to save smoothing file {self.smooth_file_path}: {e}")

    def update_frame_counter(self, frame_number):
        """Update the current frame number for smoothing calculations."""
        self.frame_counter = frame_number

    def add_keyframe(self, filename, view_width_absolute):
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
            "target_view_width": view_width_absolute,
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
        
        # if timer > 1.0:
            # print(f"Normalize timer {timer}")

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

