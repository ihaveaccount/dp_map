
import numpy as np
import pyopencl as cl
import os
import math
import pathlib
import re
from PIL import Image
import ctypes
from scipy.ndimage import median_filter

import json






class Mapper:
    def __init__(self, width, height, params=None):
        self.width = width
        self.height = height
        self.params = params or {}
        
    @staticmethod
    
    def interpolate(anim):
        t = anim['step'] / anim['total_steps']
        # Начальные и целевые границы
        start_vx_min, start_vx_max, start_vy_min, start_vy_max = anim['start_view']
        target_vx_min, target_vx_max, target_vy_min, target_vy_max = anim['target_view']
        phase_cutoff = 0.05  # 5% времени на центрирование
        

        # Исходные параметры
        start_scale_x = start_vx_max - start_vx_min
        target_scale_x = target_vx_max - target_vx_min
        start_scale_y = start_vy_max - start_vy_min
        target_scale_y = target_vy_max - target_vy_min

        start_center = ((start_vx_min + start_vx_max)/2, 
                    (start_vy_min + start_vy_max)/2)
        target_center = ((target_vx_min + target_vx_max)/2, 
                        (target_vy_min + target_vy_max)/2)

        # Рассчитываем целевой масштаб для фазы 1 (уменьшение в 2 раза)
        phase1_target_scale_x = start_scale_x / 2
        phase1_target_scale_y = start_scale_y / 2

        if t <= phase_cutoff:
            # Фаза 1: Панорамирование + уменьшение масштаба
            t_phase = t / phase_cutoff
            t_eased = 0.5 * (1 - math.cos(t_phase * math.pi))  # Плавное ускорение
            
            # Интерполяция центра
            current_center_x = start_center[0] + (target_center[0] - start_center[0]) * t_eased
            current_center_y = start_center[1] + (target_center[1] - start_center[1]) * t_eased
            
            # Интерполяция масштаба до уменьшенного в 2 раза
            current_scale_x = start_scale_x - (start_scale_x - phase1_target_scale_x) * t_eased
            current_scale_y = start_scale_y - (start_scale_y - phase1_target_scale_y) * t_eased
            
        else:
            # Фаза 2: Экспоненциальное увеличение
            t_phase = (t - phase_cutoff) / (1 - phase_cutoff)
            
            # Начальный масштаб - результат фазы 1
            start_scale_phase2_x = phase1_target_scale_x
            start_scale_phase2_y = phase1_target_scale_y
            
            # Экспоненциальная интерполяция
            current_scale_x = start_scale_phase2_x * (target_scale_x / start_scale_phase2_x) ** t_phase
            current_scale_y = start_scale_phase2_y * (target_scale_y / start_scale_phase2_y) ** t_phase
            
            # Центр фиксируется на целевом
            current_center_x = target_center[0]
            current_center_y = target_center[1]

        # Рассчет границ области
        interp_x_min = current_center_x - current_scale_x / 2
        interp_x_max = current_center_x + current_scale_x / 2
        interp_y_min = current_center_y - current_scale_y / 2
        interp_y_max = current_center_y + current_scale_y / 2

        # Гарантия, что целевая область остается в кадре
        interp_x_min = max(interp_x_min, target_vx_min)
        interp_x_max = min(interp_x_max, target_vx_max)
        interp_y_min = max(interp_y_min, target_vy_min)
        interp_y_max = min(interp_y_max, target_vy_max)

    
        # Рассчитываем границы
        interp_x_min = current_center_x - current_scale_x / 2
        interp_x_max = current_center_x + current_scale_x / 2
        interp_y_min = current_center_y - current_scale_y / 2
        interp_y_max = current_center_y + current_scale_y / 2


        return interp_x_min, interp_x_max, interp_y_min, interp_y_max

    def compute_map(self, x_min, x_max, y_min, y_max):
        raise NotImplementedError

    def normalize_data(self, raw_data):
        """ Нормализует сырые данные (логарифм + min-max масштабирование в 0-255). """
        # Применяем натуральный логарифм (log1p для обработки нулей: log(1+x))
        log_data = np.log1p(raw_data.astype(np.float64))

        min_log_val = np.min(log_data)
        max_log_val = np.max(log_data)
        
        if max_log_val == min_log_val: # Если все значения одинаковы
            normalized_data = np.zeros_like(log_data, dtype=np.uint8)
        else:
            normalized_data = 255 * (log_data - min_log_val) / (max_log_val - min_log_val)
        
        normalized_data = np.clip(normalized_data, 0, 255).astype(np.uint8)
        return normalized_data.reshape((self.height, self.width))

    
    def save_state(self, filename, target_point=None):
        state = {
            "params": self.params,
            "target_point": target_point
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, filename, existing_mapper=None):
        """Загружает состояние в существующий экземпляр Mapper"""
        with open(filename) as f:
            state = json.load(f)
        
        # Если передан существующий маппер - обновляем его параметры
        if existing_mapper:
            existing_mapper.params.update(state.get("params", {}))
            return existing_mapper, state.get("target_point")
        
        # Для обратной совместимости (лучше всегда использовать existing_mapper)
        dummy_mapper = cls(1, 1, state.get("params"))
        return dummy_mapper, state.get("target_point")
    

    def get_current_view(self):
        return self.params["current_view"]


    def set_current_view(self, x_min, x_max, y_min, y_max):
        self.params['current_view'] = (x_min, x_max, y_min, y_max)





class DPMapper(Mapper):
    def __init__(self, width, height, kernel_file, params=None):
        super().__init__(width, height, params)
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        
        with open(kernel_file, "r") as f:
            kernel_code = f.read()
        self.prg = cl.Program(self.ctx, kernel_code).build()
        
        self.num_points = width * height
        self.raw_results_np = np.empty(self.num_points, dtype=np.int32)
        self.mf = cl.mem_flags
        self.output_buf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.raw_results_np.nbytes)
        self.input_theta1_np = np.empty(self.num_points, dtype=np.double)
        self.input_theta2_np = np.empty(self.num_points, dtype=np.double)
        self.params = params or {}

    def get_kernel_params(self):
        return (
            np.double(self.params.get('L1', 1.0)),
            np.double(self.params.get('L2', 1.0)),
            np.double(self.params.get('M1', 1.0)),
            np.double(self.params.get('M2', 1.0)),
            np.double(self.params.get('G', 9.81)),
            np.double(self.params.get('DT', 0.2)),
            np.int32(self.params.get('MAX_ITER', 5000))
        )

    def compute_map(self):
        x_min, x_max, y_min, y_max = self.params['current_view']
        theta1_vals = np.linspace(x_min, x_max, self.width, dtype=np.double)
        theta2_vals = np.linspace(y_min, y_max, self.height, dtype=np.double)

        for r_idx in range(self.height):
            for c_idx in range(self.width):
                idx = r_idx * self.width + c_idx
                self.input_theta1_np[idx] = theta1_vals[c_idx]
                self.input_theta2_np[idx] = theta2_vals[r_idx]

        input_theta1_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.input_theta1_np)
        input_theta2_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.input_theta2_np)

        kernel_args = (
            input_theta1_buf, input_theta2_buf, self.output_buf,
            *self.get_kernel_params(),
            np.int32(self.num_points)
        )
        
        self.prg.simulate_pendulum(self.queue, (self.num_points,), None, *kernel_args).wait()
        cl.enqueue_copy(self.queue, self.raw_results_np, self.output_buf).wait()
        return self.raw_results_np.copy()





