
import numpy as np
import pyopencl as cl
import os
import pygame
import math
import Metal as metal
import Foundation

from PIL import Image


# --- Интерфейс IMappable и его реализация ---
class IMappable:
    def get_initial_ranges(self):
        """Возвращает начальные диапазоны для X и Y."""
        raise NotImplementedError

    def get_kernel_params(self):
        """Возвращает параметры, специфичные для функции, для передачи в OpenCL ядро."""
        raise NotImplementedError

    def update_params(self, key):
        """Обновляет параметры на основе нажатой клавиши."""
        raise NotImplementedError

    def print_params(self):
        """Выводит текущие параметры в консоль."""
        raise NotImplementedError
    
    def get_param_string(self):
        """Возвращает строку с текущими параметрами."""
        raise NotImplementedError




class DoublePendulum(IMappable):
    def __init__(self, L1=1.0, L2=1.0, M1=1.0, M2=1.0, G=9.81, DT=0.2, MAX_ITER=5000, theta1_min=-np.pi, theta1_max=np.pi, theta2_min=-np.pi, theta2_max=np.pi):
        self.L1_initial, self.L2_initial, self.M1_initial, self.M2_initial = L1, L2, M1, M2
        self.L1, self.L2, self.M1, self.M2 = L1, L2, M1, M2
        self.G = G
        self.DT = DT
        self.MAX_ITER = MAX_ITER

        self.theta1_min_initial, self.theta1_max_initial = theta1_min, theta1_max
        self.theta2_min_initial, self.theta2_max_initial = theta2_min, theta2_max
        
        self.current_view_x_min, self.current_view_x_max = self.theta1_min_initial, self.theta1_max_initial
        self.current_view_y_min, self.current_view_y_max = self.theta2_min_initial, self.theta2_max_initial

        self.print_params()

    def get_initial_ranges(self):
        return (self.theta1_min_initial, self.theta1_max_initial), \
               (self.theta2_min_initial, self.theta2_max_initial)

    def get_current_view_ranges(self):
         return (self.current_view_x_min, self.current_view_x_max), \
                (self.current_view_y_min, self.current_view_y_max)

    def set_current_view_ranges(self, x_min, x_max, y_min, y_max):
        self.current_view_x_min, self.current_view_x_max = x_min, x_max
        self.current_view_y_min, self.current_view_y_max = y_min, y_max

    def reset_view_ranges(self):
        self.current_view_x_min, self.current_view_x_max = self.theta1_min_initial, self.theta1_max_initial
        self.current_view_y_min, self.current_view_y_max = self.theta2_min_initial, self.theta2_max_initial
        print("View reset to initial ranges.")
        self.print_params()


    def get_kernel_params(self):
        return np.double(self.L1), np.double(self.L2), \
               np.double(self.M1), np.double(self.M2), \
               np.double(self.G), np.double(self.DT), \
               np.int32(self.MAX_ITER)

    def update_params(self, key):
        changed = False
        param_step_L = 0.1
        param_step_M = 0.1

        
        if changed:
            self.print_params()
        return changed

    def print_params(self):
        print(self.get_param_string())

    def get_param_string(self):
        return (f"L1(Q/W): {self.L1:.2f}, L2(A/S): {self.L2:.2f}, "
                f"M1(Z/X): {self.M1:.2f}, M2(C/V): {self.M2:.2f}\n"
                f"View : {self.current_view_x_min}, {self.current_view_x_max}, {self.current_view_y_min}, {self.current_view_y_max} (R to reset view)")


class MapperOpenCL:
    """Класс для вычисления карты двойного маятника с использованием OpenCL."""

    def __init__(self, width, height, mappable_function):
        self.width = width
        self.height = height
        self.mappable = mappable_function

        # --- OpenCL Setup ---
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        
        kernel_path = os.path.join(os.path.dirname(__file__), "pendulum_kernel.c")
        with open(kernel_path, "r") as f:
            kernel_code = f.read()
        self.prg = cl.Program(self.ctx, kernel_code).build()

        # --- Buffers ---
        self.num_points = self.width * self.height
        # Выходной буфер для результатов с GPU
        self.raw_results_np = np.empty(self.num_points, dtype=np.int32)
        self.mf = cl.mem_flags
        self.output_buf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.raw_results_np.nbytes)
        
        # Входные буферы (будут создаваться при каждом вызове, т.к. их содержимое меняется)
        self.input_theta1_np = np.empty(self.num_points, dtype=np.double)
        self.input_theta2_np = np.empty(self.num_points, dtype=np.double)

    def compute_map_raw(self, x_min, x_max, y_min, y_max):
        """ Вычисляет сырые данные карты (количество циклов) """
        theta1_vals = np.linspace(x_min, x_max, self.width, dtype=np.double)
        theta2_vals = np.linspace(y_min, y_max, self.height, dtype=np.double)

        # Заполнение входных массивов: каждому пикселю своя пара (theta1, theta2)
        # Пиксель (col, row) -> (theta1_vals[col], theta2_vals[row])
        # Индекс в 1D массиве: idx = row * self.width + col
        for r_idx in range(self.height):
            for c_idx in range(self.width):
                idx = r_idx * self.width + c_idx
                self.input_theta1_np[idx] = theta1_vals[c_idx]
                self.input_theta2_np[idx] = theta2_vals[r_idx]
        
        # Создание и копирование входных буферов на GPU
        input_theta1_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.input_theta1_np)
        input_theta2_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.input_theta2_np)

        kernel_args = (
            input_theta1_buf, input_theta2_buf, self.output_buf,
            *self.mappable.get_kernel_params(), # L1, L2, M1, M2, G, DT, MAX_ITER
            np.int32(self.num_points)
        )
        
        self.prg.simulate_pendulum(self.queue, (self.num_points,), None, *kernel_args).wait()
        cl.enqueue_copy(self.queue, self.raw_results_np, self.output_buf).wait()
        
        return self.raw_results_np.copy() # Возвращаем копию, чтобы избежать проблем с изменением

    def normalize_data(self, raw_data):
        """ Нормализует сырые данные (логарифм + min-max масштабирование в 0-255). """
        # Применяем натуральный логарифм (log1p для обработки нулей: log(1+x))
        log_data = np.log1p(raw_data.astype(np.double))

        min_log_val = np.min(log_data)
        max_log_val = np.max(log_data)
        
        if max_log_val == min_log_val: # Если все значения одинаковы
            normalized_data = np.zeros_like(log_data, dtype=np.uint8)
        else:
            normalized_data = 255 * (log_data - min_log_val) / (max_log_val - min_log_val)
        
        normalized_data = np.clip(normalized_data, 0, 255).astype(np.uint8)
        return normalized_data.reshape((self.height, self.width))



class MapperMetal:
    """Класс для вычисления карты двойного маятника с использованием Metal."""

    def __init__(self, width, height, mappable_function):
        self.width = width
        self.height = height
        self.mappable = mappable_function

        # --- Metal Setup ---
        self.device = metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Metal device not found")
            
        # Загружаем шейдеры
        shader_path = os.path.join(os.path.dirname(__file__), "pendulum_kernel.metal")
        self.library = self.device.newLibraryWithFile_error_(shader_path, None)[0]
        if self.library is None:
            raise RuntimeError("Failed to load Metal library")
            
        self.command_queue = self.device.newCommandQueue()
        self.kernel_function = self.library.newFunctionWithName_("simulate_pendulum")
        self.pipeline_state = self.device.newComputePipelineStateWithFunction_error_(self.kernel_function, None)[0]
        
        # --- Буферы ---
        self.num_points = self.width * self.height
        # Выходной буфер для результатов
        self.raw_results_np = np.empty(self.num_points, dtype=np.int32)
        self.output_buffer = self.device.newBufferWithLength_options_(self.raw_results_np.nbytes, metal.MTLResourceStorageModeShared)
        
        # Входные буферы (будут создаваться при каждом вызове)
        self.input_theta1_np = np.empty(self.num_points, dtype=np.float64)
        self.input_theta2_np = np.empty(self.num_points, dtype=np.float64)
        
    def compute_map_raw(self, x_min, x_max, y_min, y_max):
        """ Вычисляет сырые данные карты (количество циклов) """
        theta1_vals = np.linspace(x_min, x_max, self.width, dtype=np.float64)
        theta2_vals = np.linspace(y_min, y_max, self.height, dtype=np.float64)

        # Заполнение входных массивов: каждому пикселю своя пара (theta1, theta2)
        for r_idx in range(self.height):
            for c_idx in range(self.width):
                idx = r_idx * self.width + c_idx
                self.input_theta1_np[idx] = theta1_vals[c_idx]
                self.input_theta2_np[idx] = theta2_vals[r_idx]
        
        # Создание буферов для Metal
        input_theta1_buffer = self.device.newBufferWithBytes_length_options_(
            self.input_theta1_np, self.input_theta1_np.nbytes, metal.MTLResourceStorageModeShared)
        input_theta2_buffer = self.device.newBufferWithBytes_length_options_(
            self.input_theta2_np, self.input_theta2_np.nbytes, metal.MTLResourceStorageModeShared)
            
        # Параметры маятника
        L1, L2, M1, M2, G, DT, MAX_ITER = self.mappable.get_kernel_params()
        
        # Создаем буферы для параметров
        L1_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([L1], dtype=np.float64), 8, metal.MTLResourceStorageModeShared)
        L2_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([L2], dtype=np.float64), 8, metal.MTLResourceStorageModeShared)
        M1_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([M1], dtype=np.float64), 8, metal.MTLResourceStorageModeShared)
        M2_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([M2], dtype=np.float64), 8, metal.MTLResourceStorageModeShared)
        G_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([G], dtype=np.float64), 8, metal.MTLResourceStorageModeShared)
        DT_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([DT], dtype=np.float64), 8, metal.MTLResourceStorageModeShared)
        MAX_ITER_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([MAX_ITER], dtype=np.int32), 4, metal.MTLResourceStorageModeShared)
        
        # Создание и настройка команды
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        compute_encoder.setComputePipelineState_(self.pipeline_state)
        
        # Настройка буферов
        compute_encoder.setBuffer_offset_atIndex_(input_theta1_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(input_theta2_buffer, 0, 1)
        compute_encoder.setBuffer_offset_atIndex_(self.output_buffer, 0, 2)
        compute_encoder.setBuffer_offset_atIndex_(L1_buffer, 0, 3)
        compute_encoder.setBuffer_offset_atIndex_(L2_buffer, 0, 4)
        compute_encoder.setBuffer_offset_atIndex_(M1_buffer, 0, 5)
        compute_encoder.setBuffer_offset_atIndex_(M2_buffer, 0, 6)
        compute_encoder.setBuffer_offset_atIndex_(G_buffer, 0, 7)
        compute_encoder.setBuffer_offset_atIndex_(DT_buffer, 0, 8)
        compute_encoder.setBuffer_offset_atIndex_(MAX_ITER_buffer, 0, 9)
        
        # Вычисление количества потоков и групп
        threads_per_group = metal.MTLSize(width=256, height=1, depth=1)
        num_groups = metal.MTLSize(
            width=(self.num_points + threads_per_group.width - 1) // threads_per_group.width,
            height=1,
            depth=1
        )
        
        # Запуск вычислений
        compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)
        compute_encoder.endEncoding()
        
        # Выполнение команды и ожидание завершения
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        # Копирование результатов
        result_ptr = self.output_buffer.contents()
        np.copyto(self.raw_results_np, np.ctypeslib.as_array(result_ptr, shape=(self.num_points,)).astype(np.int32))
        
        return self.raw_results_np.copy()
        
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





def create_mapper(width, height, pendulum, backend):
    """Creates the appropriate mapper based on the selected backend."""
    if backend == 'opencl':
        return MapperOpenCL(width, height, pendulum)
    elif backend == 'metal':
        
        return MapperMetal(width, height, pendulum)
    else:
        raise ValueError(f"Unsupported backend: {backend}")



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
