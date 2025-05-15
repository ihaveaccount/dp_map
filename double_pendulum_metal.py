import numpy as np
import metal
import Foundation
import os
import pygame
import math
from dp_lib import IMappable, DoublePendulum

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
        shader_path = os.path.join(os.path.dirname(__file__), "pendulum.metal")
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
