#include <metal_stdlib>
using namespace metal;

kernel void simulate_pendulum(
    device const float* initial_theta1s [[buffer(0)]],
    device const float* initial_theta2s [[buffer(1)]],
    device int* cycle_counts [[buffer(2)]],
    constant float& L1 [[buffer(3)]],
    constant float& L2 [[buffer(4)]],
    constant float& M1 [[buffer(5)]],
    constant float& M2 [[buffer(6)]],
    constant float& G [[buffer(7)]],
    constant float& DT [[buffer(8)]],
    constant int& MAX_ITERATIONS [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    float th1 = initial_theta1s[gid];
    float th2 = initial_theta2s[gid];
    float w1 = 0.0f;
    float w2 = 0.0f;
    
    int cycles = 0;
    const float PI = 3.14159265358979323846f;
    
    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        float delta_th = th1 - th2;
        
        // Вычисления ускорений
        float num1_1 = -G * (2.0f * M1 + M2) * sin(th1);
        float num1_2 = -M2 * G * sin(th1 - 2.0f * th2);
        float num1_3 = -2.0f * sin(delta_th) * M2;
        float num1_4 = w2 * w2 * L2 + w1 * w1 * L1 * cos(delta_th);
        float den = L1 * (2.0f * M1 + M2 - M2 * cos(2.0f * delta_th));
        
        float alpha1 = (fabs(den) < 1e-6f) ? 0.0f : (num1_1 + num1_2 + num1_3 * num1_4) / den;
        
        float num2_1 = 2.0f * sin(delta_th);
        float num2_2 = w1 * w1 * L1 * (M1 + M2);
        float num2_3 = G * (M1 + M2) * cos(th1);
        float num2_4 = w2 * w2 * L2 * M2 * cos(delta_th);
        float den2 = L2 * (2.0f * M1 + M2 - M2 * cos(2.0f * delta_th));
        
        float alpha2 = (fabs(den2) < 1e-6f) ? 0.0f : (num2_1 * (num2_2 + num2_3 + num2_4)) / den2;
        
        // Интегрирование
        w1 += alpha1 * DT;
        w2 += alpha2 * DT;
        th1 += w1 * DT;
        th2 += w2 * DT;
        
        cycles++;
        
        if (fabs(th1) > 2.0f * PI) break;
    }
    
    cycle_counts[gid] = cycles;
}