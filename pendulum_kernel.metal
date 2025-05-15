#include <metal_stdlib>
using namespace metal;

kernel void simulate_pendulum(
    const device double* initial_theta1s [[buffer(0)]],
    const device double* initial_theta2s [[buffer(1)]],
    device int* cycle_counts [[buffer(2)]],
    constant double& L1 [[buffer(3)]],
    constant double& L2 [[buffer(4)]],
    constant double& M1 [[buffer(5)]],
    constant double& M2 [[buffer(6)]],
    constant double& G [[buffer(7)]],
    constant double& DT [[buffer(8)]],
    constant int& MAX_ITERATIONS [[buffer(9)]],
    uint id [[thread_position_in_grid]])
{
    const double M_PI_F = 3.14159265358979323846;
    
    double th1 = initial_theta1s[id];
    double th2 = initial_theta2s[id];
    double w1 = 0.0; // Начальная угловая скорость первого сегмента
    double w2 = 0.0; // Начальная угловая скорость второго сегмента

    int cycles = 0;

    for (int i = 0; i < MAX_ITERATIONS; ++i) 
    {
        // Уравнения движения для двойного маятника
        double delta_th = th1 - th2;

        double den1 = (M1 + M2) * L1 - M2 * L1 * cos(delta_th) * cos(delta_th);
        
        // Ускорения alpha1, alpha2
        double num1_1 = -G * (2.0 * M1 + M2) * sin(th1);
        double num1_2 = -M2 * G * sin(th1 - 2.0 * th2);
        double num1_3 = -2.0 * sin(th1 - th2) * M2;
        double num1_4 = w2 * w2 * L2 + w1 * w1 * L1 * cos(th1 - th2);
        double den = L1 * (2.0 * M1 + M2 - M2 * cos(2.0 * th1 - 2.0 * th2));

        double alpha1, alpha2;

        if (fabs(den) < 1e-6) { // Избегаем деления на ноль
             alpha1 = 0.0;
        } else {
            alpha1 = (num1_1 + num1_2 + num1_3 * num1_4) / den;
        }

        double num2_1 = 2.0 * sin(th1 - th2);
        double num2_2 = w1 * w1 * L1 * (M1 + M2);
        double num2_3 = G * (M1 + M2) * cos(th1);
        double num2_4 = w2 * w2 * L2 * M2 * cos(th1 - th2);
        double den2_equiv = L2 * (2.0 * M1 + M2 - M2 * cos(2.0 * th1 - 2.0 * th2));

        if (fabs(den2_equiv) < 1e-6) { // Избегаем деления на ноль
            alpha2 = 0.0;
        } else {
             alpha2 = (num2_1 * (num2_2 + num2_3 + num2_4)) / den2_equiv;
        }

        // Интегрирование методом Эйлера
        w1 += alpha1 * DT;
        w2 += alpha2 * DT;
        th1 += w1 * DT;
        th2 += w2 * DT;

        cycles++;

        if (fabs(th1) > 2.0 * M_PI_F) {
            break;
        }
    }
    
    cycle_counts[id] = cycles;
}