// pendulum_kernel.cl

#define M_PI_F 3.14159265358979323846f

__kernel void simulate_pendulum(
    __global const double* initial_theta1s, // Массив начальных углов theta1 для каждой точки
    __global const double* initial_theta2s, // Массив начальных углов theta2 для каждой точки
    __global int* cycle_counts,          // Выходной массив: количество циклов для каждой точки
    const double L1,
    const double L2,
    const double M1,
    const double M2,
    const double G,
    const double DT,
    const int MAX_ITERATIONS,
    const int width,
    const int height)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }
    int gid = row * width + col;

    double th1 = initial_theta1s[gid];
    double th2 = initial_theta2s[gid];
    double w1 = 0.0f; // Начальная угловая скорость первого сегмента
    double w2 = 0.0f; // Начальная угловая скорость второго сегмента

    int cycles = 0;

    for (int i = 0; i < MAX_ITERATIONS; ++i) 
    {
        // Уравнения движения для двойного маятника (упрощенные для читаемости)
        // Источник: https://www.myphysicslab.com/pendulum/double-pendulum-en.html
        // (с поправкой на обозначения M1, M2 как массы грузов, а не стержней)

        double delta_th = th1 - th2;

        double den1 = (M1 + M2) * L1 - M2 * L1 * cos(delta_th) * cos(delta_th); // немного не то, это для m_rod=0
        // Используем более общую форму
        
        // Ускорения alpha1, alpha2
        double num1_1 = -G * (2.0f * M1 + M2) * sin(th1);
        double num1_2 = -M2 * G * sin(th1 - 2.0f * th2);
        double num1_3 = -2.0f * sin(th1 - th2) * M2;
        double num1_4 = w2 * w2 * L2 + w1 * w1 * L1 * cos(th1 - th2);
        double den = L1 * (2.0f * M1 + M2 - M2 * cos(2.0f * th1 - 2.0f * th2));

        double alpha1, alpha2;

        if (fabs(den) < 1e-6f) { // Избегаем деления на ноль
             alpha1 = 0.0f;
        } else {
            alpha1 = (num1_1 + num1_2 + num1_3 * num1_4) / den;
        }


        double num2_1 = 2.0f * sin(th1 - th2);
        double num2_2 = w1 * w1 * L1 * (M1 + M2);
        double num2_3 = G * (M1 + M2) * cos(th1);
        double num2_4 = w2 * w2 * L2 * M2 * cos(th1 - th2);
        // den общий, но для alpha2 делится на L2 * (...)
        double den2_equiv = L2 * (2.0f * M1 + M2 - M2 * cos(2.0f * th1 - 2.0f * th2));


        if (fabs(den2_equiv) < 1e-6f) { // Избегаем деления на ноль
            alpha2 = 0.0f;
        } else {
             alpha2 = (num2_1 * (num2_2 + num2_3 + num2_4)) / den2_equiv;
        }

        // Интегрирование методом Эйлера
        w1 += alpha1 * DT;
        w2 += alpha2 * DT;
        th1 += w1 * DT;
        th2 += w2 * DT;

        cycles++;

        if (fabs(th1) > 2.0f * M_PI_F) {
            break;
        }
    }

    cycle_counts[gid] = cycles;
}