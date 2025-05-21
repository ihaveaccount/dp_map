// pendulum_kernel.cl

#define M_PI_F 3.14159265358979323846f

__kernel void simulate_pendulum(
    __global const double* initial_theta1s, // Параметр r (например, для логистического отображения)
    __global const double* initial_theta2s, // Доп. параметр системы
    __global int* cycle_counts,          // Показатель Ляпунова (целочисленное представление)
    const double L1,                      // Масштабирующий коэффициент для theta1
    const double L2,                      // Масштабирующий коэффициент для theta2
    const double M1,                      // Весовой множитель 1
    const double M2,                      // Весовой множитель 2
    const double G,                       // Константа системы
    const double DT,                      // Шаг времени (используется как множитель)
    const int MAX_ITERATIONS,
    const int num_points)
{
    int gid = get_global_id(0);
    if (gid >= num_points) return;

    double r = initial_theta1s[gid] * L1; // Основной параметр системы
    double c = initial_theta2s[gid] * L2; // Дополнительный параметр
    double x1 = 0.5;                     // Начальное условие
    double x2 = x1 + 1e-5;               // Возмущенное начальное условие
    double sum_lyapunov = 0.0;
    double epsilon = 1e-5;

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        // Модифицированная динамическая система с использованием всех параметров
        double x1_next = r * x1 * (1 - x1) + c * M1 * sin(G * x1) * DT / (M2 + 1.0);
        double x2_next = r * x2 * (1 - x2) + c * M1 * sin(G * x2) * DT / (M2 + 1.0);
        
        double diff = fabs(x2_next - x1_next);
        if (diff < 1e-10) break;
        
        sum_lyapunov += log(diff / epsilon);
        
        // Перенормировка траекторий
        x1 = x1_next;
        x2 = x1_next + epsilon * (x2_next - x1_next) / diff;
    }

    // Вычисление показателя Ляпунова и преобразование в целое
    double lambda = sum_lyapunov / MAX_ITERATIONS;
    cycle_counts[gid] = (int)(fabs(lambda) * 1000);
}