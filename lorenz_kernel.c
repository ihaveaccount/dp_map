// pendulum_kernel.cl

__kernel void simulate_pendulum(
__global const double* initial_theta1s, // x0 (используется как начальная координата X)
__global const double* initial_theta2s, // y0 (используется как начальная координата Y)
__global int* cycle_counts, // Количество пересечений плоскости X=0
const double L1, // Игнорируется
const double L2, // Игнорируется
const double M1, // Игнорируется
const double M2, // Игнорируется
const double G, // Игнорируется
const double DT,
const int MAX_ITERATIONS,
const int num_points)
{
    int gid = get_global_id(0);
    if (gid >= num_points) return;
    // Параметры аттрактора Лоренца  
    const double sigma = L1;  
    const double rho = L2;  
    const double beta = M1;  
    const double brightness_scale = M2;  

    // Начальные условия  
    double x = initial_theta1s[gid];  
    double y = initial_theta2s[gid];  
    double z = 25.0;  // Фиксированное Z (типичное для Лоренца)  

    double energy_sum = 0.0;  
    int cross_count = 0;  
    double prev_z = z;  

    for (int i = 0; i < MAX_ITERATIONS; ++i) {  
        // Уравнения Лоренца  
        double dx = sigma * (y - x);  
        double dy = x * (rho - z) - y;  
        double dz = x * y - beta * z;  

        // Интегрирование методом Рунге-Кутты 4-го порядка  
        double k1x = dx * DT;  
        double k1y = dy * DT;  
        double k1z = dz * DT;  

        double k2x = sigma * ((y + k1y/2) - (x + k1x/2)) * DT;  
        double k2y = ((x + k1x/2) * (rho - (z + k1z/2)) - (y + k1y/2)) * DT;  
        double k2z = ((x + k1x/2) * (y + k1y/2) - beta * (z + k1z/2)) * DT;  

        double k3x = sigma * ((y + k2y/2) - (x + k2x/2)) * DT;  
        double k3y = ((x + k2x/2) * (rho - (z + k2z/2)) - (y + k2y/2)) * DT;  
        double k3z = ((x + k2x/2) * (y + k2y/2) - beta * (z + k2z/2)) * DT;  

        double k4x = sigma * ((y + k3y) - (x + k3x)) * DT;  
        double k4y = ((x + k3x) * (rho - (z + k3z)) - (y + k3y)) * DT;  
        double k4z = ((x + k3x) * (y + k3y) - beta * (z + k3z)) * DT;  

        x += (k1x + 2*k2x + 2*k3x + k4x) / 6.0;  
        y += (k1y + 2*k2y + 2*k3y + k4y) / 6.0;  
        z += (k1z + 2*k2z + 2*k3z + k4z) / 6.0;  

        // Условия подсчёта:  
        // 1. Пересечения Z=25 (баланс между "крыльями" аттрактора)  
        if ((prev_z - 25.0) * (z - 25.0) < 0.0) cross_count++;  
        // 2. Накопление "энергии" (x² + y² + z²)  
        energy_sum += (x*x + y*y + z*z) * DT;  
        prev_z = z;  
    }  

// Комбини
    cycle_counts[gid] = cross_count;
}