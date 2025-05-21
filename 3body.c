// pendulum_kernel.cl (модифицирован для аттрактора Рёсслера)

__kernel void simulate_pendulum(
    __global const double* initial_theta1s, // Используется как массив начальных значений X для Рёсслера
    __global const double* initial_theta2s, // Используется как массив начальных значений Y для Рёсслера
    __global int* cycle_counts,          // Выходной массив: количество итераций до события
    const double L1,                     // Параметр Рёсслера: a
    const double L2,                     // Параметр Рёсслера: b
    const double M1,                     // Параметр Рёсслера: c
    const double M2,                     // Используется для определения количества "установочных" итераций (settle_iterations)
    const double G,                      // Начальное значение Z для Рёсслера
    const double DT,                     // Шаг по времени
    const int MAX_ITERATIONS,            // Максимальное количество итераций
    const int num_points)                // Количество точек для обработки
{
    int gid = get_global_id(0);
    if (gid >= num_points) {
        return;
    }


    // Позиции трех центров притяжения  
    const double2 centers[3] = {  
        (double2)(0.0, 5.0*sqrt(3.0)/3.0),  // Центр 0  
        (double2)(2.5, -5.0*sqrt(3.0)/6.0),// Центр 1  
        (double2)(-2.5, -5.0*sqrt(3.0)/6.0)  // Центр 2  
    };  

    // Силы притяжения из параметров  
    const double forces[3] = {L1, L2, M1};  
    const double friction = M2; // Трение  

    double x = initial_theta1s[gid];  
    double y = initial_theta2s[gid];  
    double vx = 0.0, vy = 0.0;  

    for (int i = 0; i < MAX_ITERATIONS; ++i) {  
        double2 acceleration = (double2)(0.0, 0.0);  

        // Суммируем силы от всех центров  
        for (int c = 0; c < 3; ++c) {  
            double dx = centers[c].x - x;  
            double dy = centers[c].y - y;  
            double distance = hypot(dx, dy);  
            if (distance < 0.1) distance = 0.1; // Защита от деления на ноль  

            // Сила притяжения: F = k * (dx, dy) / distance  
            double k = forces[c] / (distance * distance);  
            acceleration.x += k * dx;  
            acceleration.y += k * dy;  
        }  

        // Добавляем трение  
        acceleration.x -= friction * vx;  
        acceleration.y -= friction * vy;  

        // Интегрирование скорости и позиции  
        vx += acceleration.x * DT;  
        vy += acceleration.y * DT;  
        x += vx * DT;  
        y += vy * DT;  
    }  

    // Определяем ближайший центр  
    int closest = 0;  
    double min_dist = INFINITY;  
    for (int c = 0; c < 3; ++c) {  
        double dx = x - centers[c].x;  
        double dy = y - centers[c].y;  
        double dist = dx*dx + dy*dy; // Квадрат расстояния для оптимизации  
        if (dist < min_dist) {  
            min_dist = dist;  
            closest = c;  
        }  
    }  


    
    cycle_counts[gid] = closest; // Сохраняем общее количество пройденных итераций
}