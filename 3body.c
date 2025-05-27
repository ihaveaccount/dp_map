// magnetic_pendulum_kernel.c
// PARAM: R double 1.0 // Радиус треугольника (расстояние от центра до вершин)
// PARAM: L double 2.0 // Длина маятника
// PARAM: GRAVITY double 1.0 // Нормализованное гравитационное ускорение
// PARAM: MAGNETIC_FORCE double 0.5 // Магнитная сила притяжения
// PARAM: FRICTION double 0.2 // Коэффициент трения воздуха
// PARAM: DT double 0.02 // Шаг интегрирования
// PARAM: MAX_ITERATIONS int 8000 // Максимальное число итераций
// PARAM: CONVERGENCE_THRESHOLD double 0.1 // Пороговое расстояние для сходимости
// PARAM: MIN_VELOCITY double 0.005 // Минимальная скорость для определения остановки
// VIEW_DEFAULT: x_min -15
// VIEW_DEFAULT: x_max 15
// VIEW_DEFAULT: y_min -15
// VIEW_DEFAULT: y_max 15
// OUTPUT_CHANNELS: R, G, B

__kernel void simulate(
    __global const double* initial_xs, // Исходные X координаты
    __global const double* initial_ys, // Исходные Y координаты
    __global float* out_R, // Выходной красный канал
    __global float* out_G, // Выходной зелёный канал
    __global float* out_B, // Выходной синий канал
    const double R,
    const double L,
    const double GRAVITY,
    const double MAGNETIC_FORCE,
    const double FRICTION,
    const double DT,
    const int MAX_ITERATIONS,
    const double CONVERGENCE_THRESHOLD,
    const double MIN_VELOCITY,
    const int width,
    const int height)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= height || col >= width) return;
    int gid = row * width + col;

    // Координаты трёх магнитов (равносторонний треугольник с центром в (0,0))
    double2 magnets[3];
    magnets[0] = (double2)(0.0, R);                        // Красный магнит (верхняя вершина)
    magnets[1] = (double2)(-R * 0.866025403784, -R * 0.5); // Зелёный магнит (левая нижняя)
    magnets[2] = (double2)(R * 0.866025403784, -R * 0.5);  // Синий магнит (правая нижняя)

    // Начальные условия маятника
    double2 pos = (double2)(initial_xs[gid], initial_ys[gid]); // Позиция груза
    double2 vel = (double2)(0.0, 0.0); // Начальная скорость (нулевая)
    
    int closest_magnet = -1;
    
    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        // Проверяем сходимость к одному из магнитов
        double current_speed = length(vel);
        for (int j = 0; j < 3; ++j) {
            double dist_to_magnet = distance(pos, magnets[j]);
            if (dist_to_magnet < CONVERGENCE_THRESHOLD && current_speed < MIN_VELOCITY) {
                closest_magnet = j;
                break;
            }
        }
        
        if (closest_magnet != -1) break; // Маятник сошёлся к магниту
        
        // Вычисляем силы
        double2 total_force = (double2)(0.0, 0.0);
        
        // Гравитационная восстанавливающая сила (малые углы)
        double2 gravity_force = -(GRAVITY / L) * pos;
        total_force += gravity_force;
        
        // Магнитные силы от каждого магнита
        for (int j = 0; j < 3; ++j) {
            double2 dir = magnets[j] - pos;
            double dist = length(dir);
            if (dist > 1e-6) { // Избегаем деления на ноль
                // Магнитная сила: комбинация квадратичной и кубической зависимости
                // для более реалистичного поведения на разных расстояниях
                double dist_sq = dist * dist;
                double force_magnitude = MAGNETIC_FORCE / (1.0 + dist_sq);
                total_force += force_magnitude * normalize(dir);
            }
        }
        
        // Сила трения пропорциональна скорости
        double2 friction_force = -FRICTION * vel;
        total_force += friction_force;
        
        // Интегрирование методом Эйлера
        vel += total_force * DT;
        pos += vel * DT;
        
        // Проверяем, не улетел ли маятник слишком далеко
        if (length(pos) > 5.0) {
            break; // Прерываем расчёт для избежания бесконечных колебаний
        }
    }

    // Определяем ближайший магнит, если не сошлись
    if (closest_magnet == -1) {
        double min_dist = INFINITY;
        for (int j = 0; j < 3; ++j) {
            double dist = distance(pos, magnets[j]);
            if (dist < min_dist) {
                min_dist = dist;
                closest_magnet = j;
            }
        }
    }

    // Устанавливаем цвет в зависимости от ближайшего магнита
    out_R[gid] = (closest_magnet == 0) ? 1.0f : 0.0f; // Красный
    out_G[gid] = (closest_magnet == 1) ? 1.0f : 0.0f; // Зелёный  
    out_B[gid] = (closest_magnet == 2) ? 1.0f : 0.0f; // Синий
}