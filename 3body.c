// attractors_kernel.c
// PARAM: R double 5.0 // Радиус треугольника
// PARAM: FORCE double 10.0 // Сила притяжения
// PARAM: DT double 0.01 // Шаг интегрирования
// PARAM: MAX_ITERATIONS int 5000 // Максимальное число итераций
// PARAM: EPS double 0.00001 // Пороговое расстояние для сходимости
// PARAM: DAMPING double 0.999 // Затухание скорости
// VIEW_DEFAULT: x_min -100.0
// VIEW_DEFAULT: x_max 100.0
// VIEW_DEFAULT: y_min -100.0
// VIEW_DEFAULT: y_max 100.0
// OUTPUT_CHANNELS: R, G, B

__kernel void simulate(
    __global const double* initial_xs, // Исходные X координаты
    __global const double* initial_ys, // Исходные Y координаты
    __global float* out_R, // Выходной красный канал
    __global float* out_G, // Выходной зелёный канал
    __global float* out_B, // Выходной синий канал
    const double R,
    const double FORCE,
    const double DT,
    const int MAX_ITERATIONS,
    const double EPS,
    const double DAMPING,
    const int width,
    const int height)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= height || col >= width) return;
    int gid = row * width + col;

    // Координаты трёх притягивающих точек (равносторонний треугольник)
    double2 attractors[3];
    attractors[0] = (double2)(R, 0.0);                     // Красная точка (вершина 1)
    attractors[1] = (double2)(-R*0.5, R*0.86602540378);    // Зелёная точка (вершина 2)
    attractors[2] = (double2)(-R*0.5, -R*0.86602540378);  // Синяя точка (вершина 3)

    // Начальное положение свободной точки
    double2 pos = (double2)(initial_xs[gid], initial_ys[gid]);
    double2 vel = (double2)(0.0, 0.0); // Начальная скорость

    int closest_attractor = -1;
    
    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        // Вычисляем силы от каждой точки
        double2 total_force = (double2)(0.0, 0.0);
        for (int j = 0; j < 3; ++j) {
            double2 dir = attractors[j] - pos;
            double dist = length(dir);
            if (dist < EPS) {
                closest_attractor = j;
                break;
            }
            total_force += FORCE * dir / dist; // Нормализованное направление
        }
        
        if (closest_attractor != -1) break; // Точка сошлась
        
        // Интегрирование (Эйлер)
        vel += total_force * DT;
        vel *= DAMPING; // Затухание скорости
        pos += vel * DT;
    }

    // Определяем ближайшую точку, если не определили ранее
    if (closest_attractor == -1) {
        double min_dist = INFINITY;
        for (int j = 0; j < 3; ++j) {
            double dist = distance(pos, attractors[j]);
            if (dist < min_dist) {
                min_dist = dist;
                closest_attractor = j;
            }
        }
    }

    // Устанавливаем цвет в зависимости от ближайшей точки
    out_R[gid] = (closest_attractor == 0) ? 1.0f : 0.0f;
    out_G[gid] = (closest_attractor == 1) ? 1.0f : 0.0f;
    out_B[gid] = (closest_attractor == 2) ? 1.0f : 0.0f;
}