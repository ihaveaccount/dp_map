// pendulum_kernel.c
// PARAM: L1 double 1.0
// PARAM: L2 double 1.0
// PARAM: M1 double 1.0
// PARAM: M2 double 1.0
// PARAM: G double 9.81
// PARAM: DT double 0.2
// PARAM: MAX_ITERATIONS int 5000
// VIEW_DEFAULT: x_min -3.141592653589793
// VIEW_DEFAULT: x_max 3.141592653589793
// VIEW_DEFAULT: y_min -3.141592653589793
// VIEW_DEFAULT: y_max 3.141592653589793

// В OpenCL M_PI_F не определен, используйте M_PI или определите свой PI для double
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__kernel void simulate(
    __global const double* initial_xs,
    __global const double* initial_ys,
    __global int* cycle_counts,
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

    double th1 = initial_xs[gid];
    double th2 = initial_ys[gid];
    double w1 = 0.0; // Используйте 0.0 для double
    double w2 = 0.0; // Используйте 0.0 для double

    int cycles = 0;

    // Предварительно вычисленные константы, где это возможно
    const double G_M1_plus_M2 = G * (M1 + M2);
    const double two_M1_plus_M2 = 2.0 * M1 + M2; // Литералы double

    for (int i = 0; i < MAX_ITERATIONS; ++i)
    {
        // Общие вычисления
        double sin_th1 = sin(th1);
        double cos_th1 = cos(th1);
        // double sin_th2 = sin(th2);
        // double cos_th2 = cos(th2); // Не используется напрямую, но может быть полезен для других формулировок
        double sin_th1_minus_th2 = sin(th1 - th2);
        double cos_th1_minus_th2 = cos(th1 - th2);
        double cos_2th1_minus_2th2 = cos(2.0 * (th1 - th2)); // Используйте 2.0

        double w1_sq = w1 * w1;
        double w2_sq = w2 * w2;

        // Общий знаменатель для alpha1 и alpha2 (скорректированный)
        double common_den_factor = two_M1_plus_M2 - M2 * cos_2th1_minus_2th2;

        double alpha1, alpha2;

        // Вычисление alpha1
        // den = L1 * (2.0f * M1 + M2 - M2 * cos(2.0f * th1 - 2.0f * th2));
        double den1 = L1 * common_den_factor;

        if (fabs(den1) < 1e-9) { // Используйте более подходящий порог для double
             alpha1 = 0.0;
        } else {
            double num1_1 = -G * two_M1_plus_M2 * sin_th1;
            // double num1_2 = -M2 * G * sin(th1 - 2.0 * th2); // sin(th1 - 2*th2) - это не sin_th1_minus_th2
                                                          // Если это было sin(th1-th2) * (-2.0*M2) * (w2*w2*L2 + w1*w1*L1*cos(th1-th2)), тогда:
                                                          // double num1_3_term = -2.0 * sin_th1_minus_th2 * M2;
                                                          // double num1_4_term = w2_sq * L2 + w1_sq * L1 * cos_th1_minus_th2;
                                                          // alpha1 = (num1_1 + num1_2 + num1_3_term * num1_4_term) / den1;

            // Перепроверка оригинальных уравнений:
            // num1_1 = -G * (2.0f * M1 + M2) * sin(th1);
            // num1_2 = -M2 * G * sin(th1 - 2.0f * th2);
            // num1_3 = -2.0f * sin(th1 - th2) * M2;
            // num1_4 = w2 * w2 * L2 + w1 * w1 * L1 * cos(th1 - th2);
            // alpha1 = (num1_1 + num1_2 + num1_3 * num1_4) / den;

            // Похоже, что `sin(th1 - 2.0f * th2)` было правильно, его нельзя упростить с `sin_th1_minus_th2`
            double sin_th1_minus_2th2 = sin(th1 - 2.0 * th2);
            double num1_3_and_4 = -2.0 * sin_th1_minus_th2 * M2 * (w2_sq * L2 + w1_sq * L1 * cos_th1_minus_th2);
            alpha1 = (num1_1 + (-M2 * G * sin_th1_minus_2th2) + num1_3_and_4) / den1;
        }

        // Вычисление alpha2
        // den2_equiv = L2 * (2.0f * M1 + M2 - M2 * cos(2.0f * th1 - 2.0f * th2));
        double den2 = L2 * common_den_factor;

        if (fabs(den2) < 1e-9) { // Используйте более подходящий порог для double
            alpha2 = 0.0;
        } else {
            // num2_1 = 2.0f * sin(th1 - th2);
            // num2_2 = w1 * w1 * L1 * (M1 + M2);
            // num2_3 = G * (M1 + M2) * cos(th1);
            // num2_4 = w2 * w2 * L2 * M2 * cos(th1 - th2);
            // alpha2 = (num2_1 * (num2_2 + num2_3 + num2_4)) / den2_equiv;

            double term_common_mult = 2.0 * sin_th1_minus_th2;
            double term_sum = w1_sq * L1 * (M1 + M2) + G_M1_plus_M2 * cos_th1 + w2_sq * L2 * M2 * cos_th1_minus_th2;
            alpha2 = (term_common_mult * term_sum) / den2;
        }

        // Интегрирование Эйлера
        w1 += alpha1 * DT;
        w2 += alpha2 * DT;
        th1 += w1 * DT;
        th2 += w2 * DT;

        cycles++;

        // Условие выхода можно сделать более строгим или специфичным, если необходимо
        // Например, если маятник "улетел"
        if (fabs(th1) > 2.0 * M_PI) { // Используйте M_PI для double и возможно больший предел для th2
            break;
        }
    }

    cycle_counts[gid] = cycles;
}