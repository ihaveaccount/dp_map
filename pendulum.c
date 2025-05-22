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
// OUTPUT_CHANNELS: H, S, B

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__kernel void simulate(
    __global const double* initial_xs,
    __global const double* initial_ys,
    __global float* out_H, // Output Hue as float
    __global float* out_S, // Output Saturation as float
    __global float* out_B, // Output Brightness as float
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
    double w1 = 0.0;
    double w2 = 0.0;

    int cycles = 0;

    const double G_M1_plus_M2 = G * (M1 + M2);
    const double two_M1_plus_M2 = 2.0 * M1 + M2;

    for (int i = 0; i < MAX_ITERATIONS; ++i)
    {
        double sin_th1 = sin(th1);
        double cos_th1 = cos(th1);
        double sin_th1_minus_th2 = sin(th1 - th2);
        double cos_th1_minus_th2 = cos(th1 - th2);
        double cos_2th1_minus_2th2 = cos(2.0 * (th1 - th2));

        double w1_sq = w1 * w1;
        double w2_sq = w2 * w2;

        double common_den_factor = two_M1_plus_M2 - M2 * cos_2th1_minus_2th2;

        double alpha1, alpha2;

        double den1 = L1 * common_den_factor;

        if (fabs(den1) < 1e-9) {
             alpha1 = 0.0;
        } else {
            double num1_1 = -G * two_M1_plus_M2 * sin_th1;
            double sin_th1_minus_2th2 = sin(th1 - 2.0 * th2);
            double num1_3_and_4 = -2.0 * sin_th1_minus_th2 * M2 * (w2_sq * L2 + w1_sq * L1 * cos_th1_minus_th2);
            alpha1 = (num1_1 + (-M2 * G * sin_th1_minus_2th2) + num1_3_and_4) / den1;
        }

        double den2 = L2 * common_den_factor;

        if (fabs(den2) < 1e-9) {
            alpha2 = 0.0;
        } else {
            double term_common_mult = 2.0 * sin_th1_minus_th2;
            double term_sum = w1_sq * L1 * (M1 + M2) + G_M1_plus_M2 * cos_th1 + w2_sq * L2 * M2 * cos_th1_minus_th2;
            alpha2 = (term_common_mult * term_sum) / den2;
        }

        w1 += alpha1 * DT;
        w2 += alpha2 * DT;
        th1 += w1 * DT;
        th2 += w2 * DT;

        cycles++;

        if (fabs(th1) > 2.0 * M_PI) {
            break;
        }
    }

    // Calculate Hue based on th2 (angle of second segment)
    // Map th2 from -PI to PI to 0 to 1 (or 0 to 360 for actual hue)
    // Here we'll map to [0, 1] for easier use with HSB to RGB conversion later
    float hue = fmod((th2 + M_PI), (2.0 * M_PI)) / (2.0 * M_PI); // Normalize to [0, 1)
    if (hue < 0.0) hue += 1.0; // Ensure positive

    // Calculate Saturation based on w1 (angular velocity of first segment)
    // Normalize w1. Assuming w1 can range. You might need to adjust the range here.
    // A simple approach is to use a tanh function or clamp and normalize.
    // Let's assume a reasonable max velocity to normalize by.
    // For demonstration, let's normalize w1 from 0 to some max_w1_abs.
    // Example: max_w1_abs = 10.0 (adjust based on typical w1 values)
    // float max_w1_abs = 10.0f; // This value might need tuning
    float saturation = fabs(w1); // Clamp to [0, 1]

    // Brightness based on cycles (similar to previous, but as float)
    // Higher cycles (longer stability) can be brighter.
    // Let's normalize cycles by MAX_ITERATIONS.
    float brightness = (float)cycles;
    //brightness = fmin(1.0f, brightness); // Ensure it's not over 1.0

    out_H[gid] = 0; // Output normalized hue
    out_S[gid] = 0; // Output normalized saturation
    out_B[gid] = brightness; // Output normalized brightness
}