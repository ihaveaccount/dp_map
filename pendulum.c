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

#define M_PI_F 3.14159265358979323846f

__kernel void simulate(
    __global const double* initial_xs, // Renamed from initial_theta1s
    __global const double* initial_ys, // Renamed from initial_theta2s
    __global int* cycle_counts,          // Output array: number of cycles for each point
    const double L1,
    const double L2,
    const double M1,
    const double M2,
    const double G,
    const double DT,
    const int MAX_ITERATIONS, // Renamed from MAX_ITER
    const int width,
    const int height)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }
    int gid = row * width + col;

    double th1 = initial_xs[gid]; // Use initial_xs
    double th2 = initial_ys[gid]; // Use initial_ys
    double w1 = 0.0f; // Initial angular velocity of the first segment
    double w2 = 0.0f; // Initial angular velocity of the second segment

    int cycles = 0;

    for (int i = 0; i < MAX_ITERATIONS; ++i) 
    {
        // Equations of motion for a double pendulum (simplified for readability)
        // Source: https://www.myphysicslab.com/pendulum/double-pendulum-en.html
        // (with adjustment for M1, M2 as masses of the bobs, not rods)

        double delta_th = th1 - th2;

        // Accelerations alpha1, alpha2
        double num1_1 = -G * (2.0f * M1 + M2) * sin(th1);
        double num1_2 = -M2 * G * sin(th1 - 2.0f * th2);
        double num1_3 = -2.0f * sin(th1 - th2) * M2;
        double num1_4 = w2 * w2 * L2 + w1 * w1 * L1 * cos(th1 - th2);
        double den = L1 * (2.0f * M1 + M2 - M2 * cos(2.0f * th1 - 2.0f * th2));

        double alpha1, alpha2;

        if (fabs(den) < 1e-6f) { // Avoid division by zero
             alpha1 = 0.0f;
        } else {
            alpha1 = (num1_1 + num1_2 + num1_3 * num1_4) / den;
        }


        double num2_1 = 2.0f * sin(th1 - th2);
        double num2_2 = w1 * w1 * L1 * (M1 + M2);
        double num2_3 = G * (M1 + M2) * cos(th1);
        double num2_4 = w2 * w2 * L2 * M2 * cos(th1 - th2);
        // den is common, but for alpha2 it's divided by L2 * (...)
        double den2_equiv = L2 * (2.0f * M1 + M2 - M2 * cos(2.0f * th1 - 2.0f * th2));


        if (fabs(den2_equiv) < 1e-6f) { // Avoid division by zero
            alpha2 = 0.0f;
        } else {
             alpha2 = (num2_1 * (num2_2 + num2_3 + num2_4)) / den2_equiv;
        }

        // Euler integration
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