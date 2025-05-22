// lyapunov_kernel.c
// PARAM: A double 2.5
// PARAM: B double 3.5
// PARAM: INITIAL_X double 0.5
// PARAM: MAX_ITERATIONS int 5000
// PARAM: TRANSIENT_ITERATIONS int 500
// VIEW_DEFAULT: x_min 0.0
// VIEW_DEFAULT: x_max 10.0
// VIEW_DEFAULT: y_min 0.0
// VIEW_DEFAULT: y_max 10.0
// OUTPUT_CHANNELS: R, G, B

__kernel void simulate(
    __global const double* initial_xs, 
    __global const double* initial_ys, 
    __global float* out_R, // Output Red as float
    __global float* out_G, // Output Green as float
    __global float* out_B, // Output Blue as float
    const double A,
    const double B,
    const double INITIAL_X,
    const int MAX_ITERATIONS, 
    const int TRANSIENT_ITERATIONS,
    const int width,
    const int height)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }
    int gid = row * width + col;

    double alpha = initial_xs[gid]; 
    double beta = initial_ys[gid]; 

    double x = INITIAL_X;
    double sum_log_deriv = 0.0;

    char sequence[] = {'A', 'B'}; 
    int sequence_length = 2;

    for (int i = 0; i < TRANSIENT_ITERATIONS; ++i) {
        double r;
        if (sequence[i % sequence_length] == 'A') {
            r = alpha;
        } else {
            r = beta;
        }
        x = r * x * (1.0 - x);
        if (x <= 0.0 || x >= 1.0) {
            x = 0.5;
        }
    }

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        double r;
        if (sequence[i % sequence_length] == 'A') {
            r = alpha;
        } else {
            r = beta;
        }

        double clamped_x = fmax(1e-10, fmin(1.0 - 1e-10, x));
        
        double derivative = r * (1.0 - 2.0 * clamped_x);
        
        if (fabs(derivative) < 1e-10) {
            sum_log_deriv += log(1e-10);
        } else {
            sum_log_deriv += log(fabs(derivative));
        }
        
        x = r * x * (1.0 - x);
        if (x <= 0.0 || x >= 1.0) {
            x = 0.5;
        }
    }

    double lyapunov_exponent = sum_log_deriv / MAX_ITERATIONS;

    // Custom color mapping for Lyapunov exponent to RGB
    // A possible mapping:
    // Negative exponents (stability) -> shades of blue/green
    // Positive exponents (chaos) -> shades of red/yellow

    float r_val = 0.0f;
    float g_val = 0.0f;
    float b_val = 0.0f;

    // Normalize exponent to a working range, e.g., [-2.0, 0.7] as typical
    double min_exp = -2.0; 
    double max_exp = 0.7; 
    lyapunov_exponent = fmax(min_exp, fmin(max_exp, lyapunov_exponent));

    // Normalize to [0, 1] range
    float normalized_exp = (float)((lyapunov_exponent - min_exp) / (max_exp - min_exp));

    // Simple color gradient (example):
    // From blue (stable) to green (less stable) to yellow (chaotic) to red (very chaotic)
    if (normalized_exp < 0.25f) { // Blue to Cyan
        b_val = 1.0f;
        g_val = normalized_exp * 4.0f; 
    } else if (normalized_exp < 0.5f) { // Cyan to Green
        g_val = 1.0f;
        b_val = 1.0f - (normalized_exp - 0.25f) * 4.0f;
    } else if (normalized_exp < 0.75f) { // Green to Yellow
        g_val = 1.0f;
        r_val = (normalized_exp - 0.5f) * 4.0f;
    } else { // Yellow to Red
        r_val = 1.0f;
        g_val = 1.0f - (normalized_exp - 0.75f) * 4.0f;
    }

    // Clamp values to [0, 1] just in case
    out_R[gid] = fmax(0.0f, fmin(1.0f, r_val));
    out_G[gid] = fmax(0.0f, fmin(1.0f, g_val));
    out_B[gid] = fmax(0.0f, fmin(1.0f, b_val));
}