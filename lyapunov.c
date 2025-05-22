// lyapunov_kernel.c
// PARAM: A double 2.5
// PARAM: B double 3.5
// PARAM: INITIAL_X double 0.5
// PARAM: MAX_ITERATIONS int 5000
// PARAM: TRANSIENT_ITERATIONS int 500
// VIEW_DEFAULT: x_min -5.0
// VIEW_DEFAULT: x_max 5.0
// VIEW_DEFAULT: y_min -5.0
// VIEW_DEFAULT: y_max 5.0

__kernel void simulate(
    __global const double* initial_xs, 
    __global const double* initial_ys, 
    __global int* cycle_counts,          
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

    // The sequence 'A' for the Lyapunov fractal
    // For example, an A-B-A-B... sequence, you can vary this based on desired fractal
    char sequence[] = {'A', 'B'}; 
    int sequence_length = 2;

    // Transient iterations to allow the system to settle
    for (int i = 0; i < TRANSIENT_ITERATIONS; ++i) {
        double r;
        if (sequence[i % sequence_length] == 'A') {
            r = alpha;
        } else {
            r = beta;
        }
        x = r * x * (1.0 - x);
        // Ensure x stays within (0, 1) to avoid issues with logistic map
        if (x <= 0.0 || x >= 1.0) {
            x = 0.5; // Reset or handle divergence, or it can lead to NaNs
        }
    }

    // Main iterations to calculate Lyapunov exponent
    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        double r;
        if (sequence[i % sequence_length] == 'A') {
            r = alpha;
        } else {
            r = beta;
        }

        // Avoid log(0) issues, although for logistic map, x should stay between 0 and 1
        // If x is very close to 0 or 1, the derivative can be problematic.
        // Clamp x to a small epsilon to prevent log(0) or issues with derivative calculation
        double clamped_x = fmax(1e-10, fmin(1.0 - 1e-10, x));
        
        double derivative = r * (1.0 - 2.0 * clamped_x);
        
        // Handle cases where log(fabs(derivative)) might be -INFINITY (if derivative is 0)
        if (fabs(derivative) < 1e-10) { // If derivative is effectively zero
            sum_log_deriv += log(1e-10); // Add a very small number to avoid -INFINITY
        } else {
            sum_log_deriv += log(fabs(derivative));
        }
        
        x = r * x * (1.0 - x);
        // Ensure x stays within (0, 1) to avoid issues with logistic map
        if (x <= 0.0 || x >= 1.0) {
            x = 0.5; // Reset to a valid value to continue iteration if possible
        }
    }

    // Calculate Lyapunov exponent
    double lyapunov_exponent = sum_log_deriv / MAX_ITERATIONS;

    // Map the Lyapunov exponent to an integer for cycle_counts representing brightness.
    // Lyapunov exponents can be negative (stable) or positive (chaotic).
    // A typical range for logistic map Lyapunov exponents is roughly from -2 to 0.7 (max for r=4).
    // We need to map this range to an integer range, e.g., 0 to 255 for brightness.

    // Define a min and max expected Lyapunov exponent to normalize
    // These values might need to be adjusted based on the actual range observed in your fractal
    double min_lyapunov_exponent = -2.0; 
    double max_lyapunov_exponent = 0.7; 

    // Clamp the exponent to the defined range to avoid out-of-bounds mapping
    lyapunov_exponent = fmax(min_lyapunov_exponent, fmin(max_lyapunov_exponent, lyapunov_exponent));

    // Normalize to 0-1 range
    double normalized_exponent = (lyapunov_exponent - min_lyapunov_exponent) / (max_lyapunov_exponent - min_lyapunov_exponent);

    // Scale to 0-255 (or any desired integer range)
    // We invert the scale here so that higher exponents (more chaotic) are darker
    // and lower exponents (more stable) are brighter, which is common for Lyapunov fractals.
    int brightness = (int)(normalized_exponent * 255.0);
    //brightness = 255 - brightness; // Invert for desired visual effect (e.g., stable regions brighter)

    // Ensure the brightness value is within the valid range [0, 255]
    // brightness = fmax(0, fmin(255, brightness));

    cycle_counts[gid] = brightness;
}