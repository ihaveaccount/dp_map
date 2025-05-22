// mandelbrot_kernel.c
// PARAM: MAX_ITERATIONS int 200
// PARAM: BAILOUT_RADIUS_SQUARED double 4.0
// VIEW_DEFAULT: x_min -2.5
// VIEW_DEFAULT: x_max 1.5
// VIEW_DEFAULT: y_min -2.0
// VIEW_DEFAULT: y_max 2.0
// OUTPUT_CHANNELS: R, G, B

__kernel void simulate(
    __global const double* initial_xs, 
    __global const double* initial_ys, 
    __global float* out_R, // Output Red as float
    __global float* out_G, // Output Green as float
    __global float* out_B, // Output Blue as float
    const int MAX_ITERATIONS, 
    const double BAILOUT_RADIUS_SQUARED,
    const int width,
    const int height)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }
    int gid = row * width + col;

    double c_re = initial_xs[gid]; 
    double c_im = initial_ys[gid]; 

    double z_re = 0.0;
    double z_im = 0.0;

    int iterations = 0;
    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        double z_re_sq = z_re * z_re;
        double z_im_sq = z_im * z_im;

        if (z_re_sq + z_im_sq > BAILOUT_RADIUS_SQUARED) {
            break;
        }

        double next_z_re = z_re_sq - z_im_sq + c_re;
        double next_z_im = 2.0 * z_re * z_im + c_im;

        z_re = next_z_re;
        z_im = next_z_im;
        iterations++;
    }

    // // Custom color mapping for Mandelbrot set
    // float r_val = 0.0f;
    // float g_val = 0.0f;
    // float b_val = 0.0f;

    // if (iterations == MAX_ITERATIONS) {
    //     // Inside the Mandelbrot set (black)
    //     r_val = 0.0f;
    //     g_val = 0.0f;
    //     b_val = 0.0f;
    // } else {
    //     // Outside, color based on iteration count
    //     // A simple smooth coloring based on iteration count
    //     double smooth_color = (double)iterations + 1.0 - log(log(sqrt(z_re*z_re + z_im*z_im))) / log(2.0);
        
    //     // Example: simple gradient from blue to green to red
    //     // Adjust these values for desired color scheme
    //     float hue = fmodf((float)(smooth_color * 0.1f), 1.0f); // Adjust multiplier for color frequency

    //     // Simple hue to RGB conversion (e.g., using HSV to RGB logic)
    //     // This is a basic example; more sophisticated color mapping can be used.
    //     if (hue < 0.333f) { // Blue to Green
    //         b_val = 1.0f;
    //         g_val = hue * 3.0f;
    //         r_val = 0.0f;
    //     } else if (hue < 0.666f) { // Green to Red
    //         g_val = 1.0f;
    //         r_val = (hue - 0.333f) * 3.0f;
    //         b_val = 0.0f;
    //     } else { // Red to Blueish
    //         r_val = 1.0f;
    //         b_val = (hue - 0.666f) * 3.0f;
    //         g_val = 0.0f;
    //     }
    // }

    // Clamp values to [0, 1]
    out_R[gid] = iterations;//fmax(0.0f, fmin(1.0f, r_val));
    out_G[gid] = iterations;//fmax(0.0f, fmin(1.0f, g_val));
    out_B[gid] = iterations;//fmax(0.0f, fmin(1.0f, b_val));
}