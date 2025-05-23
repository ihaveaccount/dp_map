// magnetic_pendulum_kernel.c
// Simulates a magnetic pendulum with three fixed-point attractors (magnets).
// The pendulum bob's motion is influenced by:
// 1. Attractive forces from three magnets (inverse square law).
// 2. A restoring force due to gravity (small-angle approximation, Hooke's Law like).
// 3. A drag force proportional to velocity (air friction).
// The simulation determines which magnet the bob eventually settles near,
// starting from various initial positions with zero initial velocity.
// This creates pseudo-fractal basins of attraction.

// PARAM: R_MAGNETS double 1.0       // Radius of the equilateral triangle on which magnets are placed
// PARAM: MAGNET_STRENGTH double 10.0// Constant determining the strength of magnetic attraction
// PARAM: PENDULUM_K double 0.5      // Effective spring constant for gravity (mg/L), pulls bob to origin
// PARAM: DRAG_COEFFICIENT double 0.2// Coefficient for air drag (F_drag = -DRAG_COEFFICIENT * velocity)
// PARAM: MASS double 0.5            // Mass of the pendulum bob
// PARAM: DT double 0.001             // Time step for numerical integration
// PARAM: MAX_ITERATIONS int 10000    // Maximum number of simulation steps per initial condition
// PARAM: EPS double 0.05            // Convergence threshold: distance to a magnet to be considered "attracted"
// VIEW_DEFAULT: x_min -15          // Default view window minimum x-coordinate
// VIEW_DEFAULT: x_max 15           // Default view window maximum x-coordinate
// VIEW_DEFAULT: y_min -15          // Default view window minimum y-coordinate
// VIEW_DEFAULT: y_max 15           // Default view window maximum y-coordinate
// OUTPUT_CHANNELS: R, G, B          // Output color channels for visualization

__kernel void simulate(
    __global const double* initial_xs, // Array of initial X coordinates for the pendulum bob
    __global const double* initial_ys, // Array of initial Y coordinates for the pendulum bob
    __global float* out_R,             // Output buffer for the Red color channel
    __global float* out_G,             // Output buffer for the Green color channel
    __global float* out_B,             // Output buffer for the Blue color channel
    const double R_MAGNETS,            // Value from PARAM: R_MAGNETS
    const double MAGNET_STRENGTH,      // Value from PARAM: MAGNET_STRENGTH
    const double PENDULUM_K,           // Value from PARAM: PENDULUM_K
    const double DRAG_COEFFICIENT,     // Value from PARAM: DRAG_COEFFICIENT
    const double MASS,                 // Value from PARAM: MASS
    const double DT,                   // Value from PARAM: DT
    const int MAX_ITERATIONS,          // Value from PARAM: MAX_ITERATIONS
    const double EPS,                  // Value from PARAM: EPS
    const int width,                   // Width of the grid of initial conditions (pixels)
    const int height)                  // Height of the grid of initial conditions (pixels)
{
    // Get global thread IDs for 2D grid
    int row = get_global_id(0); // Corresponds to y-coordinate in the grid
    int col = get_global_id(1); // Corresponds to x-coordinate in the grid

    // Boundary check: ensure the thread is within the defined grid
    if (row >= height || col >= width) return;
    int gid = row * width + col; // Linear global ID for accessing 1D arrays

    // Define positions of the three magnets
    // They form an equilateral triangle centered at the origin (0,0)
    double2 magnets[3];
    double R_sin_60 = R_MAGNETS * 0.8660254037844386; // R_MAGNETS * sqrt(3)/2
    double R_cos_60_half = R_MAGNETS * 0.5;          // R_MAGNETS * 1/2

    magnets[0] = (double2)(R_MAGNETS, 0.0);               // Magnet 0 (typically colored Red)
    magnets[1] = (double2)(-R_cos_60_half, R_sin_60);     // Magnet 1 (typically colored Green)
    magnets[2] = (double2)(-R_cos_60_half, -R_sin_60);    // Magnet 2 (typically colored Blue)

    // Initialize pendulum bob's state for the current initial condition
    double2 pos = (double2)(initial_xs[gid], initial_ys[gid]); // Initial position
    double2 vel = (double2)(0.0, 0.0);                         // Initial velocity is zero

    int converged_magnet_index = -1; // Stores the index of the magnet the bob converges to

    // Main simulation loop
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        double2 total_force = (double2)(0.0, 0.0); // Accumulator for forces in this step
        
        // --- Calculate forces acting on the pendulum bob ---

        // 1. Magnetic attractive forces from each of the three magnets
        for (int j = 0; j < 3; ++j) {
            double2 vector_to_magnet = magnets[j] - pos;
            double dist_sq_to_magnet = dot(vector_to_magnet, vector_to_magnet); // distance^2

            // Check for convergence to this magnet
            if (dist_sq_to_magnet < EPS * EPS) {
                converged_magnet_index = j;
                break; // Bob has converged, exit magnet loop
            }
            
            // Safety check for division by zero if bob is exactly at magnet position
            // This should ideally be caught by the EPS check if EPS > 0.
            if (dist_sq_to_magnet == 0.0) { // This case implies dist_to_magnet would be 0
                 converged_magnet_index = j;
                 break;
            }

            double dist_to_magnet = sqrt(dist_sq_to_magnet);
            // Force magnitude is MAGNET_STRENGTH / dist_sq_to_magnet (inverse square law)
            // Force vector is F_magnitude * (vector_to_magnet / dist_to_magnet)
            // = (MAGNET_STRENGTH / (dist_sq_to_magnet * dist_to_magnet)) * vector_to_magnet
            // This formula gives F ~ 1/r^2, with vector direction included.
            total_force += (MAGNET_STRENGTH / (dist_sq_to_magnet * dist_to_magnet)) * vector_to_magnet;
        }
        
        if (converged_magnet_index != -1) {
            break; // Exit simulation loop if bob has converged
        }
        
        // 2. Gravitational restoring force (small-angle pendulum approximation)
        // F_gravity = -PENDULUM_K * pos (acts like a spring pulling bob towards origin (0,0))
        total_force -= PENDULUM_K * pos;
        
        // 3. Drag force (air friction)
        // F_drag = -DRAG_COEFFICIENT * vel (opposes motion)
        total_force -= DRAG_COEFFICIENT * vel;
        
        // --- Update bob's state using numerical integration (Semi-Implicit Euler method) ---
        // Acceleration a = F_total / MASS (from Newton's second law, F=ma)
        double2 acceleration = total_force / MASS;
        
        // Update velocity: v_new = v_old + acceleration * DT
        vel += acceleration * DT;
        
        // Update position: p_new = p_old + v_new * DT (using the *new* velocity for stability)
        pos += vel * DT;
    }

    // If the bob did not converge to any magnet within MAX_ITERATIONS,
    // determine the closest magnet at the end of the simulation.
    if (converged_magnet_index == -1) {
        double min_dist_sq = INFINITY;
        for (int j = 0; j < 3; ++j) {
            double2 diff_vector = pos - magnets[j];
            double current_dist_sq = dot(diff_vector, diff_vector);
            if (current_dist_sq < min_dist_sq) {
                min_dist_sq = current_dist_sq;
                converged_magnet_index = j;
            }
        }
    }

    // Assign an output color based on which magnet the bob ended up closest to.
    // (Assuming magnet 0 -> Red, magnet 1 -> Green, magnet 2 -> Blue)
    out_R[gid] = (converged_magnet_index == 0) ? 1.0f : 0.0f;
    out_G[gid] = (converged_magnet_index == 1) ? 1.0f : 0.0f;
    out_B[gid] = (converged_magnet_index == 2) ? 1.0f : 0.0f;
}