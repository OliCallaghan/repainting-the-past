#define MAP_4D(m,n,c,d,width,height,channels) (m + n*width + c*width*height + d*width*height*channels)
#define MAP_3D(m,n,c,width,height) (m + n*width + c*width*height)

// Calculate deltas for mean squares error on last layer
// Parameters:
//  - d_in: final layer output
//  - d_channels: final layer output channel count
//  - t_in: target data
//  - d_delta: delta data
kernel void calcFinalLayerDeltas(global float* d_in, global float* t_in, global float* d_delta) {
    // Output pixel position
    size_t loc = get_global_id(0);
    
    d_delta[loc] = t_in[loc] - d_in[loc];
}

// Calculate loss for mean squares error on last layer
// Parameters:
//  - d_in: final layer output
//  - d_channels: final layer output channel count
//  - t_in: target data
//  - d_delta: delta data
kernel void calcFinalLayerLoss(global float* d_in, global float* t_in, global float* d_delta) {
    // Output pixel position
    size_t loc = get_global_id(0);
    
    d_delta[loc] = 0.5*pow(t_in[loc] - d_in[loc], 2);
}

kernel void calcWeightDeltas(global float* err_deltas,
                             int err_x,
                             int err_y,
                             global float* lay_input,
                             int lay_x,
                             int lay_y,
                             global float* weight_deltas,
                             int k_n,
                             int k_x,
                             int k_y,
                             int k_channels) {
    size_t loc_x = get_global_id(0);
    size_t loc_y = get_global_id(1);
    size_t loc_channel = get_global_id(2);
    
    int i;
    int j;
    int n;
    
    for (n = 0; n < k_n; n++) {
        float accum = 0;
        for (i = 0; i < lay_x - k_x; i++) {
            for (j = 0; j < lay_y - k_y; j++) {
                accum += err_deltas[MAP_3D(i,j,n,err_x,err_y)] * lay_input[MAP_3D(i+loc_x,j+loc_y,loc_channel,lay_x,lay_y)];
                //accum = err_deltas[MAP_3D(i,j,n,err_x,err_y)];
            }
        }
        weight_deltas[MAP_4D(loc_x,loc_y,loc_channel,n,k_x,k_y,k_channels)] = accum;
    }
}

kernel void calcPrevLayerDeltas(global float* err_deltas,
                                int err_x,
                                int err_y,
                                global float* k_d,
                                int k_n,
                                int k_x,
                                int k_y,
                                int k_channels,
                                global float* prev_err_deltas,
                                int prev_err_x,
                                int prev_err_y) {
    // Output pixel position
    size_t loc_x = get_global_id(0);
    size_t loc_y = get_global_id(1);
    size_t loc_channel = get_global_id(2);
    
    int m = 0;
    int p = 0;
    int q = 0;
    
    float accum = 0;
    
    while ((m >= 0) && ((m < k_y) && (m <= loc_y))) {
        while ((p>=0) && ((p < k_x) && (p <= loc_x))) {
            for (q = 0; q < k_n; q++) {
                accum += k_d[MAP_4D(p,m,loc_channel,q,k_x,k_y,k_channels)] * err_deltas[MAP_3D(loc_x-p,loc_y-m,q,err_x,err_y)];
            }
            p++;
        }
        m++;
    }
    
    prev_err_deltas[MAP_3D(loc_x,loc_y,loc_channel,prev_err_x,prev_err_y)] = accum;
    //prev_err_deltas[MAP_3D(loc_x,loc_y,loc_channel,prev_err_x,prev_err_y)] = 1;
}
