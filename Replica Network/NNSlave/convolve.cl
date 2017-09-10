#define MAP_4D(m,n,c,d,width,height,channels) (m + n*width + c*width*height + d*width*height*channels)
#define MAP_3D(m,n,c,width,height) (m + n*width + c*width*height)

// Convolve Kernel & ReLu
// Parameters:
//  - d_in:       data in
//  - d_x:        data width
//  - d_y:        data height
//  - d_channels: channels of d_in
//  - d_o_x:      data out width
//  - d_o_y:      data out height
//  - k_d:        kernel data
//  - k_n:        number of kernels
//  - k_x:        kernel width
//  - k_y:        kernel height
//  - k_channels: kernel channels
//  - d_out:      data out

kernel void convolve(global float* d_in,
                     int d_x,
                     int d_y,
                     int d_channels,
                     int d_o_x,
                     int d_o_y,
                     global float* k_d,
                     int k_n,
                     int k_x,
                     int k_y,
                     int k_channels,
                     global float* d_out) {
    // Output pixel position
    size_t loc_x = get_global_id(0);
    size_t loc_y = get_global_id(1);
    // NOTE TO SELF YOU CAN HAVE 3D WORK GROUP 3RD DIMENSION FOR KERNEL N
    
    int m; // Kernel position X
    int n; // Kernel position Y
    int c; // Kernel channel number
    int d; // Kernel number
    
    for (d = 0; d < k_n; d++) {
        float accum = 0;
        for (c = 0; c < k_channels; c++) {
            for (n = 0; n < k_y; n++) {
                for (m = 0; m < k_x; m++) {
                    accum += d_in[MAP_3D(loc_x + m, loc_y + n,c,d_x,d_y)] * k_d[MAP_4D(m,n,c,d,k_x,k_y,k_channels)];
                    //accum = k_d[MAP_4D(m,n,c,d,k_x,k_y,k_channels)] * k_d[MAP_4D(m,n,c,d,k_x,k_y,k_channels)];
                }
            }
        }
        //d_out[MAP_3D(loc_x,loc_y,d,d_o_x,d_o_y)] = fmax((float)0,accum); // fmax() is ReLu layer
        d_out[MAP_3D(loc_x,loc_y,d,d_o_x,d_o_y)] = accum; // no ReLu layer
    }
}

// ReLu Layer
// Parameters:
//  - d_in:  data in
//  - d_out: data out
kernel void reLu(global float* d_in, global float* d_rect) {
    size_t x_loc = get_global_id(0);
    d_rect[x_loc] = fmax((float)0,d_in[x_loc]);
}
