//
//  conv_layer.hpp
//  NNSlave
//
//  Created by Oli Callaghan on 10/08/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef conv_layer_hpp
#define conv_layer_hpp

#include <stdio.h>
#include "frame_helper.hpp"
#include <OpenCL/opencl.h>

class conv_layer {
    // Input parameters
    int d_x; // Data width
    int d_y; // Data height
    
    // Kernel data
    cl_float* k_d; // Kernel data
    cl_float* k_delta_accum; // Kernel deltas accumulated
    cl_float* k_delta; // Kernel deltas
    
    // Kernel parameters
    int kernel_n; // Number of kernels
    int k_x; // Kernel width
    int k_y; // Kernel height
    int k_channels; // Kernel channels
    
    // Layer meta-data
    cl_float stride;
    cl_int accum_stride;
    cl_int dilation;
    cl_int effect_dilation;
    
    // Post layer
    bool batch_norm;
    bool cross_entropy;
    
    float* layer_out;
    
    float* delta_map;
    float* lossmap;
    
    float* errmap;
public:
    // Constructor
    conv_layer(int k_n, int k_x, int k_y, int k_channels);
    
    // Generate random kernel data
    void generate();
    
    // Allocate memory locations for network
    //size_frame allocate(size_frame input, bool final_layer);
    
    // Forward propagation
    frame forward(dispatch_queue_t* queue, frame frame_in);
    frame calcDelta(dispatch_queue_t* queue, frame output_frame, frame target_frame);
    frame calcPrevDelta(dispatch_queue_t* queue, frame prev_delta);
    void backwards(dispatch_queue_t* queue, frame err_delta, frame lay_input);
};

#endif /* conv_layer_hpp */
