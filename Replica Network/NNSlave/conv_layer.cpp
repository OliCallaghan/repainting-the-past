//
//  conv_layer.cpp
//  NNSlave
//
//  Created by Oli Callaghan on 10/08/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "conv_layer.hpp"
#include "math.h"
#include <OpenCL/opencl.h>

// 4D array interface macros
#define MAP_4D(m,n,c,d,width,height,channels) (m + n*width + c*width*height + d*width*height*channels)

// Include OpenCL header files (Xcode generated)
#include "convolve.cl.h"
#include "backpropagate.cl.h"

//
// conv_layer constructor
// parameters:
//  - k_x:        kernel width
//  - k_y:        kernel height
//  - k_channels: kernel channels
//  - k_n:        number of kernels

conv_layer::conv_layer(int k_n, int k_x, int k_y, int k_channels) {
    // Allocate memory for kernels
    this->k_x = k_x;
    this->k_y = k_y;
    this->k_channels = k_channels;
    this->kernel_n = k_n;
    
    this->k_d = (cl_float*)malloc(sizeof(cl_float) * this->k_x * this->k_y * this->k_channels * this->kernel_n);
    this->k_delta_accum = (cl_float*)malloc(sizeof(cl_float) * this->k_x * this->k_y * this->k_channels * this->kernel_n);
}

//
// conv_layer generate
// purpose: generates random weight values for each kernel
// parameters:
//   none;

void conv_layer::generate() {
    int k_n;
    int k_channels;
    int k_x;
    int k_y;
    for (k_n = 0; k_n < this->kernel_n; k_n++) {
        for (k_channels = 0; k_channels < this->k_channels; k_channels++) {
            for (k_x = 0; k_x < this->k_x; k_x++) {
                for (k_y = 0; k_y < this->k_y; k_y++) {
                    this->k_d[MAP_4D(k_x, k_y, k_channels, k_n, this->k_x, this->k_y, this->k_channels)] = ((float)rand() / RAND_MAX) * 2 - 1;
                    this->k_delta_accum[MAP_4D(k_x, k_y, k_channels, k_n, this->k_x, this->k_y, this->k_channels)] = 0;
                }
            }
        }
    }
}

frame conv_layer::forward(dispatch_queue_t* queue, frame frame_in) {
    frame out;
    
    // FRAME_MEM_SIZE IS ONLY OPTIMAL FOR IMAGE SIZE SAME SIZE AS INPUT NEED TO RECALCULATE
    int frame_in_mem_size = sizeof(cl_float) * frame_in.height * frame_in.width * frame_in.channels;
    int frame_out_mem_size = sizeof(cl_float) * (frame_in.width - this->k_x + 1) * (frame_in.height - this->k_y + 1) * this->kernel_n;
    
    // Initialise memory location for output frame
    float* frame_out_d = (float*)malloc(frame_out_mem_size);
    out.data = frame_out_d;
    out.channels = this->kernel_n;
    out.width = (frame_in.width - this->k_x + 1);
    out.height = (frame_in.height - this->k_y + 1);
    
    // Allocate memory on device for input, and move frame_in.data data to it
    void* mem_in = gcl_malloc(frame_in_mem_size, frame_in.data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* kernel_in = gcl_malloc(sizeof(cl_float) * this->k_x * this->k_y * this->k_channels * this->kernel_n, this->k_d, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    
    // Allocate memory on device for output
    void *mem_out = gcl_malloc(frame_out_mem_size, NULL, CL_MEM_WRITE_ONLY);
    
    dispatch_sync(*queue, ^{
        // Define work-group size for convole kernel
        size_t wgs = 1;
        gcl_get_kernel_block_workgroup_info(convolve_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, NULL);
        wgs = 1;
        cl_ndrange range_conv = {
            2,
            {0,0,0},
            {(frame_in.width - this->k_x) + 1,frame_in.height - this->k_y + 1,0},
            {wgs,wgs,0}
        };
        
        // Run convolve kernel
        convolve_kernel(&range_conv,
                        (cl_float*)mem_in,
                        (cl_int)frame_in.width,
                        (cl_int)frame_in.height,
                        (cl_int)frame_in.channels,
                        (cl_int)frame_in.width - this->k_x + 1,
                        (cl_int)frame_in.height - this->k_y + 1,
                        (cl_float*)kernel_in,
                        (cl_int)this->kernel_n,
                        (cl_int)this->k_x,
                        (cl_int)this->k_y,
                        (cl_int)this->k_channels,
                        (cl_float*)mem_out);
        
        
        // Move data from GPU to host program
        gcl_memcpy(frame_out_d, mem_out, frame_out_mem_size);
    });
    
    /*for (int loc = 0; loc < frame_out_mem_size / sizeof(cl_float); loc++) {
        // std::cout << frame_out_d[loc] << " ";
    }*/
    
    gcl_free(mem_in);
    gcl_free(mem_out);
    gcl_free(kernel_in);
    
    return out;
}

void conv_layer::backwards(dispatch_queue_t* queue, frame err_delta, frame lay_input) {
    
    
    int err_delta_size = sizeof(cl_float) * err_delta.width * err_delta.height * err_delta.channels;
    int lay_input_size = sizeof(cl_float) * lay_input.width * lay_input.height * lay_input.channels;
    int weight_delta_size = sizeof(cl_float) * this->kernel_n * this->k_x * this->k_y * this->k_channels;
    
    kernel weight_delta;
    weight_delta.data = (float*)malloc(weight_delta_size);
    weight_delta.height = this->k_y;
    weight_delta.width = this->k_x;
    weight_delta.channels = this->k_channels;
    weight_delta.n = this->kernel_n;
    
    void* err_delta_gpu = gcl_malloc(err_delta_size, err_delta.data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* lay_input_gpu = gcl_malloc(lay_input_size, lay_input.data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* weight_delta_gpu = gcl_malloc(weight_delta_size, NULL, CL_MEM_WRITE_ONLY);
    
    dispatch_sync(*queue, ^{
        // Define work-group size for convole kernel
        size_t wgs = 1;
        gcl_get_kernel_block_workgroup_info(convolve_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, NULL);
        wgs = 1;
        cl_ndrange range_weight_delta = {
            3,
            {0,0,0},
            {static_cast<size_t>(this->k_x),static_cast<size_t>(this->k_y),static_cast<size_t>(this->k_channels)},
            {wgs,wgs,wgs}
        };
        
        // Run convolve kernel
        calcWeightDeltas_kernel(&range_weight_delta,
                                (cl_float*)err_delta_gpu,
                                (cl_int)err_delta.width,
                                (cl_int)err_delta.height,
                                (cl_float*)lay_input_gpu,
                                (cl_int)lay_input.width,
                                (cl_int)lay_input.height,
                                (cl_float*)weight_delta_gpu,
                                (cl_int)this->kernel_n,
                                (cl_int)this->k_x,
                                (cl_int)this->k_y,
                                (cl_int)this->k_channels);
        
        // Move data from GPU to host program
        gcl_memcpy(weight_delta.data, weight_delta_gpu, weight_delta_size);
    });
    
    gcl_free(err_delta_gpu);
    gcl_free(lay_input_gpu);
    gcl_free(weight_delta_gpu);
    
    // Update Kernel Data
    for (int loc = 0; loc < weight_delta_size / sizeof(cl_float); loc++) {
        this->k_d[loc] += weight_delta.data[loc] * powf(10.0, -10.0);
    }
    
    free(weight_delta.data);
    
    /*std::cout << "ERR: " << err_delta.width << " x " << err_delta.height << " x " << err_delta.channels << "\n";
    std::cout << "LAY: " << lay_input.width << " x " << lay_input.height << " x " << lay_input.channels << "\n";
    std::cout << "WEI: " << weight_delta.width << " x " << weight_delta.height << " x " << weight_delta.channels << " x " << weight_delta.n << "\n";
    
    for (int loc = 0; loc < weight_delta_size / sizeof(cl_float); loc++) {
        //std::cout << weight_delta.data[loc] << " ";
    }*/
}

frame conv_layer::calcDelta(dispatch_queue_t* queue, frame output_frame, frame target_frame) {
    int cmmn_frame_size = sizeof(cl_float) * output_frame.height * output_frame.width * output_frame.channels;
    size_t dim = output_frame.height * output_frame.width * output_frame.channels;
    
    frame deltamap;
    deltamap.data = (float*)malloc(cmmn_frame_size);
    deltamap.width = output_frame.width;
    deltamap.height = output_frame.height;
    deltamap.channels = output_frame.channels;
    
    frame lossmap;
    lossmap.data = (float*)malloc(cmmn_frame_size);
    lossmap.width = output_frame.width;
    lossmap.height = output_frame.height;
    lossmap.channels = output_frame.channels;
    
    void* output_frame_gpu = gcl_malloc(cmmn_frame_size, output_frame.data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* target_frame_gpu = gcl_malloc(cmmn_frame_size, target_frame.data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    
    void* deltamap_gpu = gcl_malloc(cmmn_frame_size, NULL, CL_MEM_WRITE_ONLY);
    void* lossmap_gpu = gcl_malloc(cmmn_frame_size, NULL, CL_MEM_WRITE_ONLY);
    
    dispatch_sync(*queue, ^{
        // Define work-group size for convole kernel
        size_t wgs = 1;
        gcl_get_kernel_block_workgroup_info(convolve_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, NULL);
        wgs = 1;
        cl_ndrange range_delta_map = {
            1,
            {0,0,0},
            {dim,0,0},
            {wgs,0,0}
        };
        
        // Run convolve kernel
        calcFinalLayerDeltas_kernel(&range_delta_map,
                                    (cl_float*)output_frame_gpu,
                                    (cl_float*)target_frame_gpu,
                                    (cl_float*)deltamap_gpu);
        calcFinalLayerLoss_kernel(&range_delta_map,
                                  (cl_float*)output_frame_gpu,
                                  (cl_float*)target_frame_gpu,
                                  (cl_float*)lossmap_gpu);
        
        // Move data from GPU to host program
        gcl_memcpy(deltamap.data, deltamap_gpu, cmmn_frame_size);
        gcl_memcpy(lossmap.data, lossmap_gpu, cmmn_frame_size);
    });
    
    gcl_free(output_frame_gpu);
    gcl_free(target_frame_gpu);
    gcl_free(deltamap_gpu);
    gcl_free(lossmap_gpu);
    
    float total_err = 0;
    
    for (int loc = 0; loc < cmmn_frame_size / sizeof(cl_float); loc++) {
        total_err += lossmap.data[loc];
    }
    
    std::cout << "TOTAL ERROR: " << total_err << "\n";
    
    free(lossmap.data);
    
    return deltamap;
}

frame conv_layer::calcPrevDelta(dispatch_queue_t* queue, frame prev_delta) {
    int prev_delta_size = sizeof(cl_float) * prev_delta.height * prev_delta.width * prev_delta.channels;
    int kernel_size = sizeof(cl_float) * this->k_x * this->k_y * this->k_channels * this->kernel_n;
    int delta_size = sizeof(cl_float) * (prev_delta.height + this->k_y - 1) * (prev_delta.width + this->k_x - 1) * (this->k_channels);
    
    frame deltamap;
    deltamap.data = (float*)malloc(delta_size);
    deltamap.width = prev_delta.height + this->k_y - 1;
    deltamap.height = prev_delta.width + this->k_x - 1;
    deltamap.channels = this->k_channels;
    
    void* prev_delta_gpu = gcl_malloc(prev_delta_size, prev_delta.data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* k_d_gpu = gcl_malloc(kernel_size, this->k_d, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    
    void* delta_gpu = gcl_malloc(delta_size, NULL, CL_MEM_WRITE_ONLY);
    
    dispatch_sync(*queue, ^{
        // Define work-group size for convole kernel
        size_t wgs = 1;
        gcl_get_kernel_block_workgroup_info(convolve_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, NULL);
        wgs = 1;
        cl_ndrange range_delta_map = {
            3,
            {0,0,0},
            {deltamap.width,deltamap.height,deltamap.channels},
            {wgs,wgs,wgs}
        };
        
        // Run convolve kernel
        calcPrevLayerDeltas_kernel(&range_delta_map,
                                   (cl_float*)prev_delta_gpu,
                                   (cl_int)prev_delta.width,
                                   (cl_int)prev_delta.height,
                                   (cl_float*)k_d_gpu,
                                   (cl_int)this->kernel_n,
                                   (cl_int)this->k_x,
                                   (cl_int)this->k_y,
                                   (cl_int)this->k_channels,
                                   (cl_float*)delta_gpu,
                                   (cl_int)deltamap.width,
                                   (cl_int)deltamap.height);
        
        // Move data from GPU to host program
        gcl_memcpy(deltamap.data, delta_gpu, delta_size);
    });
    
    gcl_free(prev_delta_gpu);
    gcl_free(k_d_gpu);
    gcl_free(delta_gpu);
    
    return deltamap;
}
