//
//  main.cpp
//  NNSlave
//
//  Created by Oli Callaghan on 10/08/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include <iostream>
#include <OpenCL/opencl.h>
#include <ctime>
#include <math.h>

#include "conv_layer.hpp"
#include "frame_helper.hpp"

#define k_wx 12
#define k_wy 12

// Converts to 2D array
#define MAP2D(x,y,width) (x+y*width)

void getDispatchQueue(dispatch_queue_t* queue) {
    cl_uint num = 0;
    clGetDeviceIDs(NULL,CL_DEVICE_TYPE_GPU,0,NULL,&num);
    
    cl_device_id devices[num];
    clGetDeviceIDs(NULL,CL_DEVICE_TYPE_GPU,num,devices,NULL);
    
    // Attempt to use GPU
    *queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, devices[1]);
    // Fallback on CPU
    if (*queue == NULL) {
        *queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    }
}

// Feed forward method
// Purpose: Performs the learning loop across the whole network
// Parameters:
//  - layers: Pointer array to each layer
//  - l_n: Number of layers
void learn(dispatch_queue_t* queue, conv_layer** layers, frame frame_in, frame frame_comp, int l_n) {
    frame featuremaps[l_n + 1];
    frame err_deltamaps[l_n];
    featuremaps[0] = frame_in;
    for (int rpt = 0; rpt < 3600; rpt++) {
        for (int layer = 0; layer < l_n; layer++) {
            featuremaps[layer + 1] = layers[layer]->forward(queue, featuremaps[layer]);
        }
        
        err_deltamaps[l_n - 1] = layers[l_n]->calcDelta(queue, featuremaps[l_n], frame_comp);
        
        for (int layer = l_n - 1; layer >= 0; layer--) {
            layers[layer]->backwards(queue, err_deltamaps[layer], featuremaps[layer], powf(10.0, -10.0));
            if (layer == 0) { break; }
            err_deltamaps[layer - 1] = layers[layer]->calcPrevDelta(queue, err_deltamaps[layer]);
        }
        
        // CLOCK
        std::clock_t start;
        double duration;
        
        // START CLOCK
        start = std::clock();
        // -----------
        
        // STOP CLOCK
        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        std::cout<< "TIME: " << duration * 1000 << '\n';
        // ----------
        
        for (int i = 0; i < l_n; i++) {
            free(featuremaps[i + 1].data);
            free(err_deltamaps[i].data);
        }
    }
}

void generateRandomWeights(conv_layer** layers, int l_n) {
    for (int layer = 0; layer < l_n; layer++) {
        layers[layer]->generate();
    }
}

//void allocateMemory(conv_layer** layers, frame frame_in, int l_n) {
//    size_frame layer_dims[l_n + 1];
//    layer_dims[0].width = frame_in.width;
//    layer_dims[0].height = frame_in.height;
//    layer_dims[0].channels = frame_in.channels;
//    
//    for (int layer = 0; layer < l_n; layer++) {
//        layer_dims[layer+1] = layers[layer]->allocate(layer_dims[layer], true);
//    }
//}

int main(int argc, const char * argv[]) {
    // Define contants for network
    const int k_w = 12; // Kernel width
    const int k_h = 12; // Kernel height
    const int batch_size = 30;
    srand((unsigned)time(0));
    
    // Define frame dimensions
    int frame_width;
    int frame_height;
    
    // Initialise dispatch queue
    dispatch_queue_t queue;
    
    getDispatchQueue(&queue);
    
    // Prints device used in queue
    char name[128];
    cl_device_id gpu_id = gcl_get_device_id_with_dispatch_queue(queue);
    clGetDeviceInfo(gpu_id, CL_DEVICE_NAME, 128, name, NULL);
    std::cout << "Created dispatch queue using " << name << "\n";
    
    std::cout << "Initialising kernel data\n";
    /*conv_layer conv1_1(64,k_wx,k_wy,1,1,1,1,false,false);
    conv_layer conv1_2(64,k_wx,k_wy,2,1,1,1,true,false);
    
    conv_layer conv2_1(128,k_wx,k_wy,1,1,2,2,false,false);
    conv_layer conv2_2(128,k_wx,k_wy,2,1,2,2,true,false);
    
    conv_layer conv3_1(256,k_wx,k_wy,1,1,4,4,false,false);
    conv_layer conv3_2(256,k_wx,k_wy,1,1,4,4,false,false);
    conv_layer conv3_3(256,k_wx,k_wy,2,1,4,4,true,false);
    
    conv_layer conv4_1(512,k_wx,k_wy,1,1,8,8,false,false);
    conv_layer conv4_2(512,k_wx,k_wy,1,1,8,8,false,false);
    conv_layer conv4_3(512,k_wx,k_wy,1,1,8,8,true,false);
    
    conv_layer conv5_1(512,k_wx,k_wy,1,2,8,16,false,false);
    conv_layer conv5_2(512,k_wx,k_wy,1,2,8,16,false,false);
    conv_layer conv5_3(512,k_wx,k_wy,1,2,8,16,true,false);
    
    conv_layer conv6_1(512,k_wx,k_wy,1,2,8,16,false,false);
    conv_layer conv6_2(512,k_wx,k_wy,1,2,8,16,false,false);
    conv_layer conv6_3(512,k_wx,k_wy,1,2,8,16,true,false);
    
    conv_layer conv7_1(256,k_wx,k_wy,1,1,8,8,false,false);
    conv_layer conv7_2(256,k_wx,k_wy,1,1,8,8,false,false);
    conv_layer conv7_3(256,k_wx,k_wy,1,1,8,8,true,false);
    
    conv_layer conv8_1(128,k_wx,k_wy,0.5,1,4,4,false,false);
    conv_layer conv8_2(128,k_wx,k_wy,1,1,4,4,false,false);
    conv_layer conv8_3(128,k_wx,k_wy,1,1,4,4,false,true);*/
    std::cout << "Initialised kernel data\n";
    
    // FOR TESTING PURPOSES
    // Load test frame
    frame frame_in = getLocalFrame("/Users/Oli/Documents/CRGS/Computer Science Society/NNSlave/NNSlave/data.bmp");
    
    // Frame L channel and AB channels
    frame frame_in_L = stripLChannel(frame_in);
    
    frame frame_in_AB = stripABChannels(frame_in);
    
    // Cleans up combined LAB data (unnecessary)
    free(frame_in.data);
    
    
    // Constructor
    // conv_layer(int k_n, int k_x, int k_y, int k_channels);
    conv_layer* layers[3];
    conv_layer conv1_1(16,7,7,1);
    conv_layer conv1_2(4,7,7,16);
    conv_layer conv1_3(2,5,5,4);
    layers[0] = &conv1_1;
    layers[1] = &conv1_2;
    layers[2] = &conv1_3;
    
    //allocateMemory(layers, frame_in_L, 1);
    
    generateRandomWeights(layers, 3);
    
    learn(&queue, layers, frame_in_L, frame_in_AB, 3);
    
    dispatch_release(queue);

    return 0;
}
