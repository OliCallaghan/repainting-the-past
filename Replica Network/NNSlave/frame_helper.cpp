//
//  frame_helper.cpp
//  NNSlave
//
//  Created by Oli Callaghan on 30/08/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "frame_helper.hpp"
#include <OpenCL/opencl.h>
#include <math.h>

// Define XYZ reference white (CIE 1964: Daylight, sRGB)
#define X_r 94.811
#define Y_r 100
#define Z_r 107.304

// Convert RGB colour to XYZ
xyz RGB2XYZ(float r, float g, float b) {
    xyz colour;
    
    if (r > 0.04045) { r = powf((r + 0.055) / 1.055, 2.4); } else { r = r / 12.92; }
    if (g > 0.04045) { g = powf((g + 0.055) / 1.055, 2.4); } else { g = g / 12.92; }
    if (b > 0.04045) { b = powf((b + 0.055) / 1.055, 2.4); } else { b = b / 12.92; }
    
    r = r * 100;
    g = g * 100;
    b = b * 100;
    
    colour.x = (0.4124564*r + 0.3575761*g + 0.1804375*b);
    colour.y = (0.2126729*r + 0.7151522*g + 0.0721750*b);
    colour.z = (0.0193339*r + 0.1191920*g + 0.9503041*b);
    return colour;
}

// Convert XYZ colour to LAB
lab XYZ2LAB(float x, float y, float z) {
    lab colour;
    
    float x_r = x / X_r;
    float y_r = y / Y_r;
    float z_r = z / Z_r;
    
    float f_x;
    float f_y;
    float f_z;
    
    if (x_r > 0.008856) { f_x = powf(x_r, 1/3.); } else { f_x = (903.3 * x_r + 16) / 116; }
    if (y_r > 0.008856) { f_y = powf(y_r, 1/3.); } else { f_y = (903.3 * y_r + 16) / 116; }
    if (z_r > 0.008856) { f_z = powf(z_r, 1/3.); } else { f_z = (903.3 * z_r + 16) / 116; }
    
    colour.L = (116 * f_y) - 16;
    colour.a = 500 * (f_x - f_y);
    colour.b = 200 * (f_y - f_z);
    
    colour.L = colour.L / 100;
    colour.a = colour.a / 184.439;
    colour.b = colour.b / 202.345;
    
    return colour;
}

// Extract local frame data as LAB (must be BMP format stored as RGB)
frame getLocalFrame(char* filename) {
    // Initialise output struct and iterator
    frame output;
    int i;
    
    // Open file
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // Reads header
    
    // Extract image height and width from header
    output.width = *(int*)&info[18];
    output.height = *(int*)&info[22];
    output.channels = 3;
    
    int size = 3 * output.width * output.width;
    
    unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * size); // allocate 3 bytes per pixel
    fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
    fclose(f);
    
    // Switch positions of red and blue (stored as BGR instead of RGB)
    for(i = 0; i < size; i += 3) {
        unsigned char tmp = data[i];
        data[i] = data[i+2];
        data[i+2] = tmp;
    }
    
    // Initialise LAB data map
    float* lab_data = (float*)malloc(sizeof(cl_float) * size);
    
    for (i = 0; i < size; i += 3) {
        // Image data starts from bottom left corner and works right and up
        xyz pixel_xyz = RGB2XYZ((float)(data[i]) / 255, (float)(data[i+1]) / 255, (float)(data[i+2]) / 255);
        lab pixel_lab = XYZ2LAB(pixel_xyz.x, pixel_xyz.y, pixel_xyz.z);
        
        // Store LAB data
        lab_data[i] = pixel_lab.L;
        lab_data[i+1] = pixel_lab.a;
        lab_data[i+2] = pixel_lab.b;
    }
    
    free(data);
    
    // Add data location to output
    output.data = lab_data;
    
    return output;
}

// Strips out AB channels leaving only L channel from LAB data
frame stripLChannel(frame f_in) {
    frame output;
    int size = f_in.height * f_in.width;
    int i;
    float* l_channel = (float*)malloc(sizeof(float) * size);
    for (i = 0; i < size; i++) {
        l_channel[i] = f_in.data[i*3];
    }
    output.data = l_channel;
    output.height = f_in.height;
    output.width = f_in.width;
    output.channels = 1;
    return output;
}

// Strips out L channel leaving only AB channels from LAB data
frame stripABChannels(frame f_in) {
    frame output;
    int size = f_in.height * f_in.width;
    int i;
    float* ab_channels = (float*)malloc(sizeof(float) * size * 2);
    for (i = 0; i < size; i++) {
        ab_channels[2*i] = f_in.data[3*i + 1];
        ab_channels[2*i + 1] = f_in.data[3*i + 2];
    }
    output.data = ab_channels;
    output.height = f_in.height;
    output.width = f_in.width;
    output.channels = 2;
    return output;
}
