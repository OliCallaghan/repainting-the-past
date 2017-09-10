//
//  frame_helper.hpp
//  NNSlave
//
//  Created by Oli Callaghan on 30/08/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef frame_helper_hpp
#define frame_helper_hpp

#include <iostream>

struct frame {
    float* data;
    unsigned int width;
    unsigned int height;
    unsigned int channels;
};
struct size_frame {
    unsigned int width;
    unsigned int height;
    unsigned int channels;
};
struct kernel {
    float* data;
    unsigned int width;
    unsigned int height;
    unsigned int channels;
    unsigned int n;
};
struct xyz {
    float x;
    float y;
    float z;
};
struct lab {
    float L;
    float a;
    float b;
};

frame getLocalFrame(char* filename);
frame stripLChannel(frame f_in);
frame stripABChannels(frame f_in);

#endif /* frame_helper_hpp */
