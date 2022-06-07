#pragma once

#include <fstream>
#include <iostream>
#include "settings.h"
#include <vector>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#include <CL/cl.hpp>
#include <CL/cl_platform.h>
#endif


cl::Program CreateProgram(const std::string& file);

cl::Buffer CreateMixedBuffer(cl::Context context, std::string functionName, int size);

cl::Buffer CreateInitBuffer(cl::Context context, std::vector<float>& data, std::string functionnameError);

std::vector<float> getFromBuffer(cl::CommandQueue queue, cl::Buffer buf, int size);