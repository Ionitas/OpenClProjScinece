#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#include <CL/cl.hpp>
#include <CL/cl_platform.h>
#endif


cl::Program CreateProgram(const std::string& file);
