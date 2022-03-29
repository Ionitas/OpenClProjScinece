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


class ScinetificKernelFuncs{
public: 
	int count = 1;
	int initData = 0;

	ScinetificKernelFuncs(int count, int initData) {
		count = count;
		initData = initData;
	}

private:
	std::vector<int> vecName();
	
	std::vector<int> initWithData(int count, int initData);

};


