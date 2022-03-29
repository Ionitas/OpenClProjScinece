#pragma once

#include "openCLHelper.h"
#include <vector>
#include "config.h"

struct WaveConditions {
	float a;
	float b;
	float c;

};

std::vector<float> waveEquationParalel(std::vector<float> uDer2, std::vector<float> uDer1, std::vector<float> u, WaveConditions border) {
	//u'' + a*u' + b*u -c = g ;
	// return g 


	cl::Program program = CreateProgram("myWaveEquation.cl");
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto& device = devices.front();

	std::vector<float> resultParalel(u.size(), 0.0);
	//Create Buffers 
	cl_int error_ret;

	cl::Buffer dataBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * u.size(), u.data(), &error_ret);
	if (error_ret != CL_SUCCESS) {
		std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
	}

	cl::Buffer dataDer1Buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * uDer1.size(), uDer1.data(), &error_ret);
	if (error_ret != CL_SUCCESS) {
		std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
	}

	cl::Buffer dataDer2Buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * uDer2.size(), uDer2.data(), &error_ret);
	if (error_ret != CL_SUCCESS) {
		std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
	}


	cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, resultParalel.size() * sizeof(float), nullptr, &error_ret);
	if (error_ret != CL_SUCCESS) {
		std::cout << "Create buffer failed resulBuf: " << error_ret << std::endl;
	}




	// выствляем аргументы Kernel
	cl::Kernel kernel(program, "wave_kernel");
	error_ret = kernel.setArg(0, dataDer2Buf);

	if (error_ret != CL_SUCCESS) {
		std::cout << "Kernel 0 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(1, dataDer1Buf);

	if (error_ret != CL_SUCCESS) {
		std::cout << "Kernel 1 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(2, dataBuf);

	if (error_ret != CL_SUCCESS) {
		std::cout << "Kernel 2 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(3, resulBuf);

	if (error_ret != CL_SUCCESS) {
		std::cout << "Kernel 3 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(4, border.a);
	error_ret = kernel.setArg(5, border.b);
	error_ret = kernel.setArg(6, border.c);

	// Выпоняем kernel функцию и получаем результат
	cl::CommandQueue queue(context, device);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(u.size()));
	error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0, resultParalel.size() * sizeof(float), resultParalel.data());

	if (error_ret != CL_SUCCESS) {
		std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;

		exit(1);
	}



	helpFuncs::printFileData("waveFuncParalel.txt", resultParalel, "");
	return resultParalel;
}




std::vector<float> waveEquationCPU(std::vector<float> uDer2, std::vector<float> uDer1, std::vector<float> u, WaveConditions border) {
	//u'' + a*u' + b*u -c = g ;
	// return g 

	std::vector<float> g(uDer2.size());

	for (int i = 0; i < uDer2.size(); i++) {
		g[i] = uDer2[i] + border.a * uDer1[i] + border.b * u[i] - border.c;
	}
	helpFuncs::printFileData("waveFuncCPU.txt", g, "");
	return g;
}


