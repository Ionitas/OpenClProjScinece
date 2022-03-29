#pragma once

#include "openCLHelper.h"
#include <vector>
#include "config.h"


template <class T>
std::vector<T> paralelDerivate(std::vector<T> vecDataSet) {
	cl::Program program = CreateProgram("myDerivateKernel.cl");
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto& device = devices.front();

	std::vector <T> resultParalel(VectorArraySize, 0.00);


	//Create Buffers 
	cl_int error_ret;

	cl::Buffer dataBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(T) * vecDataSet.size(), vecDataSet.data(), &error_ret);
	if (error_ret != CL_SUCCESS) {
		std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
	}
	cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, VectorArraySize * sizeof(T), nullptr, &error_ret);
	if (error_ret != CL_SUCCESS) {
		std::cout << "Create buffer failed resulBuf: " << error_ret << std::endl;
	}


	// выствляем аргументы Kernel
	cl::Kernel kernel(program, "calc_kernel");
	error_ret = kernel.setArg(0, dataBuf);

	if (error_ret != CL_SUCCESS) {
		std::cout << "Kernel 0 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(1, resulBuf);

	if (error_ret != CL_SUCCESS) {
		std::cout << "Kernel 1 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(2, dx);
	error_ret = kernel.setArg(3, int(vecDataSet.size()));


	// Выпоняем kernel функцию и получаем результат
	cl::CommandQueue queue(context, device);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(VectorArraySize));
	error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0, VectorArraySize * sizeof(T), resultParalel.data());

	if (error_ret != CL_SUCCESS) {
		std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;

		exit(1);
	}

	return resultParalel;
}
