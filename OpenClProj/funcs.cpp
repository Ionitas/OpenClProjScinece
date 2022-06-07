#pragma once
#include "funcs.h"
#include "openCLHelper.h"

void derivateFuncs::derivateParalel(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer& vecDataSet, std::string errorName, std::string kernelName, cl::Buffer& res , int size) {
	cl_int error_ret;
	// выствляем аргументы Kernel
	cl::Kernel kernel(program, kernelName.data());
	error_ret = kernel.setArg(0, vecDataSet);

	if (error_ret != CL_SUCCESS) {
		std::cout << errorName << " - Kernel 0 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(1, res);

	if (error_ret != CL_SUCCESS) {
		std::cout << errorName.data() << " - Kernel 1 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(2, settings::dx);
	error_ret = kernel.setArg(3,size);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size));
}

void derivateFuncs::paralelfirstDerivate(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer& vecDataSet, cl::Buffer& res, int size) {
	return derivateFuncs::derivateParalel(program,context,queue, vecDataSet, "paralelfirstDerivate" , "first_dirivate", res, size);
}

void derivateFuncs::paralelSecDerivate(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer& vecDataSet, cl::Buffer& res, int size) {
	return derivateFuncs::derivateParalel(program,context, queue, vecDataSet, "paralelSecDerivate", "second_derivate", res, size);
}
	
void derivateFuncs::fxDerivateNonUNiform(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer& f_z, cl::Buffer& x_z, cl::Buffer& res, int size) {
	cl_int error_ret;
	cl::Kernel kernel(program, "firstNonUnoformderivate");
	error_ret = kernel.setArg(0, f_z);
	
	if (error_ret != CL_SUCCESS) {
		std::cout << " [firstNonUnoformderivate] Kernel 0 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(1, x_z);
	
	if (error_ret != CL_SUCCESS) {
		std::cout << " [firstNonUnoformderivate] Kernel 1 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(2, res);

	if (error_ret != CL_SUCCESS) {
		std::cout << " [firstNonUnoformderivate] Kernel 2 arg " << error_ret << std::endl;
	}

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size));

}

void derivateFuncs::fxDer2NonUNiform(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer& f_zz, cl::Buffer& f_x, cl::Buffer& x_z, cl::Buffer& x_zz, cl::Buffer& res, int size) {
	cl_int error_ret;
	cl::Kernel kernel(program, "secondNonUnoformderivate");
	error_ret = kernel.setArg(0, f_zz);
	if (error_ret != CL_SUCCESS) {
		std::cout << " [secondNonUnoformderivate] Kernel 0 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(1, f_x);
	if (error_ret != CL_SUCCESS) {
		std::cout << " [secondNonUnoformderivate] Kernel 1 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(2, x_z);
	if (error_ret != CL_SUCCESS) {
		std::cout << " [secondNonUnoformderivate] Kernel 2 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(3, x_zz);
	if (error_ret != CL_SUCCESS) {
		std::cout << " [secondNonUnoformderivate] Kernel 3 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(4, res);

	if (error_ret != CL_SUCCESS) {
		std::cout << " [secondNonUnoformderivate] Kernel 4 arg " << error_ret << std::endl;
	}
	
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size));

}

bool equation::checkStable1D(cl::Program& program, cl::Context& context, cl::Buffer &vec, float tau, float a, int size){
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto& device = devices.front();
	std::vector <float> resultParalel(size, 0.00);

	//Create Buffers 
	cl_int error_ret;

	
	cl::Buffer resulBuf = CreateMixedBuffer(context, "checkStable1D", size);


	// выствляем аргументы Kernel
	cl::Kernel kernel(program, "minDiff");
	error_ret = kernel.setArg(0, vec);

	if (error_ret != CL_SUCCESS) {
		std::cout << "Kernel 0 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(1, resulBuf);

	if (error_ret != CL_SUCCESS) {
		std::cout << "Kernel 1 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(2, int(size) - 1);


	// Выпоняем kernel функцию и получаем результат
	cl::CommandQueue queue(context, device);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size));
	error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0, size * sizeof(float), resultParalel.data());

	if (error_ret != CL_SUCCESS) {
		std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;

		exit(1);
	}


	float min = resultParalel[0];

	for (int i = 1; i < resultParalel.size() - 1; i++) {
		if (min > resultParalel[i]) min = resultParalel[i];
	}
	float coef = tau * pow((a / min), 2);
	std::cout << "Stability coef = " << coef << std::endl;

	if (coef < 0.5f)
		return true;
	else
		return false;
	}



void equation::heatEquationParalel(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer& d2u, const float a, cl::Buffer& res, int size) {
	cl_int error_ret;

	cl::Kernel kernel(program, "heat_calc");
	error_ret = kernel.setArg(0, d2u);

	if (error_ret != CL_SUCCESS) {
		std::cout << "Kernel 0 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(1, res);

	if (error_ret != CL_SUCCESS) {
		std::cout << "Kernel 1 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(2, a);

	if (error_ret != CL_SUCCESS) {
		std::cout << "[heat]Kernel 2 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(3, size);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size));
}

void equation::nextUN(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer& uu, cl::Buffer& dudt, float const dt, cl::Buffer& res, int size) {
	cl_int error_ret;
	cl::Kernel kernel(program, "nextun");
	error_ret = kernel.setArg(0, uu);

	if (error_ret != CL_SUCCESS) {
		std::cout << "[nextun]Kernel 0 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(1, dudt);

	if (error_ret != CL_SUCCESS) {
		std::cout << "[nextun] Kernel 1 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(2, res);

	if (error_ret != CL_SUCCESS) {
		std::cout << "[nextun]Kernel 2 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(3, dt);

	if (error_ret != CL_SUCCESS) {
		std::cout << "[nextun]Kernel 2 arg " << error_ret << std::endl;
	}

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size));
}
cl::Buffer equation::steaperHeatEquation(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer& xx, cl::Buffer& uu, float const dt, cl::Buffer& f_z, cl::Buffer& x_z, cl::Buffer& f_zz, cl::Buffer& x_zz, cl::Buffer& du, cl::Buffer& d2u, cl::Buffer& heats, int size) {
	
	
	derivateFuncs::paralelfirstDerivate(program, context, queue, uu, f_z, size);
	derivateFuncs::paralelfirstDerivate(program, context, queue, xx, x_z, size);
	derivateFuncs::paralelSecDerivate(program, context, queue, uu, f_zz , size);
	derivateFuncs::paralelSecDerivate(program, context, queue, xx, x_zz, size);

	
	derivateFuncs::fxDerivateNonUNiform(program, context, queue, f_z, x_z, du, size);
	derivateFuncs::fxDer2NonUNiform(program, context, queue, f_zz, du, x_z, x_zz, d2u, size);
	
	equation::heatEquationParalel(program, context, queue, d2u, settings::a, du, size);
	cl::Buffer res = CreateMixedBuffer(context, "Steper", size);
	equation::nextUN(program,context, queue, uu, du, dt, res, size);
	return res;
}




