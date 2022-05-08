#pragma once
#include "funcs.h"
#include "openCLHelper.h"

cl::Buffer derivateFuncs::derivateParalel(cl::Program& program, cl::Context& context, cl::Buffer& vecDataSet, std::string errorName, std::string kernelName) {
	std::vector <float> resultParalel(settings::VectorArraySize, 0.00);
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto& device = devices.front();
	cl::CommandQueue queue(context, device);

	cl_int error_ret;

	cl::Buffer resulBuf = CreateMixedBuffer(context, errorName);

	// выствляем аргументы Kernel
	cl::Kernel kernel(program, kernelName.data());
	error_ret = kernel.setArg(0, vecDataSet);

	if (error_ret != CL_SUCCESS) {
		std::cout << errorName << " - Kernel 0 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(1, resulBuf);

	if (error_ret != CL_SUCCESS) {
		std::cout << errorName << " - Kernel 1 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(2, settings::dx);
	error_ret = kernel.setArg(3, settings::VectorArraySize);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(settings::VectorArraySize));
	return resulBuf;
}

cl::Buffer derivateFuncs::paralelfirstDerivate(cl::Program& program, cl::Context& context, cl::Buffer& vecDataSet) {
	return derivateParalel(program,context, vecDataSet, "paralelfirstDerivate" , "first_dirivate");
}

cl::Buffer derivateFuncs::paralelSecDerivate(cl::Program& program, cl::Context& context, cl::Buffer& vecDataSet) {
	return derivateParalel(program, context, vecDataSet, "paralelSecDerivate", "second_derivate");
}
	
cl::Buffer derivateFuncs::fxDerivateNonUNiform(cl::Program& program, cl::Context& context, cl::Buffer f_z, cl::Buffer x_z) {
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();
		cl::CommandQueue queue(context, device);

		std::vector <float> resultParalel(settings::VectorArraySize, 0.00);

		cl_int error_ret;

		cl::Buffer resulBuf = CreateMixedBuffer(context, "fxDerivateNonUNiform");


		cl::Kernel kernel(program, "firstNonUnoformderivate");
		error_ret = kernel.setArg(0, f_z);

		if (error_ret != CL_SUCCESS) {
			std::cout << " [firstNonUnoformderivate] Kernel 0 arg " << error_ret << std::endl;
		}
		error_ret = kernel.setArg(1, x_z);

		if (error_ret != CL_SUCCESS) {
			std::cout << " [firstNonUnoformderivate] Kernel 1 arg " << error_ret << std::endl;
		}

		error_ret = kernel.setArg(2, resulBuf);

		if (error_ret != CL_SUCCESS) {
			std::cout << " [firstNonUnoformderivate] Kernel 2 arg " << error_ret << std::endl;
		}


		
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(settings::VectorArraySize));


		return resulBuf;
	}

cl::Buffer derivateFuncs::fxDer2NonUNiform(cl::Program& program, cl::Context& context, cl::Buffer f_zz, cl::Buffer f_x, cl::Buffer const x_z, cl::Buffer x_zz) {
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();
		cl::CommandQueue queue(context, device);

		std::vector <float> resultParalel(settings::VectorArraySize, 0.00);

		cl_int error_ret;

		cl::Buffer resulBuf=  CreateMixedBuffer(context, "fxDer2NonUNiform");


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

		error_ret = kernel.setArg(4, resulBuf);

		if (error_ret != CL_SUCCESS) {
			std::cout << " [secondNonUnoformderivate] Kernel 4 arg " << error_ret << std::endl;
		}


		
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(settings::VectorArraySize));

		return resulBuf;
	}
		
	
bool equation::checkStable1D(cl::Program& program, cl::Context& context, cl::Buffer vec, float tau, float a){
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto& device = devices.front();

	std::vector <float> resultParalel(settings::VectorArraySize, 0.00);

	//Create Buffers 
	cl_int error_ret;

	
	cl::Buffer resulBuf = CreateMixedBuffer(context, "checkStable1D");


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
	error_ret = kernel.setArg(2, int(settings::VectorArraySize) - 1);


	// Выпоняем kernel функцию и получаем результат
	cl::CommandQueue queue(context, device);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(settings::VectorArraySize));
	error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0, settings::VectorArraySize * sizeof(float), resultParalel.data());

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

	
std::vector<float> equation::get_u_n(std::vector <float> u_n1, std::vector <float> u_n2, float tau, std::vector<float> f_res) {
	std::vector<float> res(u_n1.size());

	for (int i = 0; i < res.size(); i++) {
		res[i] = (4 * u_n1[i] - 3 * u_n2[i] + 2 * tau * f_res[i]);
	}

	return res;
}



	/*std::vector<float> equation::heatEquation(std::vector<float> const & d2u, const float a) {
		std::vector <float> dudt (d2u.size(),0);
		for (int i = 1; i < d2u.size()-1; i++) {
			dudt[i] = pow(a, 2) * d2u[i];
		} 
		return dudt;
	}*/

cl::Buffer equation::heatEquationParalel(cl::Program& program, cl::Context& context, cl::Buffer d2u, const float a) {
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto& device = devices.front();
	cl::CommandQueue queue(context, device);

	//Create Buffers 
	cl_int error_ret;
		
	cl::Buffer dudtBuf = CreateMixedBuffer(context, "heatEquationParalel");


	cl::Kernel kernel(program, "heat_calc");
	error_ret = kernel.setArg(0, d2u);

	if (error_ret != CL_SUCCESS) {
		std::cout << "Kernel 0 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(1, dudtBuf);

	if (error_ret != CL_SUCCESS) {
		std::cout << "Kernel 1 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(2, a);

	if (error_ret != CL_SUCCESS) {
		std::cout << "[heat]Kernel 2 arg " << error_ret << std::endl;
	}

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(settings::VectorArraySize));

	return dudtBuf;
}

cl::Buffer equation::nextUN(cl::Program& program, cl::Context& context, cl::Buffer& uu, cl::Buffer& dudt, float const dt) {
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto& device = devices.front();
	cl::CommandQueue queue(context, device);

	cl_int error_ret;
	cl::Buffer rezBuf = CreateMixedBuffer(context, "nextUN");

	cl::Kernel kernel(program, "nextun");
	error_ret = kernel.setArg(0, uu);

	if (error_ret != CL_SUCCESS) {
		std::cout << "[nextun]Kernel 0 arg " << error_ret << std::endl;
	}

	error_ret = kernel.setArg(1, dudt);

	if (error_ret != CL_SUCCESS) {
		std::cout << "[nextun] Kernel 1 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(2, rezBuf);

	if (error_ret != CL_SUCCESS) {
		std::cout << "[nextun]Kernel 2 arg " << error_ret << std::endl;
	}
	error_ret = kernel.setArg(3, dt);

	if (error_ret != CL_SUCCESS) {
		std::cout << "[nextun]Kernel 2 arg " << error_ret << std::endl;
	}

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(settings::VectorArraySize));

	return rezBuf;



}

cl::Buffer equation::steaperHeatEquation(cl::Program& program, cl::Context& context, cl::Buffer& xx, cl::Buffer& uu, float const dt) {

	cl::Buffer f_z = derivateFuncs::paralelfirstDerivate(program, context,uu);
	cl::Buffer x_z = derivateFuncs::paralelfirstDerivate(program, context,xx);
	cl::Buffer f_zz = derivateFuncs::paralelSecDerivate(program, context, uu);
	cl::Buffer x_zz = derivateFuncs::paralelSecDerivate(program, context, xx);

	cl::Buffer du = derivateFuncs::fxDerivateNonUNiform(program, context, f_z, x_z);
	cl::Buffer d2u = derivateFuncs::fxDer2NonUNiform(program, context, f_zz, du, x_z, x_zz);
	
	cl::Buffer dudt = equation::heatEquationParalel(program, context, d2u, settings::a);

	return equation::nextUN(program , context, uu, dudt, dt);
}




