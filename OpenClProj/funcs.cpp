#pragma once
#include "funcs.h"
#include "openCLHelper.h"
	
	
	std::vector<float> derivateFuncs::paralelfirstDerivate(std::vector<float>& vecDataSet) {
		cl::Program program = CreateProgram("myDerivateKernel.cl");
		cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();

		std::vector <float> resultParalel(settings::VectorArraySize, 0.00);


		//Create Buffers 
		cl_int error_ret;

		cl::Buffer dataBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * vecDataSet.size(), vecDataSet.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}
		cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,settings:: VectorArraySize * sizeof(float), nullptr, &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed resulBuf: " << error_ret << std::endl;
		}


		// выствляем аргументы Kernel
		cl::Kernel kernel(program, "first_dirivate");
		error_ret = kernel.setArg(0, dataBuf);

		if (error_ret != CL_SUCCESS) {
			std::cout << "Kernel 0 arg " << error_ret << std::endl;
		}

		error_ret = kernel.setArg(1, resulBuf);

		if (error_ret != CL_SUCCESS) {
			std::cout << "Kernel 1 arg " << error_ret << std::endl;
		}
		error_ret = kernel.setArg(2, settings::dx);
		error_ret = kernel.setArg(3, int(vecDataSet.size()));


		// Выпоняем kernel функцию и получаем результат
		cl::CommandQueue queue(context, device);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(settings::VectorArraySize));
		error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0,settings::VectorArraySize * sizeof(float), resultParalel.data());

		if (error_ret != CL_SUCCESS) {
			std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;

			exit(1);
		}

		return resultParalel;
	}
	

	std::vector<float> derivateFuncs::paralelSecDerivate(std::vector<float>& vecDataSet) {
		cl::Program program = CreateProgram("myDerivateKernel.cl");
		cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();

		std::vector <float> resultParalel(settings::VectorArraySize, 0.00);


		//Create Buffers 
		cl_int error_ret;

		cl::Buffer dataBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * vecDataSet.size(), vecDataSet.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}
		cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,settings::VectorArraySize * sizeof(float), nullptr, &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed resulBuf: " << error_ret << std::endl;
		}


		// выствляем аргументы Kernel
		cl::Kernel kernel(program, "second_derivate");
		error_ret = kernel.setArg(0, dataBuf);

		if (error_ret != CL_SUCCESS) {
			std::cout << "Kernel 0 arg " << error_ret << std::endl;
		}

		error_ret = kernel.setArg(1, resulBuf);

		if (error_ret != CL_SUCCESS) {
			std::cout << "Kernel 1 arg " << error_ret << std::endl;
		}
		error_ret = kernel.setArg(2, settings::dx);
		error_ret = kernel.setArg(3, int(vecDataSet.size()));


		// Выпоняем kernel функцию и получаем результат
		cl::CommandQueue queue(context, device);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(settings::VectorArraySize));
		error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0,settings::VectorArraySize * sizeof(float), resultParalel.data());

		if (error_ret != CL_SUCCESS) {
			std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;

			exit(1);
		}
		return resultParalel;
	}

	
	std::vector<float> derivateFuncs::paralel_first_derivateNonUniform(std::vector<float>& const f_z, std::vector<float>&const x_z) {
		
		cl::Program program = CreateProgram("myLinearKernel.cl");
		cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();

		std::vector <float> resultParalel(settings::VectorArraySize, 0.00);


		//Create Buffers 
		cl_int error_ret;

		cl::Buffer f_z_Buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * f_z.size(), f_z.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}

		cl::Buffer x_z_Buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * x_z.size(), x_z.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}

		cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, settings::VectorArraySize * sizeof(float), nullptr, &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed resulBuf: " << error_ret << std::endl;
		}


		// выствляем аргументы Kernel
		cl::Kernel kernel(program, "firstNonUnoformderivate");
		error_ret = kernel.setArg(0, f_z_Buf);

		if (error_ret != CL_SUCCESS) {
			std::cout << " [firstNonUnoformderivate] Kernel 0 arg " << error_ret << std::endl;
		}
		error_ret = kernel.setArg(1, x_z_Buf);

		if (error_ret != CL_SUCCESS) {
			std::cout << " [firstNonUnoformderivate] Kernel 1 arg " << error_ret << std::endl;
		}

		error_ret = kernel.setArg(2, resulBuf);

		if (error_ret != CL_SUCCESS) {
			std::cout << " [firstNonUnoformderivate] Kernel 2 arg " << error_ret << std::endl;
		}


		// Выпоняем kernel функцию и получаем результат
		cl::CommandQueue queue(context, device);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(settings::VectorArraySize));
		error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0, settings::VectorArraySize * sizeof(float), resultParalel.data());

		if (error_ret != CL_SUCCESS) {
			std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;
			exit(1);
		}

		return resultParalel;
	}


	std::vector<float> derivateFuncs::paralel_second_derivateNonUniform(std::vector<float>& const f_zz, std::vector<float>& const f_x, std::vector<float>& const x_z, std::vector<float>& const x_zz) {
		cl::Program program = CreateProgram("myLinearKernel.cl");
		cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();

		std::vector <float> resultParalel(settings::VectorArraySize, 0.00);


		//Create Buffers 
		cl_int error_ret;

		cl::Buffer f_zz_Buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * f_zz.size(), f_zz.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}

		cl::Buffer f_x_Buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * f_x.size(), f_x.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}

		cl::Buffer x_z_Buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * x_z.size(), x_z.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}

		cl::Buffer x_zz_Buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * x_zz.size(), x_zz.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}




		cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, settings::VectorArraySize * sizeof(float), nullptr, &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed resulBuf: " << error_ret << std::endl;
		}


		// выствляем аргументы Kernel
		cl::Kernel kernel(program, "secondNonUnoformderivate");


		error_ret = kernel.setArg(0, f_zz_Buf);
		if (error_ret != CL_SUCCESS) {
			std::cout << " [secondNonUnoformderivate] Kernel 0 arg " << error_ret << std::endl;
		}
		error_ret = kernel.setArg(1, f_x_Buf);
		if (error_ret != CL_SUCCESS) {
			std::cout << " [secondNonUnoformderivate] Kernel 1 arg " << error_ret << std::endl;
		}
		error_ret = kernel.setArg(2, x_z_Buf);
		if (error_ret != CL_SUCCESS) {
			std::cout << " [secondNonUnoformderivate] Kernel 2 arg " << error_ret << std::endl;
		}
		error_ret = kernel.setArg(3, x_zz_Buf);
		if (error_ret != CL_SUCCESS) {
			std::cout << " [secondNonUnoformderivate] Kernel 3 arg " << error_ret << std::endl;
		}

		error_ret = kernel.setArg(4, resulBuf);

		if (error_ret != CL_SUCCESS) {
			std::cout << " [secondNonUnoformderivate] Kernel 4 arg " << error_ret << std::endl;
		}


		// Выпоняем kernel функцию и получаем результат
		cl::CommandQueue queue(context, device);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(settings::VectorArraySize));
		error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0, settings::VectorArraySize * sizeof(float), resultParalel.data());

		if (error_ret != CL_SUCCESS) {
			std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;

			exit(1);
		}
		return resultParalel;
	}
		
	
	bool equation::checkStable1D(std::vector<float>& const vec, float tau, float a){

		cl::Program program = CreateProgram("myMinDiff.cl");
		cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();

		std::vector <float> resultParalel(settings::VectorArraySize, 0.00);


		//Create Buffers 
		cl_int error_ret;

		cl::Buffer dataBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * vec.size(), vec.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}
		cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, settings::VectorArraySize * sizeof(float), nullptr, &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed resulBuf: " << error_ret << std::endl;
		}


		// выствляем аргументы Kernel
		cl::Kernel kernel(program, "minDiff");
		error_ret = kernel.setArg(0, dataBuf);

		if (error_ret != CL_SUCCESS) {
			std::cout << "Kernel 0 arg " << error_ret << std::endl;
		}

		error_ret = kernel.setArg(1, resulBuf);

		if (error_ret != CL_SUCCESS) {
			std::cout << "Kernel 1 arg " << error_ret << std::endl;
		}
		error_ret = kernel.setArg(2, int(vec.size()) - 1);


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


	std::vector<float> equation::get_u_n_pararlel(std::vector <float>& u_n1, std::vector <float>& u_n2, float tau, std::vector<float>& f_res) {
		cl::Program program = CreateProgram("myEquations.cl");
		cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();

		std::vector <float> resultParalel(u_n1.size(), 0.00);


		//Create Buffers 
		cl_int error_ret;

		cl::Buffer un1(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * u_n1.size(), u_n1.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}
		cl::Buffer un2(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * u_n2.size(), u_n2.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}
		cl::Buffer fres(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * f_res.size(), f_res.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}

		cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, settings::VectorArraySize * sizeof(float), nullptr, &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed resulBuf: " << error_ret << std::endl;
		}


		// выствляем аргументы Kernel
		cl::Kernel kernel(program, "get_u_n");
		error_ret = kernel.setArg(0, un1);

		if (error_ret != CL_SUCCESS) {
			std::cout << "Kernel 0 arg " << error_ret << std::endl;
		}


		error_ret = kernel.setArg(1, un2);

		if (error_ret != CL_SUCCESS) {
			std::cout << "Kernel 0 arg " << error_ret << std::endl;
		}

		error_ret = kernel.setArg(2, fres);

		if (error_ret != CL_SUCCESS) {
			std::cout << "Kernel 0 arg " << error_ret << std::endl;
		}


		error_ret = kernel.setArg(3, resulBuf);

		if (error_ret != CL_SUCCESS) {
			std::cout << "Kernel 1 arg " << error_ret << std::endl;
		}
		error_ret = kernel.setArg(2, tau);


		// Выпоняем kernel функцию и получаем результат
		cl::CommandQueue queue(context, device);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(settings::VectorArraySize));
		error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0, settings::VectorArraySize * sizeof(float), resultParalel.data());

		if (error_ret != CL_SUCCESS) {
			std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;

			exit(1);
		}


		return resultParalel;
	}

	std::vector<float> equation::heatEquation(std::vector<float> const & d2u, const float a) {
		std::vector <float> dudt (d2u.size(),0);
		for (int i = 1; i < d2u.size()-1; i++) {
			dudt[i] = pow(a, 2) * d2u[i];
		} 
		return dudt;
	}

	std::vector<float> equation::heatEquationParalel(std::vector<float> d2u, const float a) {
		cl::Program program = CreateProgram("myEquations.cl");
		cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();

		std::vector <float> dudtVec(d2u.size(), 0.00);


		//Create Buffers 
		cl_int error_ret;

		cl::Buffer d2uBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * d2u.size(), d2u.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed d2u: " << error_ret << std::endl;
		}
		

		cl::Buffer dudtBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, dudtVec.size() * sizeof(float), nullptr, &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed dudt: " << error_ret << std::endl;
		}


		// выствляем аргументы Kernel
		cl::Kernel kernel(program, "heat_calc");
		error_ret = kernel.setArg(0, d2uBuf);

		if (error_ret != CL_SUCCESS) {
			std::cout << "Kernel 0 arg " << error_ret << std::endl;
		}


		error_ret = kernel.setArg(1, dudtBuf);

		if (error_ret != CL_SUCCESS) {
			std::cout << "Kernel 1 arg " << error_ret << std::endl;
		}

		error_ret = kernel.setArg(2, a);

		if (error_ret != CL_SUCCESS) {
			std::cout << "Kernel 2 arg " << error_ret << std::endl;
		}


		


		// Выпоняем kernel функцию и получаем результат
		cl::CommandQueue queue(context, device);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(settings::VectorArraySize));
		error_ret = queue.enqueueReadBuffer(dudtBuf, CL_TRUE, 0, dudtVec.size() * sizeof(float), dudtVec.data());

		if (error_ret != CL_SUCCESS) {
			std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;

			exit(1);
		}

		dudtVec[0] = 0;
		dudtVec[dudtVec.size() - 1] = 0;

		return dudtVec;
	}

	std::vector<float> equation::steaperHeatEquation( std::vector<float> d2u,std::vector<float> uu, float const dt) {
		std::vector<float> temp(uu.size());
		std::vector <float> heat = equation::heatEquation(d2u, settings::a);
		for (int i = 0; i < uu.size(); i++) {
			temp[i] = uu[i] + dt * heat[i];
		}
	}




