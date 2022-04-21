#pragma once
#include "funcs.h"
#include "openCLHelper.h"
	

	//Функция для тестирования правильности алгоритма
	float scientificFuncs::testFunc(float x) {
		return x * x;//sin(x);
	}

	//Производная функций
	float scientificFuncs::firstderivateTestFunc(float x) {
		return cos(x);
	}

	//Вторая производная 
	float scientificFuncs::secondderivativeTestFunc(float x) {
		return -sin(x);
	}

	
	//Функция берет и записывает в файл значение и описание
	void helpFuncs::printFileData(std::string filename, std::vector<float> data , std::string description) {
		std::ofstream myfile;
		myfile.open(filename);
		for (int it = 0; it < data.size(); ++it) {
			myfile << description << std::setprecision(5) << std::fixed << data[it] << std::endl;
		}
		myfile << " End of file " << std::endl;
		myfile.close();
	}


	//подсчет производной
	std::vector<float> scientificFuncs::paralelfirstDerivate(std::vector<float>& vecDataSet) {
		cl::Program program = CreateProgram("myDerivateKernel.cl");
		cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();

		std::vector <float> resultParalel(VectorArraySize, 0.00);


		//Create Buffers 
		cl_int error_ret;

		cl::Buffer dataBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * vecDataSet.size(), vecDataSet.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}
		cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, VectorArraySize * sizeof(float), nullptr, &error_ret);
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
		error_ret = kernel.setArg(2, dx);
		error_ret = kernel.setArg(3, int(vecDataSet.size()));


		// Выпоняем kernel функцию и получаем результат
		cl::CommandQueue queue(context, device);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(VectorArraySize));
		error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0, VectorArraySize * sizeof(float), resultParalel.data());

		if (error_ret != CL_SUCCESS) {
			std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;

			exit(1);
		}

		return resultParalel;
	}


	//подсчет 2 производной
	std::vector<float> scientificFuncs::paralelSecDerivate(std::vector<float>& vecDataSet) {
		cl::Program program = CreateProgram("myDerivateKernel.cl");
		cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();

		std::vector <float> resultParalel(VectorArraySize, 0.00);


		//Create Buffers 
		cl_int error_ret;

		cl::Buffer dataBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(float) * vecDataSet.size(), vecDataSet.data(), &error_ret);
		if (error_ret != CL_SUCCESS) {
			std::cout << "Create buffer failed vecDataSet: " << error_ret << std::endl;
		}
		cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, VectorArraySize * sizeof(float), nullptr, &error_ret);
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
		error_ret = kernel.setArg(2, dx);
		error_ret = kernel.setArg(3, int(vecDataSet.size()));


		// Выпоняем kernel функцию и получаем результат
		cl::CommandQueue queue(context, device);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(VectorArraySize));
		error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0, VectorArraySize * sizeof(float), resultParalel.data());

		if (error_ret != CL_SUCCESS) {
			std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;

			exit(1);
		}
		return resultParalel;
	}

	
	//Производная функций
	std::vector<float> scientificFuncs::paralel_first_derivateNonUniform(std::vector<float>& const f_z, std::vector<float>&const x_z) {
		
		cl::Program program = CreateProgram("myLinearKernel.cl");
		cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();

		std::vector <float> resultParalel(VectorArraySize, 0.00);


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

		cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, VectorArraySize * sizeof(float), nullptr, &error_ret);
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
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(VectorArraySize));
		error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0, VectorArraySize * sizeof(float), resultParalel.data());

		if (error_ret != CL_SUCCESS) {
			std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;
			exit(1);
		}

		return resultParalel;
	}






	//Вторая производная 
	std::vector<float> scientificFuncs::paralel_second_derivateNonUniform(std::vector<float>& const f_zz, std::vector<float>& const f_x, std::vector<float>& const x_z, std::vector<float>& const x_zz) {
		cl::Program program = CreateProgram("myLinearKernel.cl");
		cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		auto& device = devices.front();

		std::vector <float> resultParalel(VectorArraySize, 0.00);


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




		cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, VectorArraySize * sizeof(float), nullptr, &error_ret);
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
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(VectorArraySize));
		error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0, VectorArraySize * sizeof(float), resultParalel.data());

		if (error_ret != CL_SUCCESS) {
			std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;

			exit(1);
		}
		return resultParalel;
	}
	
	
	
	
	
	//подсчет невязки 
	std::vector<float> scientificFuncs::waveEquationCPU(std::vector<float>& uDer2, std::vector<float>& uDer1, std::vector<float>& u, WaveConditions border) {
		//u'' + a*u' + b*u -c = g ;
		// return g 

		std::vector<float> g(uDer2.size());

		for (int i = 0; i < uDer2.size(); i++) {
			g[i] = uDer2[i] + border.a * uDer1[i] + border.b * u[i] - border.c;
		}
		helpFuncs::printFileData("waveFuncCPU.txt", g, "");
		return g;
	}



	//подсчет невязки OpenCL
	std::vector<float> scientificFuncs::waveEquationParalel(std::vector<float>&uDer2, std::vector<float>& uDer1, std::vector<float>& u, WaveConditions border) {
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




