#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#include <CL/cl.hpp>
#include <CL/cl_platform.h>
#endif

cl::Device device;
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)


//Функция для тестирования правильности алгоритма
float testFunc(float x) {
	return sin(x);
}

//Производная функций
float firstderivateTestFunc(float x) {
	return cos(x);
}

//Вторая производная 
float secondderivativeTestFunc(float x) {
	return -sin(x);
}


int main() {

	char string[MEM_SIZE];


	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem memobj = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;







	int const VectorArraySize = 1000;   //1 << 14; //16*1024
	float const dx = 0.0001;






	// Reading kernels 
	FILE* fp;
	char fileName[] = "./myDerivateKernel.cl";
	char* source_str;
	size_t source_size;

	/* Load the source code containing the kernel*/
	fopen_s(&fp, fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);






	/* Get Platform and Device Info */
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);


	//Showing the platfomr we use 
	if (0 < ret_num_platforms)
	{
		cl_platform_id* platforms = new cl_platform_id[ret_num_platforms];
		clGetPlatformIDs(ret_num_platforms, platforms, NULL);
		platform_id = platforms[0];

		char platform_name[128];
		clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 128, platform_name, nullptr);
		std::cout << "Platform using = " << platform_name << std::endl;

		delete[] platforms;
	}


	//Show device using 
	char device_name[128];
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, 128, device_name, nullptr);
	std::cout << "Device using:  " << device_name << std::endl;
	

	/* Create OpenCL context */
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	if (ret != CL_SUCCESS) {
		std::cout << "Context error: " << ret << std::endl;
	}

	/* Create Command Queue */
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	if (ret != CL_SUCCESS) {
		std::cout << "Queue error " << ret << std::endl;
	}









	std::vector <float> vecDataSet(VectorArraySize, 0.00); //создаем вектор размерностью  VectorArraySize 

	std::vector <float> resDataFirstDerivate(VectorArraySize, 0.00); // забиваем все значениями 0
    std::vector <float> resDataSecondDerivate(VectorArraySize, 0.00); // забиваем все значениями 0
	
    std::vector <float> resDataSetParralel (VectorArraySize, 11.00); // забиваем все значениями 0
    
	float* resDataSetParralels = new float[VectorArraySize];



	std::vector <float> resDataSetParralelFirstDerivate(VectorArraySize, 1.00);
	std::vector <float> resDataSetParralelSecondDerivate(VectorArraySize, 2.00);


    //Примеряем Функцию и заполняем данные в  vecDataSet
    for (int it = 0; it < VectorArraySize; ++it) {
        vecDataSet[it] = testFunc(it * dx);
		resDataFirstDerivate[it] = firstderivateTestFunc(it * dx);
        resDataSecondDerivate[it] = secondderivativeTestFunc(it * dx);
	
    }





	//Create Buffers 
	cl_mem paralelData = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, VectorArraySize * sizeof(float), vecDataSet.data(), &ret);
	if (ret != CL_SUCCESS) {
		std::cout << "Create buffer failed vecDataSet: " << ret << std::endl;
	}

	cl_mem resultParalelFirstDerivateData_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, VectorArraySize * sizeof(float), NULL, &ret);
	if (ret != CL_SUCCESS) {
		std::cout << "Create buffer failed resDataSetParralelFirstDerivate : " << ret << std::endl;
	}

	//cl_mem resultParalelFirstSecondData = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, VectorArraySize * sizeof(float), resDataSetParralelSecondDerivate.data(), &ret);
	//if (ret != CL_SUCCESS) {
	//	std::cout << "Create buffer failed resDataSetParralelSecondDerivate: " << ret << std::endl;
	//}



//	cl_mem memobj = NULL;
	/* Create Memory Buffer */
//  memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(char), NULL, &ret);




	

	




	/* Create Kernel Program from the source */
	program = clCreateProgramWithSource(context, 1, (const char**)&source_str,
		(const size_t*)&source_size, &ret);




	/* Build Kernel Program */
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret != CL_SUCCESS) {
		std::cout << "Error building program: " << ret << std::endl;
		// Retrieve the latest compilation results
		char buffer[4096];
		size_t length;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		printf("compilation results = %s\n", buffer);
		exit(1);
	}





	/* Create OpenCL Kernel */
	 kernel = clCreateKernel(program, "calc_kernel", &ret);






	 ret = clEnqueueWriteBuffer(command_queue, paralelData, CL_TRUE, 0, sizeof(float) * VectorArraySize,
		 vecDataSet.data(),
		 0,
		 NULL,
		 NULL);

	 if (ret != CL_SUCCESS) {
		 std::cout << "Enqueue write buffer vecDataSet failed: " << ret << std::endl;
	 }

	 




	

	/* Set OpenCL Kernel Parameters */
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&paralelData);
	if (ret != CL_SUCCESS) {
		std::cout << "Set kernel args for b failed: " << ret << std::endl;
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&resultParalelFirstDerivateData_buffer);
	if (ret != CL_SUCCESS) {
		std::cout << "Set kernel args for resultParalelFirstDerivateData_buffer  failed: " << ret << std::endl;
	}
	ret = clSetKernelArg(kernel, 2, sizeof(float), (void*)&dx); if (ret != CL_SUCCESS) {
		std::cout << "Set kernel args for b failed: " << ret << std::endl;
	}







	/* Execute OpenCL Kernel */
	ret = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);






	/* Copy results from the memory buffer */
	ret = clEnqueueReadBuffer(command_queue, resultParalelFirstDerivateData_buffer, CL_TRUE, 0, sizeof(float) * VectorArraySize, resDataSetParralelFirstDerivate.data(), 0, NULL, NULL);

	if (ret != CL_SUCCESS) {
		std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << ret << std::endl;
		// Retrieve the latest compilation results
		char buffer[4096];
		size_t length;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		printf("compilation results = %s\n", buffer);
		exit(1);
	}














	    for (int i = 0; i < VectorArraySize; ++i) {
        std::cout << "  Paralel  1 Derivate  :  " << resDataSetParralelFirstDerivate[i] << " //   Real 1 Derivate :   " << resDataFirstDerivate[i] << std::endl;
        /*if (resDataSet[i] - resDataSetParralel[i] <= dx) arrayForTestingCPUvsGPU[i] = 0;
        else
        arrayForTestingCPUvsGPU[i] = resDataSetParralel[i] - resDataSet[i];
        std::cout << " Paralel : " << resDataSetParralel[i] << " Real : " << resDataSet[i] << " Diff : " << arrayForTestingCPUvsGPU[i] << std::endl;*/
    }

	
	
	
	
	
	
	
	
	
	
	
	
	/* Finalization */
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(memobj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	free(source_str);

	return 0;
}