
#include "funcs.h"
#include "openCLHelper.h"

std::vector<float> get_u_n(std::vector <float> u_n1,std::vector <float> u_n2, float tau,std::vector<float> f_res) {
	std::vector<float> res(u_n1.size());

	for (int i = 0; i < res.size(); i++) {
		res[i] = (4 * u_n1[i] - 3 * u_n2[i] + 2 * tau * f_res[i]);
	}

	return res;
}

std::vector<float> get_u_n_pararlel(std::vector <float>& u_n1, std::vector <float>& u_n2, float tau, std::vector<float>& f_res) {


	cl::Program program = CreateProgram("getUN.cl");
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


	cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, VectorArraySize * sizeof(float), nullptr, &error_ret);
	if (error_ret != CL_SUCCESS) {
		std::cout << "Create buffer failed resulBuf: " << error_ret << std::endl;
	}


	// выствляем аргументы Kernel
	cl::Kernel kernel(program, "calc_kernel");
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
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(VectorArraySize));
	error_ret = queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0, VectorArraySize * sizeof(float), resultParalel.data());

	if (error_ret != CL_SUCCESS) {
		std::cout << "Error reading from buffer : resultParalelFirstDerivateData_buffer : " << error_ret << std::endl;

		exit(1);
	}
	

	return resultParalel;
}



float InitFunctionForNonUniforms(float a) {
	return sin(a);
}


std::vector<float> InitGrid() {
	std::vector <float> vecDataSet(VectorArraySize, 0.00);

	for (int i = 0; i < vecDataSet.size(); i++) {
		vecDataSet[i] = InitFunctionForNonUniforms(i * dx);
	}
	return vecDataSet;
}


std::vector<float> InitData(std::vector<float>& vecData) {
	int i = 0;
	std::vector <float> vecDataSet(VectorArraySize, 0.00);

	for (int i = 0; i < vecDataSet.size(); i++) {
		vecDataSet[i] = scientificFuncs::testFunc(vecData[i]);
	}
	return vecDataSet;
}




int main() {
	std::vector <float> gridData = InitGrid();
	std::vector <float> funcDataonGrid = InitData(gridData);

	std::vector <float> f_z = scientificFuncs::paralelfirstDerivate(gridData);
	std::vector <float> f_zz = scientificFuncs::paralelSecDerivate(gridData);

	std::vector <float> x_z = scientificFuncs::paralelfirstDerivate(funcDataonGrid);
	std::vector <float> x_zz = scientificFuncs::paralelSecDerivate(funcDataonGrid);

	std::vector <float>  f_x = scientificFuncs::paralel_first_derivateNonUniform(f_z,x_z);
	std::vector <float> f_xx = scientificFuncs::paralel_second_derivateNonUniform(f_zz,f_x,x_z,x_zz);


	for (auto it : f_xx) std::cout << it << std::endl;



	return 0;
}
