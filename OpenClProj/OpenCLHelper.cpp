#include "openCLHelper.h"



cl::Program CreateProgram(const std::string& file) {
	std::vector<cl::Platform> all_platforms; //Найдти все плaтfормы поддерживаюшие OpenCl и использовать фронт то есть Nvidia
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	cl::Platform platform = all_platforms.front(); // Это и есть Nvidia - all_platforms[0]
	//std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << platform.getInfo<CL_PLATFORM_VERSION>() << "\n";

	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	if (devices.empty()) {
		std::cout << "No devices found!" << std::endl;
	}
	auto& device = devices.front();
	//std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl; // Используем видеокарту у меня 940M


	// Прочитать OpenCl kernel как строку 
	std::ifstream kernel_file(file);
	std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));



	// Подготовим програму котрорая будут работать на устройстве Nvidia 940M
	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
	cl::Context context(device);
	cl::Program program(context, sources);

	auto err = program.build("-cl-std=CL3.0");
	if (err != CL_BUILD_SUCCESS) {
		std::cout << "Error!\n Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
			<< "\nBuild Info :\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
	}
	//Если нет ошибок device и program из kernel глобально выбран  


	return program;
}


cl::Buffer CreateMixedBuffer( cl::Context context, std::string functionName, int size) {
	cl_int error_ret;

	cl::Buffer buf(context, CL_MEM_READ_WRITE, size * sizeof(float), nullptr, &error_ret);
	if (error_ret != CL_SUCCESS) {
		std::cout <<
			" [ " << functionName << "]" << "Create buffer failed mixbuf: " << error_ret << std::endl;
	}

	return buf;
}


cl::Buffer CreateInitBuffer(cl::Context context, std::vector<float>& data, std::string functionnameError) {
	cl_int error_ret;

	cl::Buffer buf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.size()  * sizeof(float), data.data(), &error_ret);
	if (error_ret != CL_SUCCESS) {
		std::cout <<
			" [ " << functionnameError << "]" << "Create buffer failed mixbuf: " << error_ret << std::endl;
	}

	return buf;

}

std::vector<float> getFromBuffer(cl::CommandQueue queue, cl::Buffer buf , int size) {
	std::vector <float> resultParalel(size, 0.00);
	queue.enqueueReadBuffer(buf, CL_TRUE, 0, size * sizeof(float), resultParalel.data());
	return resultParalel;
}