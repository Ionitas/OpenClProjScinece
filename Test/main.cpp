//#include <stdio.h>
//#include <stdlib.h>
//#include <iostream>
//#include <string>
//#include <fstream>
//#include <vector>
//#ifdef __APPLE__
//#include <OpenCL/cl.h>
//#else
//#include <CL/cl.h>
//#include <CL/cl.hpp>
//#include <CL/cl_platform.h>
//#endif
//
//
//
//// Глобальные переменные и константы которые мы используем
//cl::Program program;
//cl::Context context;
//cl::Device device;
//
//
////Функция для тестирования правильности алгоритма
//float testFunc(float x) {
//    return sin(x);
//}
//
////Производная функций
//float derivativeTestFunc(float x) {
//    return cos(x);
//}
//
////Вторая производная 
//float secondderivativeTestFunc(float x) {
//    return -sin(x);
//}
//
//int main() {
//    int const VectorArraySize = 1 << 14; //16*1024
//    float const dx = 0.0001;
//
//    //Инициализация и поисп устроиства для работы ( Подготовительная работа )
//
//    std::vector<cl::Platform> all_platforms; //Найдти все плптвормы поддерживаюшие OpenCl и использовать фронт то есть Nvidia
//    cl::Platform::get(&all_platforms);
//    if (all_platforms.size() == 0) {
//        std::cout << " No platforms found. Check OpenCL installation!\n";
//        exit(1);
//    }
//    cl::Platform platform = all_platforms[0]; // Это и есть Nvidia - all_platforms[0]
//    std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << platform.getInfo<CL_PLATFORM_VERSION>() << "\n";
//
//    std::vector<cl::Device> devices;
//    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
//
//    if (devices.empty()) {
//        std::cout << "No devices found!" << std::endl;
//    }
//    device = devices[0];
//    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl; // Используем видеокарту у меня 940M
//
//
//    // Прочитать OpenCl kernel как строку 
//    std::ifstream kernel_file("myDerivateKernel.cl");
//    std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//    // Подготовим програму котрорая будут работать на устройстве Nvidia 940M
//    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
//    context = cl::Context(device);
//    program = cl::Program(context, sources);
//
//    auto err = program.build();
//    if (err != CL_BUILD_SUCCESS) {
//        std::cout << "Error!\n Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
//            << "\nBuild Info :\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
//    }
//
//    //Если нет ошибок device и program из kernel глобально выбран  
//
//
//    std::vector <float> vecDataSet(VectorArraySize, 0.00); //создаем вектор размерностью  VectorArraySize 
//    std::vector <float> resDataSet(VectorArraySize, 0.00); // забиваем все значениями 0
//    //std::vector <float> resDataSetParralel (VectorArraySize, 11.00); // забиваем все значениями 0
//    float* resDataSetParralel = new float[VectorArraySize];
//
//    //Примеряем Функцию и заполняем данные в  vecDataSet
//    for (int it = 0; it < VectorArraySize; ++it) {
//        vecDataSet[it] = testFunc(it * dx);
//        resDataSet[it] = secondderivativeTestFunc(it * dx); //  )))
//    }
//
//
//
//
//    // Создаем буферы и выделяем память 
//    cl::Buffer dataBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, VectorArraySize * sizeof(float), vecDataSet.data());
//    cl::Buffer resulBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, VectorArraySize * sizeof(float));
//
//
//    // выствляем аргументы Kernel
//    cl::Kernel kernel(program, "calc_kernel");
//    kernel.setArg(0, dataBuf);
//    kernel.setArg(1, resulBuf);
//    kernel.setArg(2, dx);
//
//
//
//    // Выпоняем kernel функцию и получаем результат
//    cl::CommandQueue queue(context, device);
//    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(VectorArraySize));
//    queue.enqueueReadBuffer(resulBuf, CL_TRUE, 0, VectorArraySize * sizeof(float), resDataSetParralel);
//   
//
//    //Проверка результатов
//    for (int i = 0; i < VectorArraySize; ++i) {
//        std::cout << " Paralel : " << resDataSetParralel[i] << " Real : " << resDataSet[i] << std::endl;
//        /*if (resDataSet[i] - resDataSetParralel[i] <= dx) arrayForTestingCPUvsGPU[i] = 0;
//        else
//        arrayForTestingCPUvsGPU[i] = resDataSetParralel[i] - resDataSet[i];
//        std::cout << " Paralel : " << resDataSetParralel[i] << " Real : " << resDataSet[i] << " Diff : " << arrayForTestingCPUvsGPU[i] << std::endl;*/
//    }
//
//
//
//    return 0;
//}