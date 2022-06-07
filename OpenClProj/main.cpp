#include "funcs.h"
#include "helpers.h"
int main() {
	cl::Program program = CreateProgram("myDerivateKernel.cl");
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto& device = devices.front();
	cl::CommandQueue queue(context, device);
	//for (int i = 0; i < devices.size(); i++)
		//std::cout << "ID: " << i << ", Device: " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
	//std::cout << devices.getInfo<CL_DEVICE_NAME>() << std::endl;
	
	//std::ofstream time;
	//time.open("times.txt");
	//std::ofstream sized;
	//sized.open("sizes.txt");


	int currentSize = 101;
	
	//while (currentSize < 1000000) {
	

		std::vector <float> gridData = settings::InitGrid1D(currentSize);
		std::vector <float> funcDataonGrid = settings::InitData1D(settings::t0, gridData);

		cl_int error_ret;
		cl::Buffer gridBuff = CreateInitBuffer(context, gridData, "MAIN FUNC INITBUF");
		cl::Buffer dataBuff = CreateInitBuffer(context, funcDataonGrid, "MAIN FUNC DATABUF");
		

		//std::cout << "Is it stable ? " << (equation::checkStable1D(program, context, gridBuff, settings::tau, settings::a) ? "Yes, < 1/2" : "No, >1/2") << std::endl;
		std::cout<<"SIZE: " << currentSize << std::endl;

		//double avaragetime = 0.0f;
		//for (int kk = 0; kk < 1; kk++) {

			cl::Buffer f_z = CreateMixedBuffer(context, "Func", currentSize);
			cl::Buffer x_z = CreateMixedBuffer(context, "Func" , currentSize);
			cl::Buffer f_zz = CreateMixedBuffer(context, "Func", currentSize);
			cl::Buffer x_zz = CreateMixedBuffer(context, "Func", currentSize);
			cl::Buffer du = CreateMixedBuffer(context, "Func" , currentSize);
			cl::Buffer d2u = CreateMixedBuffer(context, "Func", currentSize);
			cl::Buffer heats = CreateMixedBuffer(context, "Func", currentSize);
			
			std::ofstream myfile;
			myfile.open("m2y.txt");

			//auto start = std::chrono::system_clock::now();
			
			cl::Buffer step = dataBuff;
			std::vector<float> resultParalel = getFromBuffer(queue, step, currentSize);

			for (int it = 0; it < resultParalel.size(); ++it) {
				myfile << resultParalel[it] << ",";
			}
			for (int i = 0; i < settings::tsteps; i++) {
				step = equation::steaperHeatEquation(program, context, queue, gridBuff, step, settings::tau, 
					f_z, x_z, f_zz, x_zz, du, d2u, heats, currentSize);

				std::vector<float> resultParalel = getFromBuffer(queue, step, currentSize);

				for (int it = 0; it < resultParalel.size(); ++it) {
					myfile << resultParalel[it] << ",";
				}
				myfile << std::endl;
			}

			
			myfile.close();


			//auto end = std::chrono::system_clock::now();
			//std::chrono::duration<double> diff = end - start;
			//std::cout << "Time: " << std::setw(9) << diff.count() << std::endl;
			//avaragetime += diff.count();
		//}
		

		//time << 500*avaragetime / 4 << ",";
		//sized << currentSize << ",";
		//currentSize += 10000;
	//}
	//time.close();
	//sized.close();



	
	return 0;
}
 