#include "funcs.h"
#include "helpers.h"
int main() {
	cl::Program program = CreateProgram("myDerivateKernel.cl");
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto& device = devices.front();
	cl::CommandQueue queue(context, device);

	double avaragetime = 0.0f;
	for (int kk = 0; kk < 4; kk++) {
		std::vector <float> gridData = settings::InitGrid1D();
		std::vector <float> funcDataonGrid = settings::InitData1D(settings::t0, gridData);
		cl_int error_ret;
		cl::Buffer gridBuff = CreateInitBuffer(context, gridData, "MAIN FUNC INITBUF");
		cl::Buffer dataBuff = CreateInitBuffer(context, funcDataonGrid, "MAIN FUNC DATABUF");
		cl::Buffer resBuff = CreateMixedBuffer(context, "MAIN FUNC RES");


		std::cout << "Is it stable ? " << (equation::checkStable1D(program, context, gridBuff, settings::tau, settings::a) ? "Yes, < 1/2" : "No, >1/2") << std::endl;


		//std::ofstream myfile;
		//myfile.open("my.txt");
		
		auto start = std::chrono::system_clock::now();

		cl::Buffer step = dataBuff;
		for (int i = 0; i <settings::tsteps; i++) {
			step = equation::steaperHeatEquation(program,context, queue, gridBuff, step, settings::tau);
		} 

		/*std::vector<float> resultParalel = getFromBuffer(queue, resBuff);
		for (int it = 0; it < resultParalel.size(); ++it) {
			myfile << resultParalel[it] << ",";
		}
		
		myfile.close();*/


		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - start;
		std::cout << "Time: " << std::setw(9) << diff.count() << std::endl;
		avaragetime += diff.count();
	}
	
	std::cout << avaragetime/4;
	return 0;
}
 