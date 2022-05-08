#include "funcs.h"
#include "helpers.h"
int main() {
	cl::Program program = CreateProgram("myDerivateKernel.cl");
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto& device = devices.front();
	cl::CommandQueue queue(context, device);
	
	std::vector <float> gridData = settings::InitGrid1D();
	std::vector <float> funcDataonGrid = settings::InitData1D(settings::t0, gridData);

	auto start = std::chrono::system_clock::now();


	cl_int error_ret;
	cl::Buffer gridBuff = CreateInitBuffer(context, gridData, "MAIN FUNC");
	cl::Buffer dataBuff = CreateInitBuffer(context, funcDataonGrid, "MAIN FUNC");


	std::cout <<"Is it stable ? " << (equation::checkStable1D(program, context ,gridBuff, settings::tau, settings::a) ? "Yes, < 1/2" : "No, >1/2" )<< std::endl;


	cl::Buffer step = dataBuff;
	for (int i = 0; i < settings::tsteps; i++) {
		step = equation::steaperHeatEquation(program, context, gridBuff, step, settings::tau);
	}

	std::vector<float> resultParalel = getFromBuffer(queue,step);
	helpFuncs::printData("my.txt", resultParalel);

	for(int i = 0; i < resultParalel.size(); i++)
	{
		std::cout << resultParalel[i] << std::endl;
	}


	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << "Global time: " << std::setw(9) << diff.count() << std::endl;

	return 0;
}
 