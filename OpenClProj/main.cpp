#include "funcs.h"
#include "helpers.h"
int main() {
	auto start = std::chrono::system_clock::now();
	std::vector <float> gridData = settings::InitGrid1D();
	std::vector <float> funcDataonGrid = settings::InitData1D(settings::t0, gridData);

	std::cout <<"Is it stable ? " << (equation::checkStable1D(gridData, settings::tau, settings::a) ? "Yes, < 1/2" : "No, >1/2" )<< std::endl;
	//helpFuncs::printFileData("griddata.txt", funcDataonGrid," ");
	
	
	
	std::vector<float> steps =  funcDataonGrid;
	
	

	for (int i = 0; i < settings::tsteps; i++) {
		steps = equation::steaperHeatEquation(gridData, steps, settings::tau);
	}

	helpFuncs::printFileData("step.txt", steps, " ");

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << "Global time: " << std::setw(9) << diff.count() << std::endl;

	return 0;
}
 