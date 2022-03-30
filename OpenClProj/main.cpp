#include "funcs.h"


int main() {
	std::vector <float> vecDataSet(VectorArraySize, 0.00);

	int i = 0;
	for (auto& it : vecDataSet) it= scientificFuncs::testFunc( (i++) * dx);

	std::vector <float> resDataSetParralelFirstDerivate(VectorArraySize, 1.00);
	std::vector <float> resDataSetParralelSecondDerivate(VectorArraySize, 2.00);
	std::vector <float> resDataSetParralelSecondDerivate1(VectorArraySize, 2.00);
	resDataSetParralelFirstDerivate = scientificFuncs::paralelDerivate(vecDataSet);
	resDataSetParralelSecondDerivate = scientificFuncs::paralelSecDerivate(resDataSetParralelFirstDerivate);


	WaveConditions borders = WaveConditions{2.0, 2.0, 2.0};
	std::vector <float> wave = scientificFuncs::waveEquationCPU(resDataSetParralelSecondDerivate, resDataSetParralelFirstDerivate, vecDataSet, borders);
	std::vector <float> waveParalel = scientificFuncs::waveEquationParalel(resDataSetParralelSecondDerivate, resDataSetParralelFirstDerivate, vecDataSet, borders);
	

	std::cout << " Finished " << std::endl;
	return 0;
}
