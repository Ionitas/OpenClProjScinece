#include "funcs.h"
#include "kernFuncs.h"
#include "derivate.h"


int main() {
	std::vector <float> vecDataSet(VectorArraySize, 0.00); 

	std::vector <float> resDataFirstDerivate(VectorArraySize, 0.00); 
    std::vector <float> resDataSecondDerivate(VectorArraySize, 0.00); 

	std::vector <float> resDataSetParralelFirstDerivate(VectorArraySize, 1.00);
	std::vector <float> resDataSetParralelSecondDerivate(VectorArraySize, 2.00);

    //Примеряем Функцию и заполняем данные в  vecDataSet
    for (int it = 0; it < VectorArraySize; ++it) {
		vecDataSet[it] = scientificFuncs::testFunc(it * dx);
		resDataFirstDerivate[it] = scientificFuncs::firstderivateTestFunc(it * dx);
        resDataSecondDerivate[it] = scientificFuncs::secondderivativeTestFunc(it * dx);
	
    }

	resDataSetParralelFirstDerivate = paralelDerivate<float>(vecDataSet);
	resDataSetParralelSecondDerivate = paralelDerivate<float>(resDataSetParralelFirstDerivate);

	for (int i = 0; i < VectorArraySize; ++i) {
		resDataFirstDerivate[i] = resDataFirstDerivate[i] - resDataSetParralelFirstDerivate[i];
		resDataSecondDerivate[i] = resDataSecondDerivate[i] - resDataSetParralelSecondDerivate[i];
		//std::cout << "First : " << resDataFirstDerivate[i] << "    Second : "<< resDataSecondDerivate[i] << std::endl;
    }

	helpFuncs::printFileData("firstDerive.txt", resDataFirstDerivate, " Differnce First Derivate ");
	helpFuncs::printFileData("secDerive.txt", resDataSecondDerivate, " Differnce Second Derivate ");

	return 0;
}
