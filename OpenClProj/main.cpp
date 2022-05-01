
#include "funcs.h"


int main() {
	std::vector <float> gridData = settings::InitGrid1D();
	std::vector <float> funcDataonGrid = settings::InitData1D(gridData);

	std::cout <<"Is it stable ? " << (equation::checkStable1D(gridData, settings::tau, settings::a) ? "Yes" : "No") << std::endl;
	
	std::vector <float> f_z = derivateFuncs::paralelfirstDerivate(gridData);
	std::vector <float> f_zz = derivateFuncs::paralelSecDerivate(gridData);

	std::vector <float> x_z = derivateFuncs::paralelfirstDerivate(funcDataonGrid);
	std::vector <float> x_zz = derivateFuncs::paralelSecDerivate(funcDataonGrid);

	std::vector <float>  f_x = derivateFuncs::paralel_first_derivateNonUniform(f_z,x_z);
	std::vector <float> f_xx = derivateFuncs::paralel_second_derivateNonUniform(f_zz,f_x,x_z,x_zz);

	

	

	return 0;
}
 