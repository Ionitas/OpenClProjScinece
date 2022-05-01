#pragma once

#include "settings.h"



namespace derivateFuncs {
	//паралельно считает произдодную
	std::vector<float> paralelfirstDerivate(std::vector<float>& vecDataSet);

	//паралельно считает 2 произдодную
	std::vector<float> paralelSecDerivate(std::vector<float>& vecDataSet);

	//Производная функций
	std::vector<float> paralel_first_derivateNonUniform(std::vector<float>& const f_z, std::vector<float>& const x_z);

	//Вторая производная 
	std::vector<float> paralel_second_derivateNonUniform(std::vector<float>& const f_zz, std::vector<float>& const f_x, std::vector<float>& const x_z, std::vector<float>& const x_zz);

}




namespace equation {
	bool checkStable1D(std::vector<float>& const vec, float tau, float a);

	std::vector<float> get_u_n_pararlel(std::vector <float>& u_n1, std::vector <float>& u_n2, float tau, std::vector<float>& f_res);

	std::vector<float> get_u_n(std::vector <float> u_n1, std::vector <float> u_n2, float tau, std::vector<float> f_res);

	//  Heat transfer equation  
	//  du / dt = a ^ 2 * d2u / dx ^ 2    
	//  u[0], u[-1] = const
	std::vector<float> heatEquation(std::vector<float> const& d2u, const float a);

	std::vector<float> heatEquationParalel(std::vector<float> d2u, const float a);

	std::vector<float> steaperHeatEquation( std::vector<float> xx, std::vector<float> uu, float const dt);

}