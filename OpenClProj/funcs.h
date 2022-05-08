#pragma once

#include "settings.h"
#include "helpers.h"
#include "openCLHelper.h"

namespace derivateFuncs {

	cl::Buffer derivateParalel(cl::Program& program, cl::Context& context, cl::Buffer& vecDataSet , std::string errorName , std::string kernelName);

	//паралельно считает произдодную
	cl::Buffer paralelfirstDerivate(cl::Program& program, cl::Context& context, cl::Buffer& vecDataSet);

	//паралельно считает 2 произдодную
	cl::Buffer paralelSecDerivate(cl::Program& program, cl::Context& context, cl::Buffer& vecDataSet);


	//Производная функций
	cl::Buffer fxDerivateNonUNiform(cl::Program& program, cl::Context& context, cl::Buffer f_z, cl::Buffer x_z);

	//Вторая производная 
	cl::Buffer fxDer2NonUNiform(cl::Program& program, cl::Context& context, cl::Buffer f_zz, cl::Buffer f_x, cl::Buffer x_z, cl::Buffer x_zz);

}




namespace equation {
	bool checkStable1D(cl::Program& program, cl::Context& context, cl::Buffer vec, float tau, float a);


	std::vector<float> get_u_n(std::vector <float> u_n1, std::vector <float> u_n2, float tau, std::vector<float> f_res);

	//  Heat transfer equation  
	//  du / dt = a ^ 2 * d2u / dx ^ 2    
	//  u[0], u[-1] = const
	//std::vector<float> heatEquation(std::vector<float> const& d2u, const float a);


	cl::Buffer nextUN(cl::Program& program, cl::Context& context, cl::Buffer& uu, cl::Buffer& dudt, float const dt);
	cl::Buffer heatEquationParalel(cl::Program& program, cl::Context& context, cl::Buffer d2u, const float a);

	cl::Buffer steaperHeatEquation(cl::Program& program, cl::Context& context, cl::Buffer& xx, cl::Buffer& uu, float const dt);

}