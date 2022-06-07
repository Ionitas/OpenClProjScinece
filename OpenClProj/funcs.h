#pragma once

#include "settings.h"
#include "helpers.h"
#include "openCLHelper.h"

namespace derivateFuncs {

	void derivateParalel(cl::Program& program, cl::Context &context, cl::CommandQueue& queue, cl::Buffer& vecDataSet, std::string errorName, std::string kernelName, cl::Buffer& res , int size);
	//паралельно считает произдодную
	void paralelfirstDerivate(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer& vecDataSet, cl::Buffer& res, int size);
	//паралельно считает 2 произдодную
	void paralelSecDerivate(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer&  vecDataSet, cl::Buffer& res, int size);
	//Производная функций
	void fxDerivateNonUNiform(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer&  f_z, cl::Buffer&  x_z, cl::Buffer& res, int size);
	//Вторая производная 
	void fxDer2NonUNiform(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer& f_zz, cl::Buffer&   f_x, cl::Buffer&  x_z, cl::Buffer& x_zz, cl::Buffer& res, int size);

}




namespace equation {
	bool checkStable1D(cl::Program& program, cl::Context& context, cl::Buffer& vec, float tau, float a, int size);

	void nextUN(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer& uu, cl::Buffer& dudt, float const dt, cl::Buffer& res,int size);
	
	void heatEquationParalel(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer& d2u, const float a, cl::Buffer& res, int size);

	cl::Buffer steaperHeatEquation(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cl::Buffer& xx, cl::Buffer& uu, float const dt, cl::Buffer& f_z, cl::Buffer& x_z, cl::Buffer& f_zz, cl::Buffer& x_zz, cl::Buffer& du, cl::Buffer& d2u, cl::Buffer& heats, int size);

}