#include "settings.h"


float settings::InitFunctionForNonUniforms(float x) {
	return  x;// sin(x) / sin(1);
}

float settings::testFunc(float t , float x) {
	return settings::gEdgefunc(t,x, 0.8) + settings::gEdgefunc(t,x,0.5)/2;
}

//2 * acos(0) = pi // note
float settings::gFunc(float  a, float t, float x, float  x0) {
	return 1.0f / (2 * a * pow(2 * acos(0) * t, 0.5)) * exp( - ((x-x0)*(x-x0))/(4*a*a*t)) ;
}

std::vector<float> settings::InitGrid1D(int size) {
	std::vector <float> vecDataSet(size, 0.00);

	for (int i = 0; i <size; i++) {
		vecDataSet[i] = settings::InitFunctionForNonUniforms(i * settings::dx);
	}
	return vecDataSet;
}

std::vector<float> settings::InitData1D(float t,std::vector<float>&grid) {
	int i = 0;
	std::vector <float> vecDataSet(grid.size(), 0.00);

	for (int i = 0; i < vecDataSet.size(); i++) {
		vecDataSet[i] = settings::testFunc(t, grid[i]);
	}
	return vecDataSet;
}

float settings::gEdgefunc(float  t, float x, float  x0) {
	return settings::gFunc(settings::a, t, x, x0) - settings::gFunc(settings::a, t, x, -x0) - settings::gFunc(settings::a, t, x, 2-x0);

}
