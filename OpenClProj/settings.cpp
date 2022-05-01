#include "settings.h"


float settings::InitFunctionForNonUniforms(float a) {
	return sin(a) / sin(1);
}

float settings::testFunc(float x) {
	return x * x;//sin(x);
}

//2 * acos(0) = pi // note
float settings::gFunc(float const a, float t, float x, float const x0) {
	return 1 / (2 * a * pow(2 * acos(0) * t, 0.5) * exp( - ((x-x0)*(x-x0))/(4*a*a*t)) );
}

std::vector<float> settings::InitGrid1D() {
	std::vector <float> vecDataSet(settings::VectorArraySize, 0.00);

	for (int i = 0; i < vecDataSet.size(); i++) {
		vecDataSet[i] = settings::InitFunctionForNonUniforms(i * settings::dx);
	}
	return vecDataSet;
}

std::vector<float> settings::InitData1D(std::vector<float>&vecData) {
	int i = 0;
	std::vector <float> vecDataSet(settings::VectorArraySize, 0.00);

	for (int i = 0; i < vecDataSet.size(); i++) {
		vecDataSet[i] = settings::testFunc(vecData[i]);
	}
	return vecDataSet;
}

