#include "settings.h"


float settings::InitFunctionForNonUniforms(float a) {
	return sin(a) / sin(1);
}


float settings::testFunc(float x) {
	return x * x;//sin(x);
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

