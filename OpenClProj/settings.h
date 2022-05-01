#pragma once
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>

namespace settings {
	int const VectorArraySize = 101;// 1 << 14; //16*1024
	float const dx = 1e-2;
	float const dt = 1e-2;
	float const a = 0.05;
	float const tau = 0.005;


	float InitFunctionForNonUniforms(float a);

	//Функция для тестирования правильности алгоритма
	float testFunc(float x);

	std::vector<float> InitGrid1D();

	std::vector<float> InitData1D(std::vector<float>& vecData);

	float gFunc(float const a, float t, float x, float const x0);


}