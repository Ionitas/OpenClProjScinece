#pragma once
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <numeric>
#include <chrono>

namespace settings {
	int const VectorArraySize = 6001;// 1 << 14; //16*1024 = 1<<14
	float const dx = 1e-2;
	float const dt = 1e-2;
	float const a = 0.05;
	float const tau = 0.005;
	float const tsteps = 500;
	float const t0 = 1;

	float InitFunctionForNonUniforms(float a);

	std::vector<float> timesVector(float const dt, float t0, float tsteps);

	//Функция для тестирования правильности алгоритма
	float testFunc(float t, float x);

	std::vector<float> InitGrid1D();

	std::vector<float> InitData1D(float t, std::vector<float>& grid);

	float gFunc(float const a, float t, float x, float const x0);

	float gEdgefunc(float  t, float x, float  x0);




}