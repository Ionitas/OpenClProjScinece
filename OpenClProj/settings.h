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
	//static int VectorArraySize = 100011;// 1 << 14; //16*1024 = 1<<14
	float const dx = 0.01f;// 1.0f / (VectorArraySize - 1);// 1e-2;
	//float const dt = 1 / VectorArraySize;
	float const a = 0.05f;
	float const tau = 0.005f;
	int const tsteps = 500;
	float const t0 = 1;

	float InitFunctionForNonUniforms(float a);

	//std::vector<float> timesVector(float const dt, float t0, float tsteps);

	float testFunc(float t, float x);
	std::vector<float> InitGrid1D(int size);
	std::vector<float> InitData1D(float t, std::vector<float>& grid);
	float gFunc(float const a, float t, float x, float const x0);
	float gEdgefunc(float  t, float x, float  x0);




}