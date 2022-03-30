#pragma once

#include <cmath>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <vector>
#include <iomanip>  
#include "openCLHelper.h"
#include "config.h"


struct WaveConditions {
	float a;
	float b;
	float c;

};

namespace scientificFuncs {
	//Функция для тестирования правильности алгоритма
	float testFunc(float x);

	//Производная функций
	float firstderivateTestFunc(float x);

	//Вторая производная 
	float secondderivativeTestFunc(float x);

	//паралельно считает произдодную
	std::vector<float> paralelDerivate(std::vector<float> vecDataSet);

	//паралельно считает 2 произдодную
	std::vector<float> paralelSecDerivate(std::vector<float> vecDataSet);

	//подсчет невязки
	std::vector<float> waveEquationCPU(std::vector<float> uDer2, std::vector<float> uDer1, std::vector<float> u, WaveConditions border);


	//подсчет невязки OpenCL
	std::vector<float> waveEquationParalel(std::vector<float> uDer2, std::vector<float> uDer1, std::vector<float> u, WaveConditions border);

}


namespace helpFuncs {
	//Функция берет и записывает в файл значение и описание
	void printFileData(std::string filename, std::vector<float> data, std::string description);



}