#pragma once
#include "funcs.h"
 

	//Функция для тестирования правильности алгоритма
	float scientificFuncs::testFunc(float x) {
		return sin(x);
	}

	//Производная функций
	float scientificFuncs::firstderivateTestFunc(float x) {
		return cos(x);
	}

	//Вторая производная 
	float scientificFuncs::secondderivativeTestFunc(float x) {
		return -sin(x);
	}

	
	//Функция берет и записывает в файл значение и описание
	void helpFuncs::printFileData(std::string filename, std::vector<float> data , std::string description) {
		std::ofstream myfile;
		myfile.open(filename);
		for (int it = 0; it < data.size(); ++it) {
			myfile << description << std::setprecision(5) << std::fixed << data[it] << std::endl;
		}
		myfile << " End of file " << std::endl;
		myfile.close();
	}