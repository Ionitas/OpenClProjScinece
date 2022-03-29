#pragma once

#include <cmath>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <vector>
#include <iomanip>  

namespace scientificFuncs {
	//Функция для тестирования правильности алгоритма
	float testFunc(float x);

	//Производная функций
	float firstderivateTestFunc(float x);

	//Вторая производная 
	float secondderivativeTestFunc(float x);

}


namespace helpFuncs {
	//Функция берет и записывает в файл значение и описание
	void printFileData(std::string filename, std::vector<float> data, std::string description);



}