#pragma once
#include "settings.h"

namespace helpFuncs {
	//Функция берет и записывает в файл значение и описание
	void printFileData(std::string filename, std::vector<float> data, std::string description);
	void printData(std::string filename, std::vector<float> data);
	void exportDataToWrapper(std::vector<std::vector<float>>& const vecdata);


}
