#include "helpers.h"

//Функция берет и записывает в файл значение и описание
void helpFuncs::printFileData(std::string filename, std::vector<float> data, std::string description) {
	std::ofstream myfile;
	myfile.open(filename);
	for (int it = 0; it < data.size(); ++it) {
		myfile << description << std::setprecision(20) << std::fixed << data[it] << std::endl;
	}
	myfile << " End of file " << std::endl;
	myfile.close();
}


void helpFuncs::exportDataToWrapper(std::vector<std::vector<float>>& const vecdata) {

}