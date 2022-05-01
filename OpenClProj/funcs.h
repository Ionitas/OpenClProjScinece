#pragma once

#include "settings.h"



namespace derivateFuncs {
	//паралельно считает произдодную
	std::vector<float> paralelfirstDerivate(std::vector<float>& vecDataSet);

	//паралельно считает 2 произдодную
	std::vector<float> paralelSecDerivate(std::vector<float>& vecDataSet);


	std::vector<float> paralel_first_derivateNonUniform(std::vector<float>& const f_z, std::vector<float>& const x_z);


	std::vector<float> paralel_second_derivateNonUniform(std::vector<float>& const f_zz, std::vector<float>& const f_x, std::vector<float>& const x_z, std::vector<float>& const x_zz);

}


namespace helpFuncs {
	//Функция берет и записывает в файл значение и описание
	void printFileData(std::string filename, std::vector<float> data, std::string description);

}


namespace equation {
	bool checkStable1D(std::vector<float>& const vec, float tau, float a);

	std::vector<float> get_u_n_pararlel(std::vector <float>& u_n1, std::vector <float>& u_n2, float tau, std::vector<float>& f_res);

	std::vector<float> get_u_n(std::vector <float> u_n1, std::vector <float> u_n2, float tau, std::vector<float> f_res);


}