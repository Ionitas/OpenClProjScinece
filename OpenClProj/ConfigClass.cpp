#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>


class ConfigData {

private :



public:
	std::vector<double> boundaryConditionsVec; // AlphaU+BettaU'=Gamma
	std::vector<double> initialConditions;

	float functionf(float x) {
		return sin(x);
	}

	




};