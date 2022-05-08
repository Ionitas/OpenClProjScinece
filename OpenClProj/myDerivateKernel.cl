
__kernel  void first_dirivate(__global const float* data, __global float* result, float const dx , int const count){

int index = get_local_size(0) * get_group_id(0) + get_local_id(0);

if (index >= count-1)
	result[index] = -(3*data[index-2] - 4*data[index-1] + data[index]) / (2*dx);
else{
	if ( index == 0)
		result[index] = (3*data[index+2] - 4*data[index+1] + data[index]) / (2*dx);
	else 
		result[index] = (data[index + 1] - data[index - 1]) / (2*dx); 
	}
}



__kernel  void second_derivate(__global const  float* data, __global float* result, float const dx , int const count){

int index = get_local_size(0) * get_group_id(0) + get_local_id(0);

if (index >= count-1 )
	result[index] = (-data[index-3] + 4*data[index-2] - 5*data[index-1] +2*data[index])/(dx*dx);
else{
	if ( index < 1   )
		result[index] = (2*data[index]-5*data[index+1] +4*data[index+2]-data[index+3])/(dx*dx);
	else 
		result[index] = (data[index+1] - 2*data[index] + data[index-1])/(dx*dx);
	}
}

__kernel  void minDiff(__global const float* data, __global float* result, int count) {

	int index = get_local_size(0) * get_group_id(0) + get_local_id(0);

	if (index < count) {
		result[index] = data[index + 1] - data[index];
	}
}

__kernel  void get_u_n(__global const float* u_n1, __global const float* u_n2, __global const float* f_res, __global  float* res, float const tau) {
	int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	res[i] = (4 * u_n1[i] - 3 * u_n2[i] + 2 * tau * f_res[i]);
}



__kernel  void heat_calc(__global const float* data, __global float* res, float const a) {
	int index = get_local_size(0) * get_group_id(0) + get_local_id(0);
	res[index] = a * a * data[index];
}


__kernel  void firstNonUnoformderivate(__global const  float* f_z, __global const float* x_z, __global float* f_x) {
	int index = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (x_z[index] != 0)
		f_x[index] = f_z[index] / x_z[index];
	else
		f_x[index] = 0;
}

__kernel  void secondNonUnoformderivate(__global const  float* f_zz, __global const  float* f_x, __global const  float* x_z, __global const  float* x_zz, __global float* f_xx) {
	int index = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (x_z[index] != 0)
		f_xx[index] = f_zz[index] / (x_z[index] * x_z[index]) - (x_zz[index] * f_x[index]) / (x_z[index] * x_z[index] * x_z[index]);
	else
		f_xx[index] = 0;

}

typedef struct {
	float a;
	float b;
	float c;
} conditions;


__kernel  void wave_kernel(__global const  float* dataDer2, __global const  float* dataDer1, __global const  float* data, __global float* result, float const a, float const b, float const c) {

	int index = get_local_size(0) * get_group_id(0) + get_local_id(0);

	result[index] = dataDer2[index] + a * dataDer1[index] + b * data[index] - c;

}

__kernel  void nextun(__global const  float* uu, __global const  float* dudt, __global   float* result,  float const dt ) {

	int index = get_local_size(0) * get_group_id(0) + get_local_id(0);

	result[index] = uu[index] + dt * dudt[index] ;

}

