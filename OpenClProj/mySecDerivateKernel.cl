

__kernel  void calcSecDeriv_kernel(__global const  float* data, __global float* result, float const dx , int const count){

int index = get_local_size(0) * get_group_id(0) + get_local_id(0);

if (index >= count-2 )
	result[index] = (-data[index-3] + 4*data[index-2] - 5*data[index-1] +2*data[index])/(dx*dx);
else{
	if ( index <= 1   )
		result[index] = (2*data[index]-5*data[index+1] +4*data[index+2]-data[index+3])/(dx*dx);
	else 
		result[index] = (data[index+1] - 2*data[index] + data[index-1])/(dx*dx);
	}
}