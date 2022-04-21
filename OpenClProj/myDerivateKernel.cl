
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

if (index >= count-2 )
	result[index] = (-data[index-3] + 4*data[index-2] - 5*data[index-1] +2*data[index])/(dx*dx);
else{
	if ( index <= 1   )
		result[index] = (2*data[index]-5*data[index+1] +4*data[index+2]-data[index+3])/(dx*dx);
	else 
		result[index] = (data[index+1] - 2*data[index] + data[index-1])/(dx*dx);
	}
}

