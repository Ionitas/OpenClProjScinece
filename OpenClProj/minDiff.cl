

__kernel  void minDiff(__global const float* data, __global float* result, int count){

int index = get_local_size(0) * get_group_id(0) + get_local_id(0);

if (index < count){
	result[index] =  data[index+1] - data[index];
}
}