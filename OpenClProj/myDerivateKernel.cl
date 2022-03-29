

__kernel  void calc_kernel(__global const  float* data, __global float* result, float const dx , int const count){

int index = get_local_size(0) * get_group_id(0) + get_local_id(0);

if (index >= count-1 )
result[index] = (data[index] - data[index-1]) / dx;
else{
if ( index == 0   )
result[index] = (data[index + 1] - data[index]) / (dx);
else 
result[index] = (data[index + 1] - data[index - 1]) / (2*dx);
}
}