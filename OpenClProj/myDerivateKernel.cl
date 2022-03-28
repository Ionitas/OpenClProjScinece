

__kernel  void calc_kernel(__global const  float* data, __global float* result, float const dx ){




int indexCurrent =  get_gloval_id(0); //get_local_size(0) * get_group_id(0) + get_local_id(0);

if ( indexCurrent == get_global_size(0) ) {
result[indexCurrent] = (data[indexCurrent] - data[indexCurrent-1]) / dx;
}
else {
result[indexCurrent] = (data[indexCurrent + 1] - data[indexCurrent - 1]) / (2*dx);
}

}