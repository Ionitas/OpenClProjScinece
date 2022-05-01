

__kernel  void get_u_n(__global const float* u_n1,__global const float* u_n2,__global const float* f_res,__global  float* res, float const tau){

int i = get_local_size(0) * get_group_id(0) + get_local_id(0);

res[i] = (4 * u_n1[i] - 3 * u_n2[i] + 2 * tau * f_res[i]);
}



__kernel  void heat_calc(__global const float* data,__global float* res,__global const float* f_res, float const a){
int index = get_local_size(0) * get_group_id(0) + get_local_id(0);
res[index] = a*a* data[index];


}