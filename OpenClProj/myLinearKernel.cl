
__kernel  void firstNonUnoformderivate(__global const  float* f_z, __global const float* x_z, __global float* f_x ){
	int index = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if(x_z[index]!=0)
		f_x[index] = f_z[index] / x_z [index];
	else 
		f_x[index] = 0;
}

__kernel  void secondNonUnoformderivate(__global const  float* f_zz,__global const  float* f_x,__global const  float* x_z ,__global const  float* x_zz, __global float* f_xx){
	int index = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if(x_z[index] !=0) 
		f_xx[index] = f_zz[index]/(x_z[index]*x_z[index]) - (x_zz[index]*f_x[index])/(x_z[index]*x_z[index]*x_z[index]);
	else 
		f_xx[index] = 0;

}