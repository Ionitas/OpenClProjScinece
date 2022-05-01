typedef struct {
  float a;
  float b;
  float c;
} conditions;


__kernel  void wave_kernel(__global const  float* dataDer2,__global const  float* dataDer1,__global const  float* data, __global float* result, float const a, float const b, float const c ){

int index = get_local_size(0) * get_group_id(0) + get_local_id(0);

result[index] = dataDer2[index] + a* dataDer1[index] + b* data[index] -c;

}