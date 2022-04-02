#include <iostream>
#include "permute_cpu.h"
#include "permute_gpu.h"

void test_cpu_permute() {
  int W = 3, H = 3, CI = 9, CO = 2;
  size_t buff_isize = CO * CI * H * W;
  float *arr = new float[buff_isize/4];
  for (int i = 0; i < buff_isize / 4; ++i) {
    arr[i] = i * 1.0;
  }
  Tensor::PermuteCPU cpu_permuter;
  float *outarr = cpu_permuter.DoPermute("nh|c4w4", "nchw", {CO, CI, H, W}, arr);
}
int main() {
  test_cpu_permute();
  return 0;
}