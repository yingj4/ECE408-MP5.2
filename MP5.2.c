// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 128 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

/*
/* Helper function for the reduction step kerel
void redKernel(float* T, unsigned int t) {
  int stride = 1;
  while (stride < 2 * BLOCK_SIZE) {
    int index = (t + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE && (index - stride) >= 0) {
      T[index] += T[index - stride];
    }
    stride *= 2;
    
    __syncthreads();
  }
}

/*Helper function for the post scan step
void postScan(float* T, unsigned int t) {
  int stride = BLOCK_SIZE / 2;
  while (stride > 0) {
    __syncthreads();
    
    int index = (t + 1) * stride * 2 - 1;
    if (index + stride < 2 * BLOCK_SIZE) {
      T[index + stride] += T[index];
    }
    stride /= 2;    
  }
}
*/

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2 * BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * blockDim.x;
  if (t + start < len) {
    T[t] = input[t + start];
  }
  else {
    T[t] = 0;
  }
  if (t + start + BLOCK_SIZE < len) {
    T[t + BLOCK_SIZE] = input[t + start + BLOCK_SIZE];
  }
  else {
    T[t + BLOCK_SIZE] = 0;
  }
  
  /*Reduce kernel*/
  int stride = 1;
  while (stride < 2 * BLOCK_SIZE) {
    int index = (t + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE && (index - stride) >= 0) {
      T[index] += T[index - stride];
    }
    stride *= 2;
    
    __syncthreads();
  }
  
  /*Post scan*/
  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
    __syncthreads();
    
    int index = (t + 1) * stride * 2 - 1;
    if (index + stride < 2 * BLOCK_SIZE) {
      T[index + stride] += T[index];
    }
    stride /= 2;    
  }
  
  if (t + start < len) {
    output[t + start] = T[t];
    if (blockIdx.x > 0) {
      output[t + start] += output[start - 1];
    }
  }
  if (t + start + BLOCK_SIZE < len) {
    output[t + start + BLOCK_SIZE] = T[t + BLOCK_SIZE];
    if (blockIdx.x > 0) {
      output[t + start + BLOCK_SIZE] += output[start - 1]; 
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid = (ceil(numElements / (1.0 * BLOCK_SIZE)), 1, 1);
  dim3 dimBlock = ((BLOCK_SIZE * 1), 1, 1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
