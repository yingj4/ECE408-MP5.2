// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 64 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

/*Kernel 1 & 2: scan*/
__global__ void scan(float *input, float *output, bool auxiArr, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2 * BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * blockDim.x;
  unsigned int ldX;
  unsigned int ldStride;
  
  if (!auxiArr) {
    ldX = t + start;
    ldStride = blockDim.x;
  }
  else {
    ldX = (t + 1) * BLOCK_SIZE * 2 - 1; //See the "index" variable below
    ldStride = blockDim.x * BLOCK_SIZE * 2;
  }
  
  
  if (ldX < len) {
    T[t] = input[ldX];
  }
  else {
    T[t] = 0;
  }
  if (ldX + BLOCK_SIZE < len) {
    T[t + BLOCK_SIZE] = input[ldX + ldStride];
  }
  else {
    T[t + BLOCK_SIZE] = 0;
  }
  
  /*Reduce kernel*/
  int stride = 1;
  while (stride < 2 * BLOCK_SIZE) {
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE && (index - stride) >= 0) {
      T[index] += T[index - stride];
    }
    stride *= 2;
    
    
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
  
  __syncthreads();
  
  /*
  __shared__ float add;
  
  if (t == 0) {
    if (blockIdx.x > 0) {
      add = output[start - 1];
    }
    else {
      add = 0;
    }
  }
  */
  
  
  if (t + start < len) {
    output[t + start] = T[t];
  }
  if (t + start + BLOCK_SIZE < len) {
    output[t + start + BLOCK_SIZE] = T[t + BLOCK_SIZE];
  }
  
  __syncthreads();
  
}

/*Kernel 3: add*/

__global__ void add(float* deviceScanSums, float* deviceAuxiArr, float* deviceOutput, int numElements){
  unsigned int t = threadIdx.x;
  unsigned int start = blockIdx.x * BLOCK_SIZE * 2;
  
  __shared__ float temp;
  if (t == 0) {
    temp = 0;
    if (blockIdx.x > 0) {
      temp = deviceScanSums[blockIdx.x - 1];
    }
  }
  __syncthreads();
  
  if (t + start < numElements) {
    deviceOutput[t + start] = deviceAuxiArr[t + start] + temp;
  }
  if (t + start + BLOCK_SIZE < numElements) {
    deviceOutput[t + start + BLOCK_SIZE] = deviceAuxiArr[t + start + BLOCK_SIZE] + temp;
  }
  
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  
  float* deviceAuxiArr; // The auxiliary added by Ying
  float* deviceScanSums; // The scan block sums added by Ying

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
  wbCheck(cudaMalloc((void**)&deviceAuxiArr, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceScanSums, ceil(numElements / (2.0 * BLOCK_SIZE)) * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements / (2.0 * BLOCK_SIZE)), 1, 1);
  dim3 dimAuxiGrid(1, 1, 1);
  dim3 dimBlock((BLOCK_SIZE * 1), 1, 1);
  dim3 dimAuxiBlock(ceil(numElements / (2.0 * BLOCK_SIZE)), 1, 1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  /*
  for(int i = 0; i < ceil(numElements / (1.0 * BLOCK_SIZE)); ++i) {
    int bIdx = i * BLOCK_SIZE;
    
    if(i > 0) {
      float temp = 0;
      cudaMemcpy(&temp, &deviceOutput[bIdx - 1], sizeof(float), cudaMemcpyDeviceToHost);
      temp += hostInput[bIdx];
      cudaMemcpy(&deviceInput[bIdx], &temp, sizeof(float),cudaMemcpyHostToDevice);
    }
    
    scan<<<dimGrid, dimBlock>>>(&deviceInput[bIdx], &deviceOutput[bIdx], (numElements - bIdx));
  }
  */
  
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceAuxiArr, false, numElements);
  cudaDeviceSynchronize();
  scan<<<dimAuxiGrid, dimAuxiBlock>>>(deviceAuxiArr, deviceScanSums, true, numElements);
  cudaDeviceSynchronize();
  add<<<dimGrid, dimBlock>>>(deviceScanSums, deviceAuxiArr, deviceOutput, numElements);
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
