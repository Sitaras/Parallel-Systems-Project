/*
preform vector dot products (inner product).
(x1,x2,x3,x4).(y1,y2,y3,y4) = x1.y1 + x2.y2 + x3.y3 + x4.y4

based on example in:
source: "CUDA by example An Introduction to General-Purpose GPU Programming
        by: Jason Sanders, Edward Kandrot"
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
/*
the number of blocks we launch should be either 32 or
(N+(threadsPerBlock-1)) / threadsPerBlock, whichever value is smaller.
*/
const int blocksPerGrid =imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );


/* define functions to masure execution time on CPU and the GPU */
typedef timeval timestamp;
inline timestamp getTimestamp(void){
	timeval t;
	gettimeofday(&t, NULL);
	return t;
}
inline float getElapsedtime(timestamp t){
	timeval tn;
	gettimeofday(&tn, NULL);
	return (tn.tv_sec - t.tv_sec) * 1000.0f + (tn.tv_usec - t.tv_usec) / 1000.0f;
}

/* CUDA kernel definition */
__global__ void dot( float *a, float *b, float *c ) {
  __shared__ float cache[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  float temp = 0;
  while (tid < N) {
      temp += a[tid] * b[tid];
      tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
      if (cacheIndex < i)
        cache[cacheIndex] += cache[cacheIndex + i];
      __syncthreads();
      i /= 2;
    }

    if (cacheIndex == 0)
      c[blockIdx.x] = cache[0];
}


int main( void ) {
  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;
  cudaError_t err;

  timestamp t_start;

  // allocate memory on the CPU side
  a = (float*)malloc( N*sizeof(float));
  b = (float*)malloc( N*sizeof(float));
  partial_c = (float*)malloc( blocksPerGrid*sizeof(float));

  // allocate the memory on the GPU
  err =cudaMalloc( (void**)&dev_a, N*sizeof(float));
  if (err != cudaSuccess){
    printf("%s\n",cudaGetErrorString(err));
    return 1;
  }
  err =cudaMalloc( (void**)&dev_b, N*sizeof(float));
  if (err != cudaSuccess){
    printf("%s\n",cudaGetErrorString(err));
    return 1;
  }
  err =cudaMalloc( (void**)&dev_partial_c, blocksPerGrid*sizeof(float));
  if (err != cudaSuccess){
    printf("%s\n",cudaGetErrorString(err));
    return 1;
  }

  // fill in the host memory with data
  for (int i=0; i<N; i++) {
    a[i] = i;
    b[i] = i*2;
  }

  // CPU execution
  printf("\nDot Product calculation on CPU\n");
  t_start = getTimestamp();
  for (int i=0; i < N; i++) {
    c = c + a[i] * b[i];
  }
  printf("Execution time %.4f mSecs\n", getElapsedtime(t_start));


  // copy the arrays ‘a’ and ‘b’ to the GPU
  err =cudaMemcpy( dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
    printf("%s\n",cudaGetErrorString(err));
    return 1;
  }
  err =cudaMemcpy( dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
    printf("%s\n",cudaGetErrorString(err));
    return 1;
  }

  // CUDA execution
  printf("\nDot Product calculation on GPU\n");
  t_start = getTimestamp();
  dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b, dev_partial_c );

  printf("Execution time %.4f mSecs\n", getElapsedtime(t_start));

  // copy the array 'c' back from the GPU to the CPU
  err =cudaMemcpy( partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess){
    printf("%s\n",cudaGetErrorString(err));
    return 1;
  }

  // finish up on the CPU side
  c = 0;
  for (int i=0; i<blocksPerGrid; i++) {
    c += partial_c[i];
  }


  /* since our data are filled with integers from 0 to N-1  the dot product
  should be two times the sum of the squares of the integers from 0 to N-1.
  */
  #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
  printf( "result check: %.6g = %.6g?\n", c, 2 * sum_squares( (float)(N - 1) ) );


  // free memory on the GPU side
  cudaFree( dev_a );
  cudaFree( dev_b );
  cudaFree( dev_partial_c );

  // free memory on the CPU side
  free( a );
  free( b );
  free( partial_c );

  return 0;
}

