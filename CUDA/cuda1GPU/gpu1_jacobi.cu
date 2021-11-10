#include <lcutil.h>
#include <timestamp.h>


__global__ void compute(double xStart, double yStart,
                           int maxXCount, int maxYCount,
                           double *src, double *dst,
                           double deltaX, double deltaY,
                           double alpha, double omega,double cx,double cy,double cc,double * totalError){
 #define SRC(XX,YY) src[(YY)*maxXCount+(XX)]
 #define DST(XX,YY) dst[(YY)*maxXCount+(XX)]
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        int x = i/maxXCount;
        int y = i%maxYCount;
        double fX, fY;
        double error = 0.0;
        double updateVal;
        double f;
        if( (x>0 && x<(maxXCount-1)) && (y>0 && y < (maxYCount-1))){
              fY = yStart + (y-1)*deltaY;
              fX = xStart + (x-1)*deltaX;
              f = -alpha*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY);
              updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                  			(SRC(x,y-1) + SRC(x,y+1))*cy +
                  			SRC(x,y)*cc - f
  						)/cc;
              DST(x,y) = SRC(x,y) - omega*updateVal;
              error += updateVal*updateVal;
              atomicAdd(totalError,error);
        }
}



extern "C" double one_jacobi_iteration(double xStart, double yStart,
                           int maxXCount, int maxYCount,
                           double *src, double *dst,
                           double deltaX, double deltaY,
                           double alpha, double omega,double cx,double cy,double cc,double * totalTime){
        double *d_src, *d_dst;
        cudaError_t err;

        err =cudaMalloc((void**)&d_src, maxXCount*maxYCount*sizeof(double));
        if (err != cudaSuccess){
                fprintf(stderr, "GPUassert: %s\n",err);
                fflush(stdout);
                return err;
        }

        err =cudaMalloc((void**)&d_dst, maxXCount*maxYCount*sizeof(double));
        if (err != cudaSuccess){
                fprintf(stderr, "GPUassert: %s\n",err);
                fflush(stdout);
                return err;
        }


        // Copy data to device memory
        err =cudaMemcpy(d_src, src, maxXCount*maxYCount*sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess){
                fprintf(stderr, "GPUassert: %s\n",err);
                fflush(stdout);
                return err;
        }

        err =cudaMemcpy(d_dst, dst, maxXCount*maxYCount*sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess){
                fprintf(stderr, "GPUassert: %s\n",err);
                fflush(stdout);
                return err;
        }

        double *totalError;
        err =cudaMalloc((void**)&totalError, sizeof(double));
        if (err != cudaSuccess){
                fprintf(stderr, "GPUassert: %s\n",err);
                fflush(stdout);
                return err;
        }
        err =cudaMemset(totalError, 0, sizeof(double));  // initialize to zeros
        if (err != cudaSuccess){
                fprintf(stderr, "GPUassert: %s\n",err);
                fflush(stdout);
                return err;
        }


        timestamp t_start;
        t_start = getTimestamp();
        const int N = maxXCount*maxYCount;
        const int BLOCK_SIZE = 512;
        dim3 dimBl(BLOCK_SIZE);
        dim3 dimGr(FRACTION_CEILING(N, BLOCK_SIZE));
        compute<<<dimGr, dimBl>>>(xStart, yStart,
                                  maxXCount, maxYCount,
                                   d_src, d_dst,
                                   deltaX, deltaY,
                                   alpha, omega,cx,cy,cc,totalError);

        err =cudaGetLastError();
        if (err != cudaSuccess){
                fprintf(stderr, "GPUassert: %s\n",err);
                fflush(stdout);
                return err;
        }

        err =cudaDeviceSynchronize();
        if (err != cudaSuccess){
                fprintf(stderr, "GPUassert: %s\n",err);
                fflush(stdout);
                return err;
        }

        float msecs = getElapsedtime(t_start);
        (*totalTime) += msecs;

        // Copy results back to host memory
        err =cudaMemcpy(dst, d_dst, maxXCount*maxYCount*sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess){
                fprintf(stderr, "GPUassert: %s\n",err);
                return err;
        }


        err =cudaFree(d_src);
        if (err != cudaSuccess){
                fprintf(stderr, "GPUassert: %s\n",err);
                fflush(stdout);
                return err;
        }

        err =cudaFree(d_dst);
        if (err != cudaSuccess){
                fprintf(stderr, "GPUassert: %s\n",err);
                fflush(stdout);
                return err;
        }

        double tempErr;
        err =cudaMemcpy(&tempErr, totalError, sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess){
                fprintf(stderr, "GPUassert: %s\n",err);
                return err;
        }

        err =cudaFree(totalError);
        if (err != cudaSuccess){
                fprintf(stderr, "GPUassert: %s\n",err);
                fflush(stdout);
                return err;
        }

        return sqrt(tempErr)/((maxXCount-2)*(maxYCount-2));
}

