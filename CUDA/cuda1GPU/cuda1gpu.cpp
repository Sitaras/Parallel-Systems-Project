#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define VECT_SIZE 15000000

extern "C" double one_jacobi_iteration(double , double ,
                           int , int ,
                           double *, double *,
                           double , double ,
                           double , double ,double ,double ,double ,double * );


/**********************************************************
 * Checks the error between numerical and exact solutions
 **********************************************************/
double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double *u,
                     double deltaX, double deltaY,
                     double alpha)
{
#define U(XX,YY) u[(YY)*maxXCount+(XX)]
    int x, y;
    double fX, fY;
    double localError, error = 0.0;

    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        for (x = 1; x < (maxXCount-1); x++)
        {
            fX = xStart + (x-1)*deltaX;
            localError = U(x,y) - (1.0-fX*fX)*(1.0-fY*fY);
            error += localError*localError;
        }
    }
    return sqrt(error)/((maxXCount-2)*(maxYCount-2));
}

void readInputFile(int* n, int* m, double* alpha, double* relax, double* tol, int* mits){

      //    printf("Input n,m - grid dimension in x,y direction:\n");
      scanf("%d,%d", n, m);
      //    printf("Input alpha - Helmholtz constant:\n");
      scanf("%lf", alpha);
      //    printf("Input relax - successive over-relaxation parameter:\n");
      scanf("%lf", relax);
      //    printf("Input tol - error tolerance for the iterrative solver:\n");
      scanf("%lf", tol);
      //    printf("Input mits - maximum solver iterations:\n");
      scanf("%d", mits);
      printf( "READ n: %d, m: %d, alpha: %f, relax: %f, tol: %f, mits: %d \n", *n,*m,*alpha,*relax,*tol,*mits);
}

int main(int argc, char* argv[]) {

        size_t freeCUDAMem, totalCUDAMem;
        cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
        printf("Total GPU memory %zu, free %zu\n", totalCUDAMem, freeCUDAMem);



        int n, m, mits;
        double alpha, tol, relax;

        double maxAcceptableError;
        double *u, *u_old, *tmp;
        int allocCount;
        int iterationCount;

        readInputFile(&n, &m, &alpha, &relax, &tol, &mits);



        allocCount = (n+2)*(m+2);

        maxAcceptableError = tol;

        // Solve in [-1, 1] x [-1, 1]
        double xLeft = -1.0, xRight = 1.0;
        double yBottom = -1.0, yUp = 1.0;

        double deltaX = (xRight-xLeft)/(n-1);
        double deltaY = (yUp-yBottom)/(m-1);

        // Coefficients
        double cx = 1.0/(deltaX*deltaX);
        double cy = 1.0/(deltaY*deltaY);
        double cc = -2.0*cx-2.0*cy-alpha;

        iterationCount = 0;





        double error = HUGE_VAL;

        u = 	(double*)calloc(allocCount, sizeof(double)); //reverse order
        u_old = (double*)calloc(allocCount, sizeof(double));


        double totalTime = 0.0;

        while (iterationCount < mits && error > maxAcceptableError)
        {

            error = one_jacobi_iteration(xLeft, yBottom,
                                         n+2, m+2,
                                         u_old, u,
                                         deltaX, deltaY,
                                         alpha, relax,cx,cy,cc,&totalTime);

            iterationCount++;
            // Swap the buffers
            tmp = u_old;
            u_old = u;
            u = tmp;
        }

        printf( "Iterations=%3d Elapsed CUDA Wall time is %f msec\n", iterationCount, totalTime );

        printf("Time taken %d seconds %5d milliseconds\n", (int)totalTime/1000, (int)totalTime%1000);
        printf("Residual %g\n",error);

        // u_old holds the solution after the most recent buffers swap
        double absoluteError = checkSolution(xLeft, yBottom,
                                             n+2, m+2,
                                             u_old,
                                             deltaX, deltaY,
                                             alpha);
        printf("The error of the iterative solution is %g\n", absoluteError);

        free(u);
        free(u_old);

        return 0;
}

