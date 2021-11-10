/************************************************************
 * Program to solve a finite difference
 * discretization of the screened Poisson equation:
 * (d2/dx2)u + (d2/dy2)u - alpha u = f
 * with zero Dirichlet boundary condition using the iterative
 * Jacobi method with overrelaxation.
 *
 * RHS (source) function
 *   f(x,y) = -alpha*(1-x^2)(1-y^2)-2*[(1-x^2)+(1-y^2)]
 *
 * Analytical solution to the PDE
 *   u(x,y) = (1-x^2)(1-y^2)
 *
 * Current Version: Christian Iwainsky, RWTH Aachen University
 * MPI C Version: Christian Terboven, RWTH Aachen University, 2006
 * MPI Fortran Version: Dieter an Mey, RWTH Aachen University, 1999 - 2005
 * Modified: Sanjiv Shah,        Kuck and Associates, Inc. (KAI), 1998
 * Author:   Joseph Robicheaux,  Kuck and Associates, Inc. (KAI), 1998
 *
 * Unless READ_INPUT is defined, a meaningful input dataset is used (CT).
 *
 * Input : n     - grid dimension in x direction
 *         m     - grid dimension in y direction
 *         alpha - constant (always greater than 0.0)
 *         tol   - error tolerance for the iterative solver
 *         relax - Successice Overrelaxation parameter
 *         mits  - maximum iterations for the iterative solver
 *
 * On output
 *       : u(n,m)       - Dependent variable (solution)
 *       : f(n,m,alpha) - Right hand side function
 *
 *************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>



/**********************************************************
 * Checks the error between numerical and exact solutions
 **********************************************************/
static inline double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double **u,
                     double deltaX, double deltaY,
                     double alpha)
{
#define U(XX,YY) u[XX][YY]
    int x, y;
    double fX, fY;
    double fy2;
    double localError, error = 0.0;

    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        fy2 = (1.0-fY*fY);
        for (x = 1; x < (maxXCount-1); x++)
        {
            fX = xStart + (x-1)*deltaX;
            localError = U(x,y) - (1.0-fX*fX)*fy2;
            error += localError*localError;
        }
    }
    return sqrt(error)/((maxXCount-2)*(maxYCount-2));
}

#define SRC(XX,YY) u_old[XX][YY]
#define DST(XX,YY) u[XX][YY]

int main(int argc, char **argv)
{
    int n, m, mits;
    double alpha, tol, relax;
    double maxAcceptableError;
    double error;
    double **u, **u_old, **tmp;
    int allocCount;
    int iterationCount, maxIterationCount;
    double t1, t2;

//    printf("Input n,m - grid dimension in x,y direction:\n");
    scanf("%d,%d", &n, &m);
//    printf("Input alpha - Helmholtz constant:\n");
    scanf("%lf", &alpha);
//    printf("Input relax - successive over-relaxation parameter:\n");
    scanf("%lf", &relax);
//    printf("Input tol - error tolerance for the iterrative solver:\n");
    scanf("%lf", &tol);
//    printf("Input mits - maximum solver iterations:\n");
    scanf("%d", &mits);


    printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, tol, mits);

    allocCount = (n+2)*(m+2);
    //Those two calls also zero the boundary elements
    u = 	(double**)calloc((n+2), sizeof(double *)); //reverse order
    u_old = (double**)calloc((n+2), sizeof(double *));
    for(int i=0;i<(n+2);i++){
      u[i] = (double*)calloc((m+2), sizeof(double));
      u_old[i] = (double*)calloc((m+2), sizeof(double));
    }

    if (u == NULL || u_old == NULL)
    {
        printf("Not enough memory for two %ix%i matrices\n", n+2, m+2);
        exit(1);
    }
    maxIterationCount = mits;
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
    error = HUGE_VAL;
    clock_t start = clock(), diff;

    MPI_Init(NULL,NULL);

    int maxXCount=n+2,  maxYCount = m+2;
    double omega = relax;
    double divider = ((maxXCount-2)*(maxYCount-2));

    int x, y;
    double fX, fY;
    double fx2,fy2;
    double temp_error = 0.0;
    double updateVal;
    double f;
    t1 = MPI_Wtime();

    /* Iterate as long as it takes to meet the convergence criterion */
    while (iterationCount < maxIterationCount && error > maxAcceptableError)
    {
        temp_error = 0.0;


        for (y = 1; y < (maxYCount-1); y++)
        {
            fY = (y==1) ? yBottom : fY + deltaY;
            for (x = 1; x < (maxXCount-1); x++)
            {
                fX = (x==1) ? xLeft : fX + deltaX;

                updateVal = (	(u_old[x-1][y] + u_old[x+1][y])*cx + (u_old[x][y-1] + u_old[x][y+1])*cy + u_old[x][y]*cc - (-(alpha*(1.0-fY*fY))*(1.0 - fX*fX) - 2.0*(1.0 - fX*fX) - 2.0*(1.0 - fY*fY)))/cc;

                u[x][y] = u_old[x][y] - omega*updateVal;
                temp_error += updateVal*updateVal;
            }
        }
        error = sqrt(temp_error)/divider;

        iterationCount++;
        // Swap the buffers
        tmp = u_old;
        u_old = u;
        u = tmp;
    }

    t2 = MPI_Wtime();
    printf( "Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, t2 - t1 );
    MPI_Finalize();


    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    printf("Residual %g\n",error);

    // u_old holds the solution after the most recent buffers swap
    double absoluteError = checkSolution(xLeft, yBottom,
                                         n+2, m+2,
                                         u_old,
                                         deltaX, deltaY,
                                         alpha);
    printf("The error of the iterative solution is %g\n", absoluteError);

    return 0;
}

