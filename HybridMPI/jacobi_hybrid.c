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
#include <stdbool.h>
#include <mpi.h>
#include <omp.h>


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

    for (y = 1; y < (maxYCount-1); y++){
        fY = yStart + (y-1)*deltaY;
        fy2 = (1.0-fY*fY);
        for (x = 1; x < (maxXCount-1); x++){
            fX = xStart + (x-1)*deltaX;
            localError = U(x,y) - (1.0-fX*fX)*fy2;
            error += localError*localError;
        }
    }
    return error;
}

void readInputFile(int myrank, int* n, int* m, double* alpha, double* relax, double* tol, int* mits){

  if (myrank == 0) {
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
  }

  MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(relax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(tol, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(mits, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (myrank == 0){
    printf( "Input n: %d, m: %d, alpha: %f, relax: %f, tol: %f, mits: %d \n",myrank, *n,*m,*alpha,*relax,*tol,*mits);
  }
}


static inline double computeWhites(double xStart, double yStart,
                            int size,
                            double **src, double **dst,
                            double deltaX, double deltaY,
                            double alpha, double omega,double cx,double cy,double cc){
#define SRC(XX,YY) src[XX][YY]
#define DST(XX,YY) dst[XX][YY]
#define FX(X)	(double)(xStart +(X-1)*deltaX)
#define FY(Y) (double)(yStart + (Y-1)*deltaY)
#define F(x,y) (double)(-alpha*(1.0-FX(x)*FX(x))*(1.0-FY(y)*FY(y)) - 2.0*(1.0-FX(x)*FX(x)) - 2.0*(1.0-FY(y)*FY(y)))


      int x, y;
      double fX, fY;
      double fx2,fy2;
      double temp_error = 0.0;
      double updateVal;
      double f;

      #pragma omp parallel for collapse(2) shared(size,src,dst,xStart,yStart,deltaX,deltaY,alpha,omega,cx,cy,cc),private(y,x,updateVal),reduction(+:temp_error) schedule(static)
      for (x = 2; x < (size-2); x++){
          for (y = 2; y < (size-2); y++){
              updateVal = ((SRC(x-1,y) + SRC(x+1,y))*cx + (SRC(x,y-1) + SRC(x,y+1))*cy + SRC(x,y)*cc - F(x,y))/cc;
              DST(x,y) = SRC(x,y) - omega*updateVal;
              temp_error += updateVal*updateVal;
          }
      }

      return temp_error;

}

static inline double computeGreens(double xStart, double yStart,
                            int size,
                            double **src, double **dst,
                            double deltaX, double deltaY,
                            double alpha, double omega,double cx,double cy,double cc){
#define SRC(XX,YY) src[XX][YY]
#define DST(XX,YY) dst[XX][YY]
#define FX(X)	(double)(xStart +(X-1)*deltaX)
#define FY(Y)   (double)(yStart + (Y-1)*deltaY)
#define F(x,y)   (double)(-alpha*(1.0-FX(x)*FX(x))*(1.0-FY(y)*FY(y)) - 2.0*(1.0-FX(x)*FX(x)) - 2.0*(1.0-FY(y)*FY(y)))

    int x, y;
    double temp_error = 0.0;
    double updateVal;


    #pragma omp parallel for shared(size,src,dst,xStart,yStart,deltaX,deltaY,alpha,omega,cx,cy,cc),private(y,x,updateVal),reduction(+:temp_error) schedule(static)
    for(x=1;x<(size-1);x++){
      int y=1;
      updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx + (SRC(x,y-1) + SRC(x,y+1))*cy + SRC(x,y)*cc - F(x,y))/cc;
      DST(x,y) = SRC(x,y) - omega*updateVal;
      temp_error += updateVal*updateVal;

      y=size-2;
      updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx + (SRC(x,y-1) + SRC(x,y+1))*cy + SRC(x,y)*cc - F(x,y))/cc;
      DST(x,y) = SRC(x,y) - omega*updateVal;
      temp_error += updateVal*updateVal;
    }


    #pragma omp parallel for shared(size,src,dst,xStart,yStart,deltaX,deltaY,alpha,omega,cx,cy,cc),private(y,x,updateVal),reduction(+:temp_error) schedule(static)
    for(y=2;y<(size-2);y++){
      int x=1;
      updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx + (SRC(x,y-1) + SRC(x,y+1))*cy + SRC(x,y)*cc - F(x,y))/cc;
      DST(x,y) = SRC(x,y) - omega*updateVal;
      temp_error += updateVal*updateVal;

      x=size-2;
      updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx + (SRC(x,y-1) + SRC(x,y+1))*cy + SRC(x,y)*cc - F(x,y))/cc;
      DST(x,y) = SRC(x,y) - omega*updateVal;
      temp_error += updateVal*updateVal;
    }

    return temp_error;

}



#define inBounds(i,j,arraySize) (i<0 || j<0 || i >= arraySize || j >= arraySize)? 0 : 1
#define hasNeighbor(x) (neighbours[x]!=MPI_PROC_NULL)? 1 : 0
#define SOUTH 0
#define NORTH 1
#define EAST 2
#define WEST 3
#define N2S 222 // North part to South part
#define W2E 333 // West part to East part
#define S2N 444 // South to North
#define E2W 555 // Easth to West
#define FALSE 0
#define TRUE 1

#define UOLD(XX,YY) u_old[XX][YY]

int main(int argc, char **argv)
{


    int error = MPI_Init(NULL,NULL);

    if(error<0){
      perror("INIT: ");
    }
    int numprocs,myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int n, m, mits;
    double alpha, tol, relax;

    double globalError = HUGE_VAL;
    double globalCheckSolution = HUGE_VAL;

    double maxAcceptableError;
    double **u, **u_old, **tmp;
    int allocCount;
    int iterationCount, maxIterationCount;
    double t1, t2;


    readInputFile(myrank, &n, &m, &alpha, &relax, &tol, &mits);



    allocCount = (n+2)*(m+2);

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



    clock_t start = clock(), diff;
    int new_size = n / sqrt(numprocs);
    u = 	(double**)calloc((new_size+2), sizeof(double*)); //reverse order
    u_old = (double**)calloc((new_size+2), sizeof(double*));
    if (u == NULL || u_old == NULL)
    {
        printf("Not enough memory for two %ix%i matrices\n", n+2, m+2);
        exit(1);
    }
    for(int i=0;i<(new_size+2);i++){
      u[i] = (double*)calloc((new_size+2), sizeof(double));
      u_old[i] = (double*)calloc((new_size+2), sizeof(double));
    }


    // Ask MPI to decompose our processes in a 2D cartesian grid for us
    int dims[2] = {0, 0};
    MPI_Dims_create(numprocs, 2, dims);

    // Make both dimensions non-periodic
    int periods[2] = {false, false};

    // Let MPI assign arbitrary ranks if it deems it necessary
    int reorder = true;

    // Create a communicator with a cartesian topology.
    MPI_Comm new_communicator;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &new_communicator);


    int neighbours[4];

    // Let consider dims[0] = X, so the shift tells us our left and right neighbours
    MPI_Cart_shift(new_communicator, 0, 1, &neighbours[SOUTH], &neighbours[NORTH]);

    // Let consider dims[1] = Y, so the shift tells us our up and down neighbours
    MPI_Cart_shift(new_communicator, 1, 1, &neighbours[WEST], &neighbours[EAST]);

    int my_coords[2];
    MPI_Cart_coords(new_communicator, myrank, 2, my_coords);

    MPI_Datatype row_type;
    MPI_Type_contiguous(new_size+2, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    MPI_Datatype column_type;
    MPI_Datatype dummy;
    MPI_Type_vector(new_size+2, 1, new_size+2, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);


    int maxXCount=n+2,  maxYCount = m+2;
    double omega = relax;
    double divider = ((maxXCount-2)*(maxYCount-2));

    int myStartY = (my_coords[0]*(n/dims[0]));
    double myYBottom = yBottom + myStartY*deltaY;
    int myStartX = (my_coords[1]*(n/dims[1]));
    double myXBottom = xLeft + myStartX*deltaX;

    MPI_Request RRequest[4], SRequest[4];
  	MPI_Status  Rstatus[4], Sstatus[4];
    double ReceiveBuf[4][new_size+2];
    double SendBuf[4][new_size+2];

    MPI_Barrier(new_communicator);
    t1 = MPI_Wtime();



    /* Iterate as long as it takes to meet the convergence criterion */
    while (iterationCount < maxIterationCount && globalError > maxAcceptableError)
    {

      MPI_Irecv(&UOLD(new_size+1,0), 1, row_type, neighbours[SOUTH], N2S, new_communicator, &RRequest[SOUTH]);
      MPI_Irecv(&UOLD(0,0), 1, row_type, neighbours[NORTH], S2N, new_communicator, &RRequest[NORTH]);
      MPI_Irecv(&UOLD(0,new_size+1), 1, column_type, neighbours[EAST], W2E, new_communicator, &RRequest[EAST]);
      MPI_Irecv( &UOLD(0,0), 1, column_type, neighbours[WEST], E2W, new_communicator, &RRequest[WEST]);

      MPI_Isend( &UOLD(new_size,0), 1, row_type, neighbours[SOUTH], S2N, new_communicator, &SRequest[SOUTH]);
      MPI_Isend( &UOLD(1,0), 1, row_type, neighbours[NORTH], N2S, new_communicator, &SRequest[NORTH]);
      MPI_Isend( &UOLD(0,new_size), 1, column_type, neighbours[EAST], E2W, new_communicator, &SRequest[EAST]);
      MPI_Isend( &UOLD(0,1),1, column_type, neighbours[WEST], W2E, new_communicator, &SRequest[WEST]);



        double localError=0.0;

        localError = computeWhites(xLeft,yBottom,new_size+2,u_old,u, deltaX, deltaY, alpha, omega,cx,cy,cc);


        MPI_Wait(&RRequest[SOUTH], &Rstatus[SOUTH]);
        MPI_Wait(&RRequest[NORTH], &Rstatus[NORTH]);
        MPI_Wait(&RRequest[EAST], &Rstatus[EAST]);
        MPI_Wait(&RRequest[WEST], &Rstatus[WEST]);


        localError += computeGreens(xLeft,yBottom,new_size+2,u_old,u, deltaX, deltaY, alpha, omega,cx,cy,cc);

        // Swap the buffers
        tmp = u_old;
        u_old = u;
        u = tmp;

        iterationCount++;

        globalError=0.0;

        MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_SUM, new_communicator);

        globalError = sqrt(globalError)/(double)divider;

        MPI_Wait(&SRequest[SOUTH], &Sstatus[SOUTH]);
        MPI_Wait(&SRequest[NORTH], &Sstatus[NORTH]);
        MPI_Wait(&SRequest[EAST], &Sstatus[EAST]);
        MPI_Wait(&SRequest[WEST], &Sstatus[WEST]);

    }

    t2 = MPI_Wtime();


    if(myrank==0){
      printf( "Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, t2 - t1 );

      diff = clock() - start;
      int msec = diff * 1000 / CLOCKS_PER_SEC;
      printf("Time taken %d seconds %5d milliseconds\n", msec/1000, msec%1000);
      printf("Residual %g\n",globalError);
    }

    double local_check_sol = checkSolution(myXBottom, myYBottom, new_size+2, new_size+2, u_old, deltaX, deltaY, alpha);
    MPI_Reduce(&local_check_sol, &globalCheckSolution, 1, MPI_DOUBLE, MPI_SUM, 0, new_communicator);
    if(myrank == 0){
      globalCheckSolution = sqrt(globalCheckSolution) / divider;
      printf("The error of the iterative solution is %g\n", globalCheckSolution);
    }


    MPI_Finalize();


    return 0;
}

