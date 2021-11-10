CUDA_INSTALL_PATH = /usr/local/cuda-11.1
CC = g++
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops -fopenmp
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I../common/inc/
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR} -fopenmp
NVFLAGS = -arch=sm_60 -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
        LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart -fopenmp
CLEAN_FILES = cuda2gpu gpu2_jacobi.o cuda2gpu.o

cuda2gpu: cuda2gpu.o gpu2_jacobi.o
	${CC} ${LFLAGS} -o $@ $^
	cp $@ ../release

cuda2gpu.o: cuda2gpu.cpp
	${CC} -c ${FLAGS} -o $@ $^

gpu2_jacobi.o: gpu2_jacobi.cu
	${NVCC} ${NVFLAGS} -DUNIX -c $^ -o $@

clean:
	\rm -f $(CLEAN_FILES)
	\rm -f ../release/cuda2gpu

rebuild: clean cuda2gpu

