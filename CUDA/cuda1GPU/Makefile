CUDA_INSTALL_PATH = /usr/local/cuda-11.1
CC = g++
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I../common/inc/
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR}
NVFLAGS = -arch=sm_60 -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
        LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart
CLEAN_FILES = cuda1gpu gpu1_jacobi.o cuda1gpu.o

cuda1gpu: cuda1gpu.o gpu1_jacobi.o
	${CC} ${LFLAGS} -o $@ $^
	cp $@ ../release

cuda1gpu.o: cuda1gpu.cpp
	${CC} -c ${FLAGS} -o $@ $^

gpu1_jacobi.o: gpu1_jacobi.cu
	${NVCC} ${NVFLAGS} -DUNIX -c $^ -o $@

clean:
	\rm -f $(CLEAN_FILES)
	\rm -f ../release/cuda1gpu

rebuild: clean cuda1gpu
