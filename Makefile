#####################################
#Build binaries for the code in the paper
#see params.h for the parameters

SOURCES = main.cu GPU.cu kernel.cu import_dataset.cpp par_sort.cpp
OBJECTS = par_sort.o import_dataset.o 
CUDAOBJECTS = GPU.o kernel.o main.o 
CC = nvcc
GNUCC = g++
EXECUTABLE = main

#Change to your CUDA Compute Capability:
COMPUTE_CAPABILITY = 75
COMPUTE_CAPABILITY_FLAGS = -arch=compute_$(COMPUTE_CAPABILITY) -code=sm_$(COMPUTE_CAPABILITY) 

FLAGS = -std=c++14 -O3 -Xcompiler -fopenmp -lcuda -lineinfo
CFLAGS = -c





all: $(EXECUTABLE)

#Compile the parallel sort function separately with g++ because
#it doesn't compile with nvcc
par_sort.o: par_sort.cpp
	$(GNUCC) -O3 $(CFLAGS) par_sort.cpp

import_dataset.o: import_dataset.cpp params.h
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) $<

main.o: main.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) main.cu 

kernel.o: kernel.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) kernel.cu 		

GPU.o: GPU.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) GPU.cu


$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@




#End make for code in the paper
#####################################

#####################################
#Make python shared library

PYTHONFLAGS= -Xcompiler -fPIC --shared -DPYTHON

#Update the parameters file with the makefile to set the number of dimensions
#the parameters below will overwrite the params.h file

#The number of data dimensions (n)
MAKEDIMDATA = 2

#The number indexed dimensions (k) (default is 6, must be <=n)
MAKEDIMINDEXED = 2



make_python_shared_lib: 
	sed -i "s/#define GPUNUMDIM.*/#define GPUNUMDIM $(MAKEDIMDATA)/g" params.h
	sed -i "s/#define NUMINDEXEDDIM.*/#define NUMINDEXEDDIM $(MAKEDIMINDEXED)/g" params.h
	$(GNUCC) -O3 $(CFLAGS) par_sort.cpp
	$(CC) $(PYTHONFLAGS) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) GPU.cu
	$(CC) $(PYTHONFLAGS) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) kernel.cu
	$(CC) $(PYTHONFLAGS) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) main.cu
	$(CC) $(PYTHONFLAGS) $(FLAGS)  par_sort.o GPU.o kernel.o main.o -o libgpuknnselfjoin.so	

#End make python shared library
#####################################

clean:
	rm $(OBJECTS)
	rm $(CUDAOBJECTS)
	rm libgpuselfjoin.so
	rm main