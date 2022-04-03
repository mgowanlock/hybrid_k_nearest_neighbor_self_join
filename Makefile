#Important:
#see the params.h for some of the parameters


SOURCES = import_dataset.cpp par_sort.cpp main.cu GPU.cu kernel.cu 
OBJECTS = par_sort.o import_dataset.o 
CUDAOBJECTS = GPU.o kernel.o main.o 
CC = nvcc
GNUCC = g++
EXECUTABLE = main

#Change to your CUDA Compute Capability:
COMPCAPABIL = 60

FLAGS = -std=c++14 -O3 -Xcompiler -fopenmp -arch=compute_$(COMPCAPABIL) -code=sm_$(COMPCAPABIL) -lcuda -lineinfo 
CFLAGS = -c




all: $(EXECUTABLE)

#Compile the parallel sort function separately with g++ because
#it doesn't compile with nvcc
par_sort.o: par_sort.cpp
	$(GNUCC) -O3 $(CFLAGS) par_sort.cpp

import_dataset.o: import_dataset.cpp params.h
	$(CC) $(FLAGS) $(CFLAGS) $<

main.o: main.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) main.cu 

kernel.o: kernel.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) kernel.cu 		

GPU.o: GPU.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) GPU.cu


$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@


clean:
	rm $(OBJECTS)
	rm $(CUDAOBJECTS)
	rm main

