all: projectest

projectest: gpusolve.o creatematrixarm.o
	nvcc  -o projectest gpusolve.o creatematrixarm.o -L/usr/lib/x86_64-linux-gnu  -lcudart  -lcusparse -lcusolver -llapack -lopenblas
    
gpusolve.o:
	nvcc -c -I/usr/include gpusolve.cu -lcusolver 

creatematrixarm.o:
	g++ creatematrixarm.cpp -c -O2 -I/home/jbrzensk/USERS/include -DARMA_DONT_USE_WRAPPER -lopenblas -llapack 


clean:
	rm -f *.o projectest

