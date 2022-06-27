
//precompute direct neighbors with the GPU:
#include <cuda_runtime.h>
#include <cuda.h>
#include "structs.h"
#include <stdio.h>
#include "kernel.h"
#include <math.h>
#include "GPU.h"
#include <algorithm>
#include "omp.h"
#include <queue>
#include <unistd.h>
#include <set>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h> //for streams for thrust (added with Thrust v1.8)



//for warming up GPU:
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

//Concurrent vector for cleaning up memory from the nieghbortable batches
// #include "tbb/concurrent_vector.h"




//FOR THE BATCHED EXECUTION:
//#define BATCHTOTALELEM 1200000000 //THE TOTAL SIZE ALLOCATED ON THE HOST
//THE NUMBER OF BATCHES AND THE SIZE OF THE BUFFER FOR EACH KERNEL EXECUTION ARE NOT RELATED TO THE TOTAL NUMBER
//OF ELEMENTS (ABOVE).
// #define NUMBATCHES 20
// #define BATCHBUFFERELEM 100000000 //THE SMALLER SIZE ALLOCATED ON THE DEVICE FOR EACH KERNEL EXECUTION 




//Array used to store pointers to allocated pinned memory
//If one iteration allocated the maximum amount of pinned memory, we want to keep it and not free it.
//So that the next iteration can use it 


int * pointIDKeySaved[GPUSTREAMS]; //key
int * pointInDistValueSaved[GPUSTREAMS]; //value
DTYPE * distancesKeyValueSaved[GPUSTREAMS]; //distance between the key point and value for kNN
bool pinnedSavedFlag=0; // flag set to true if pinned memory has been previously allocated


//use these to find the unique memory addresses to free of the neighborTable
std::set<int *> ValsPtr;
std::set<DTYPE *> DistancePtr;
// tbb::concurrent_vector<int *>ValsPtrVect;
// tbb::concurrent_vector<double *>DistancePtrVect;

using namespace std;


//sort ascending
bool compareByPointValue(const key_val_sort &a, const key_val_sort &b)
{
    return a.value_at_dim < b.value_at_dim;
}

//sort ascending
bool compareByFirstDataDim(const databaseSortMap &a, const databaseSortMap &b)
{
    return a.data[0] < b.data[0];
}



//in_epsilon- the value for the guess of epsilon
//out_epsilon- the value of epsilon computed to be used
//dev_epsilon pointer to allocated epsilon for executing the estimator kernel
//fracDB- fraction of the total points going to be processed
unsigned long long callGPUBatchEst(unsigned int * DBSIZE, unsigned int N_QueryPts, unsigned int * dev_queryPts, unsigned int k_Neighbors, DTYPE* dev_database,
  DTYPE* in_epsilon, DTYPE* dev_epsilon, struct grid * dev_grid, 
	unsigned int * dev_indexLookupArr, struct gridCellLookup * dev_gridCellLookupArr, DTYPE* dev_minArr, 
	unsigned int * dev_nCells, unsigned int * dev_nNonEmptyCells, unsigned int * retNumBatches, unsigned int * retGPUBufferSize)
{



	//CUDA error code:
	cudaError_t errCode;

	printf("\n\n***********************************\nEstimating Batches:");
	cout<<"\n** BATCH ESTIMATOR: Sometimes the GPU will error on a previous execution and you won't know. \nLast error start of function: "<<cudaGetLastError();
	
	


//////////////////////////////////////////////////////////
	//ESTIMATE THE BUFFER SIZE AND NUMBER OF BATCHES ETC BY COUNTING THE NUMBER OF RESULTS
	//TAKE A SAMPLE OF THE DATA POINTS, NOT ALL OF THEM
	//Use sampleRate for this
	/////////////////////////////////////////////////////////

	
	// printf("\nDon't estimate: calculate the entire thing (for testing)");
	//Parameters for the batch size estimation.


	//original:
	/*		
	double sampleRate=0.05; //sample 1% of the points in the dataset sampleRate=0.01. 
						              //Sample the entire dataset(no sampling) sampleRate=1
	int offsetRate=1.0/sampleRate;
	printf("\nOffset: %d", offsetRate);
	*/

	//because we use a query set with fewer points at each iteration, we need to sample more
	//at each iteration. Thus, the number of points we sample is 4% of the entire dataset, but then sample that number of query points
	
	//double sampleRate=(0.05*(*DBSIZE))/(N_QueryPts*1.0); //was testing this
	
	double sampleRate=0.05; 
	int offsetRate=1.0/sampleRate;
	offsetRate=max(offsetRate,1);

	printf("\nSample rate: %f, offset: %d",sampleRate, offsetRate);


	/////////////////
	//N GPU threads
	////////////////

	
	

	unsigned int * dev_N_batchEst; 
	dev_N_batchEst=(unsigned int*)malloc(sizeof(unsigned int));

	unsigned int * N_batchEst; 
	N_batchEst=(unsigned int*)malloc(sizeof(unsigned int));
	// *N_batchEst=*DBSIZE*sampleRate;
	*N_batchEst=ceil(1.0*N_QueryPts*sampleRate);


	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N_batchEst, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_N_batchEst Got error with code " << errCode << endl; 
	}	

	//copy N to device 
	//N IS THE NUMBER OF THREADS
	errCode=cudaMemcpy( dev_N_batchEst, N_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: N batchEST Got error with code " << errCode << endl; 
	}


	/////////////
	//count the result set size 
	////////////

	unsigned int * dev_cnt_batchEst; 
	dev_cnt_batchEst=(unsigned int*)malloc(sizeof(unsigned int));

	unsigned int * cnt_batchEst; 
	cnt_batchEst=(unsigned int*)malloc(sizeof(unsigned int));
	*cnt_batchEst=0;


	//allocate on the device
	errCode=cudaMalloc((void**)&dev_cnt_batchEst, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl; 
	}	

	//copy cnt to device 
	errCode=cudaMemcpy( dev_cnt_batchEst, cnt_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl; 
	}


	//////////////////////////////////
	//Copy the "threads per distance calculation" to the GPU 
	/////////////////////////////////
	unsigned int * threadsForDistanceCalc=(unsigned int *)malloc(sizeof(unsigned int));
	#if THREADMULTI==0
	*threadsForDistanceCalc=1;
	#endif

	#if THREADMULTI>1 
	*threadsForDistanceCalc=THREADMULTI;
	#endif

	#if THREADMULTI==-1
	*threadsForDistanceCalc=ceil((DYNAMICTHRESHOLD*1.0)/(*N_batchEst));
	
	//make sure that the number of threads per point doesn't exceed the maximum allowable (to prevent 1 thread with thousands of threads)	
	*threadsForDistanceCalc=min(MAXTHREADSPERPOINT,*threadsForDistanceCalc);
	printf("\n[THREADMULTI==-1 (Dynamic)] Batch Estimator: Total Query Points: %u, Num for batch est: %u, Threads Per Point: %u",N_QueryPts, *N_batchEst, *threadsForDistanceCalc);
	#endif

	//static
	#if THREADMULTI==-2
	*threadsForDistanceCalc=STATICTHREADSPERPOINT;
	#endif




	printf("\nThreads for distance calculations (Batch Estimator): %u",*threadsForDistanceCalc);

	unsigned int * dev_threadsForDistanceCalc;

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_threadsForDistanceCalc, sizeof(unsigned int));		
	if(errCode != cudaSuccess) {
	cout << "\nError: threadsForDistanceCalc alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy threads for distance calculation to the device
	errCode=cudaMemcpy(dev_threadsForDistanceCalc, threadsForDistanceCalc, sizeof(unsigned int), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: threadsForDistanceCalc Got error with code " << errCode << endl; 
	}


	//////////////////////////////////
	//End Copy the query points to the GPU
	/////////////////////////////////
	
	//////////////////
	//SAMPLE OFFSET - TO SAMPLE THE DATA TO ESTIMATE THE TOTAL NUMBER OF KEY VALUE PAIRS
	/////////////////

	//offset into the database when batching the results
	unsigned int * sampleOffset; 
	sampleOffset=(unsigned int*)malloc(sizeof(unsigned int));
	*sampleOffset=offsetRate;


	unsigned int * dev_sampleOffset; 
	dev_sampleOffset=(unsigned int*)malloc(sizeof(unsigned int));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_sampleOffset, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: sample offset Got error with code " << errCode << endl; 
	}

	//copy offset to device 
	errCode=cudaMemcpy( dev_sampleOffset, sampleOffset, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_sampleOffset Got error with code " << errCode << endl; 
	}


	////////////////////////////////////
	//TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	//debug values
	unsigned int * dev_debug1; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;

	unsigned int * dev_debug2; 
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;

	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;



	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 alloc -- error with code " << errCode << endl; 
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 alloc -- error with code " << errCode << endl; 
	}		

	//copy debug to device
	errCode=cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug1 copy to device -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug2 copy to device -- with code " << errCode << endl; 
	}




	////////////////////////////////////
	//END TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////	


			#if THREADMULTI==0
			const int TOTALBLOCKSBATCHEST=ceil(*N_batchEst/(1.0*BLOCKSIZE));
			printf("\ntotal blocks: %d",TOTALBLOCKSBATCHEST);
			#endif

			#if THREADMULTI>=2 
			// const int TOTALBLOCKSBATCHEST=ceil((1.0*(N_QueryPts)*(THREADMULTI*1.0)*sampleRate)/(1.0*BLOCKSIZE));


			/*
			unsigned int queryPts=ceil(((1.0*N_QueryPts)*sampleRate));
			const int TOTALBLOCKSBATCHEST=ceil((queryPts*(THREADMULTI*1.0))/(1.0*BLOCKSIZE)); // need to compute the total queryPts 
																							//first or will round down
			printf("\nTotal threads: %u",TOTALBLOCKSBATCHEST*BLOCKSIZE);
			printf("\nTotal query points: %u",queryPts);
			*/


			//original
			// const int TOTALBLOCKSBATCHEST=ceil((1.0*(N_QueryPts)*sampleRate)/(1.0*BLOCKSIZE))*THREADMULTI;

			
			unsigned int tmpBLOCKSBATCHEST=0;

			//weirdness with all of the integer arithmetic, need two cases -- otherwise we will get slightly different results
			//when estimating batches (not that it matters since its just an estimate anyway)
			if (ceil(*N_batchEst/(1.0*BLOCKSIZE))>1)
			{
				tmpBLOCKSBATCHEST=ceil(*N_batchEst/(1.0*BLOCKSIZE))*THREADMULTI;
			}
			else
			{
				printf("\nIn the single block");
				tmpBLOCKSBATCHEST=ceil((*N_batchEst*THREADMULTI*1.0)/(1.0*BLOCKSIZE));
			}

			 const int TOTALBLOCKSBATCHEST=tmpBLOCKSBATCHEST;

			

			printf("\ntotal blocks (THREADMULTI==%d): %d",THREADMULTI, TOTALBLOCKSBATCHEST);
			#endif


			//dynamic (-1) and static (-2)
			#if THREADMULTI==-1 || THREADMULTI==-2

			unsigned int tmpBLOCKSBATCHEST=0;

			//weirdness with all of the integer arithmetic, need two cases -- otherwise we will get slightly different results
			//when estimating batches (not that it matters since its just an estimate anyway)
			if (ceil(*N_batchEst/(1.0*BLOCKSIZE))>1)
			{
				tmpBLOCKSBATCHEST=ceil(*N_batchEst/(1.0*BLOCKSIZE))*(*threadsForDistanceCalc);
			}
			else
			{
				printf("\nIn the single block");
				tmpBLOCKSBATCHEST=ceil((*N_batchEst*(*threadsForDistanceCalc)*1.0)/(1.0*BLOCKSIZE));
			}

			 const int TOTALBLOCKSBATCHEST=tmpBLOCKSBATCHEST;
			 
			 #if THREADMULTI==-1
			 printf("\ntotal blocks (Dynamic THREADMULTI==%d): %d",THREADMULTI, TOTALBLOCKSBATCHEST);
			 #endif

			 #if THREADMULTI==-2
			 printf("\ntotal blocks (Static THREADMULTI==%d): %d",THREADMULTI, TOTALBLOCKSBATCHEST);
			 #endif

			#endif
			

			//static
			// #if THREADMULTI==-2

			// unsigned int tmpBLOCKSBATCHEST=0;

			// //weirdness with all of the integer arithmetic, need two cases -- otherwise we will get slightly different results
			// //when estimating batches (not that it matters since its just an estimate anyway)
			// if (ceil(*N_batchEst/(1.0*BLOCKSIZE))>1)
			// {
			// 	tmpBLOCKSBATCHEST=ceil(*N_batchEst/(1.0*BLOCKSIZE))*(*threadsForDistanceCalc);
			// }
			// else
			// {
			// 	printf("\nIn the single block");
			// 	tmpBLOCKSBATCHEST=ceil((*N_batchEst*(*threadsForDistanceCalc)*1.0)/(1.0*BLOCKSIZE));
			// }

			//  const int TOTALBLOCKSBATCHEST=tmpBLOCKSBATCHEST;
			//  printf("\ntotal blocks (Static THREADMULTI==%d): %d",THREADMULTI, TOTALBLOCKSBATCHEST);

			// #endif

			//original:
	// const int TOTALBLOCKSBATCHEST=ceil((1.0*(N_QueryPts)*sampleRate)/(1.0*BLOCKSIZE));	
	// printf("\ntotal blocks: %d",TOTALBLOCKSBATCHEST);

	// __global__ void kernelNDGridIndexBatchEstimator(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
	// unsigned int * sampleOffset, double * database, double *epsilon, struct grid * index, unsigned int * indexLookupArr, 
	// struct gridCellLookup * gridCellLookupArr, double * minArr, unsigned int * nCells, unsigned int * cnt, 
	// unsigned int * nNonEmptyCells)

	kernelNDGridIndexBatchEstimatorQuerySet<<< TOTALBLOCKSBATCHEST, BLOCKSIZE>>>(dev_debug1, dev_debug2, dev_threadsForDistanceCalc, dev_queryPts, dev_N_batchEst, 
		dev_sampleOffset, dev_database, dev_epsilon, dev_grid, dev_indexLookupArr, 
		dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_cnt_batchEst, dev_nNonEmptyCells);
		// kernelNDGridIndexBatchEstimatorWithExpansion<<< TOTALBLOCKSBATCHEST, BLOCKSIZE>>>(dev_debug1, dev_debug2, dev_queryPts, dev_iteration, dev_N_batchEst, 
		// dev_sampleOffset, dev_database, dev_epsilon, dev_grid, dev_indexLookupArr, 
		// dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_cnt_batchEst, dev_nNonEmptyCells, dev_gridCellNDMask, 
		// dev_gridCellNDMaskOffsets);
	
		
		cout<<"\n** ERROR FROM KERNEL LAUNCH OF BATCH ESTIMATOR: "<<cudaGetLastError();
		// find the size of the number of results
		errCode=cudaMemcpy( cnt_batchEst, dev_cnt_batchEst, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if(errCode != cudaSuccess) {
		cout << "\nError: getting cnt for batch estimate from GPU Got error with code " << errCode << endl; 
		}
		else
		{
			printf("\nGPU: result set size for estimating the number of batches (sampled): %u",*cnt_batchEst);
		}

	

	// uint64_t estimatedNeighbors=(uint64_t)*cnt_batchEst*(uint64_t)offsetRate*fracDB;			
	uint64_t estimatedNeighbors=(uint64_t)*cnt_batchEst*(uint64_t)offsetRate;		
	printf("\nFrom gpu cnt: %u, offset rate: %d", *cnt_batchEst,offsetRate); 	
	// printf("\nNum Query Points: %u, Fraction of DB: %f Estimated Neighbors: %lu, ", N_QueryPts, N_QueryPts*1.0/(*DBSIZE), estimatedNeighbors); 	
	uint64_t maxNeighborsPossible=N_QueryPts*1.0*(*DBSIZE);	
	

	// double factorIncrease=5.0; //If we assume one expansion on average. In 2-D, this would be 4x
								
	// estimatedNeighbors=estimatedNeighbors*factorIncrease;
	// printf("\nBatch estimator assumes that the returned points are found within epsilon without expansion, new est. neighbors: %lu",estimatedNeighbors);
	printf("\nEstimated neighbors: %lu",estimatedNeighbors);
	
	

	//the maximum number of points a query point can find is the entire dataset
	//check to make sure the estimate is not bigger than this number, 
	//if so, select the smaller number of neighbors (numQueryPts*|D|)
	//Important when the query set is small and the statistics don't work well
	estimatedNeighbors=min(maxNeighborsPossible,estimatedNeighbors);
	printf("\nNum Query Points: %u, Fraction of DB: %f Estimated Neighbors: %lu, Max result size (|D|*numQueryPts): %lu", N_QueryPts, N_QueryPts*1.0/(*DBSIZE), estimatedNeighbors, maxNeighborsPossible); 			


	//since we increase the number of cells searched at each iteration
	//need to increase by the number of adjacent cells, since the batch estimator only does 
	//the one layer of adjacent cells
	// unsigned int numAdjCellsInitial=pow(3,NUMINDEXEDDIM);	
	// unsigned int numAdjCellsWithIter=pow(3+(2*iter),NUMINDEXEDDIM);
	// printf("\nNum Adj Cells initial: %u, Num cells with Iter: %u",numAdjCellsInitial,numAdjCellsWithIter);
	// double ratioCells=(numAdjCellsWithIter*1.0)/(numAdjCellsInitial*1.0);
	// uint64_t estimatedNeighbors=(uint64_t)*cnt_batchEst*(uint64_t)offsetRate*fracDB*ratioCells;	
	// printf("\nFrom gpu cnt: %d, offset rate: %d", *cnt_batchEst,offsetRate); 	
	// printf("\nEps: %f, Fraction of DB processed: %f, Ratio initial adj. cells to adj cells at iteration %d: %f, Estimated neighbors: %lu",*in_epsilon, fracDB, iter, ratioCells, estimatedNeighbors);






	if (estimatedNeighbors<1000000)
	{
		printf("\nEstimated neighbors too small, increasing");
		estimatedNeighbors=1000000;
	}





	//initial
	
	// #ifndef GPUBUFFERSMALLER
	// unsigned int GPUBufferSize=100000000; //normal size in SC paper (high-D)
	// #else
	// unsigned int GPUBufferSize=50000000; //size in HPBDC paper (low-D)
	// #endif

	// unsigned int GPUBufferSize=100000000; //normal size in SC paper (high-D)	
	// unsigned int GPUBufferSize=50000000; //size in HPBDC paper (low-D)
	unsigned int GPUBufferSize=GPUBUFFERSIZE;

	double alpha=0.1; //overestimation factor -- for kNN needs to be higher than self-join, because we 
						//select high density regions for the GPU

	
	uint64_t estimatedTotalSizeWithAlpha=estimatedNeighbors*(1.0+alpha);
	printf("\nEstimated total result set size: %lu", estimatedNeighbors);
	printf("\nEstimated total result set size (with Alpha %f): %lu", alpha,estimatedTotalSizeWithAlpha);	
	


	if (estimatedTotalSizeWithAlpha<(GPUBufferSize*GPUSTREAMS)){
		printf("\nSmall buffer size, increasing alpha to: %f",alpha*3.0); //was *3.0 
		
			
		GPUBufferSize=estimatedTotalSizeWithAlpha*(1.0+(alpha*2.0))/(GPUSTREAMS);		//we do 2*alpha for small datasets because the
																		//sampling will be worse for small datasets
																		//but we fix the 3 streams still (thats why divide by 3).			
		
		//if the modified buffer size is greater than the input buffer size, then
		//we just set it to be the input size
		if (GPUBufferSize>GPUBUFFERSIZE)
		{
			GPUBufferSize=GPUBUFFERSIZE;
		}
		
	}

	unsigned int numBatches=ceil(((1.0+alpha)*estimatedNeighbors*1.0)/((uint64_t)GPUBufferSize*1.0));
	printf("\nNumber of batches: %d, buffer size: %d", numBatches, GPUBufferSize);

	*retNumBatches=numBatches;
	*retGPUBufferSize=GPUBufferSize;
		

	printf("\nEnd Batch Estimator\n***********************************\n");




	cudaFree(dev_cnt_batchEst);	
	cudaFree(dev_N_batchEst);
	cudaFree(dev_sampleOffset);
	// cudaFree(dev_threadsForDistanceCalc);







return estimatedTotalSizeWithAlpha;

}





//similar to the original, except that we give it a list of query points to process
//the kernel takes this list of query points
//Can probably use this for the original execution too, bu where we give the kernel all of the points to process
//Iter is the number of times the function was called -- controls the number of cells searched around the query point
void distanceTableNDGridBatcheskNN(std::vector<std::vector<DTYPE> > * NDdataPoints, int * nearestNeighborTable, 
	DTYPE * nearestNeighborTableDistances,  double * totaldistance, 
	std::vector<unsigned int> *queryPtsVect, DTYPE* epsilon,  unsigned int k_neighbors, struct grid * index, 
	struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, DTYPE* minArr, unsigned int * nCells, 
	unsigned int * indexLookupArr, struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors, 
	uint64_t * totalNeighbors, CTYPE* workCounts)
{

	


	double tKernelResultsStart=omp_get_wtime();

	//CUDA error code:
	cudaError_t errCode;


	cout<<"\n** Sometimes the GPU will error on a previous execution and you won't know. \nLast error start of function: "<<cudaGetLastError();


	///////////////////////////////////
	//COPY THE kNN TO THE GPU
	///////////////////////////////////

	unsigned int * dev_k_neighbors;
	
		
	
	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_k_neighbors, sizeof(unsigned int));		
	if(errCode != cudaSuccess) {
	cout << "\nError: k_neighbors alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy to the device
	errCode=cudaMemcpy(dev_k_neighbors, &k_neighbors, sizeof(unsigned int), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: k_neighbirs Got error with code " << errCode << endl; 
	}	
		



	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	unsigned int * DBSIZE;
	DBSIZE=(unsigned int*)malloc(sizeof(unsigned int));
	*DBSIZE=NDdataPoints->size();
	
	printf("\nIn main GPU method: DBSIZE is: %u",*DBSIZE);cout.flush();

	
	DTYPE* database= (DTYPE*)malloc(sizeof(DTYPE)*(*DBSIZE)*(GPUNUMDIM));  
	DTYPE* dev_database;
	// DTYPE* dev_database= (DTYPE*)malloc(sizeof(DTYPE)*(*DBSIZE)*(GPUNUMDIM));  
	
		
	
	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE));		
	if(errCode != cudaSuccess) {
	cout << "\nError: database alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	
	


	//copy the database from the ND vector to the array:
	for (int i=0; i<(*DBSIZE); i++){
		std::copy((*NDdataPoints)[i].begin(), (*NDdataPoints)[i].end(), database+(i*(GPUNUMDIM)));
	}


		//copy database to the device
	errCode=cudaMemcpy(dev_database, database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database2 Got error with code " << errCode << endl; 
	}




	//printf("\n size of database: %d",N);



	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	//////////////////////////////////
	//Copy the query points to the GPU
	/////////////////////////////////
	unsigned int * QUERYSIZE;
	QUERYSIZE=(unsigned int*)malloc(sizeof(unsigned int));
	*QUERYSIZE=queryPtsVect->size();






	printf("\nIn subsequent execution. Num. query points is: %u",*QUERYSIZE);cout.flush();

	unsigned int * queryPts= (unsigned int *)malloc(sizeof(unsigned int )*(*QUERYSIZE));  
	// unsigned int * dev_queryPts= (unsigned int *)malloc(sizeof(unsigned int )*(*QUERYSIZE));  
	unsigned int * dev_queryPts;

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_queryPts, sizeof(unsigned int)*(*QUERYSIZE));		
	if(errCode != cudaSuccess) {
	cout << "\nError: queryPts alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy the query point ids from the vector to the array:
	std::copy((*queryPtsVect).begin(), (*queryPtsVect).end(), queryPts);
	


	//copy database to the device
	errCode=cudaMemcpy(dev_queryPts, queryPts, sizeof(unsigned int)*(*QUERYSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: queryPts Got error with code " << errCode << endl; 
	}


	// for (int i=0; i<*QUERYSIZE; i++)
	// {
	// 	printf("\nQuery point: %u",queryPts[i]);
	// }


	//number of query points
	unsigned int * dev_sizequerypts;
	// dev_sizequerypts=(unsigned int*)malloc(sizeof(unsigned int));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_sizequerypts, sizeof(unsigned int));		
	if(errCode != cudaSuccess) {
	cout << "\nError: size query pts alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy the number of query points to the devid
	errCode=cudaMemcpy(dev_sizequerypts, QUERYSIZE, sizeof(unsigned int), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: size query pts got error with code " << errCode << endl; 
	}


	//////////////////////////////////
	//END Copy the query points to the GPU
	/////////////////////////////////



	///////////////////////////////////
	//COPY THE INDEX TO THE GPU
	///////////////////////////////////

	


	

	struct grid * dev_grid;
	// dev_grid=(struct grid*)malloc(sizeof(struct grid)*(*nNonEmptyCells));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_grid, sizeof(struct grid)*(*nNonEmptyCells));	
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index -- error with code " << errCode << endl; cout.flush(); 
	}


	//copy grid index to the device:
	errCode=cudaMemcpy(dev_grid, index, sizeof(struct grid)*(*nNonEmptyCells), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index copy to device -- error with code " << errCode << endl; 
	}	

	printf("\nSize of index sent to GPU (MiB): %f", (DTYPE)sizeof(struct grid)*(*nNonEmptyCells)/(1024.0*1024.0));


	///////////////////////////////////
	//END COPY THE INDEX TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY THE LOOKUP ARRAY TO THE DATA ELEMS TO THE GPU
	///////////////////////////////////

	#if SORT==1
	printf("\nSORTING ALL DIMENSIONS FOR VARIANCE NOW");
	printf("\nSORTIDU USES THE FIRST UNINXEDED DIMENSION");
	
	double tstartsort=omp_get_wtime();
	
	int sortDim=0;
	struct key_val_sort tmp;
	std::vector<struct key_val_sort> tmp_to_sort;
	unsigned int totalLength=0;

	if(GPUNUMDIM > NUMINDEXEDDIM)
		sortDim = NUMINDEXEDDIM;

	for (int i=0; i<(*nNonEmptyCells); i++)
	// for (int i=0; i<1; i++)
	{
		// if(index[i].indexmin < index[i].indexmax){
			// printf("Size cell: %d, %d\n", i,(index[i].indexmax-index[i].indexmin)+1);
			for (int j=0; j<(index[i].indexmax-index[i].indexmin)+1; j++)
			{
			unsigned int idx=index[i].indexmin+j;	
			tmp.pid=indexLookupArr[idx];
			tmp.value_at_dim=database[indexLookupArr[idx]*GPUNUMDIM+sortDim];
			tmp_to_sort.push_back(tmp);
			}

			
			totalLength+=tmp_to_sort.size();
			std::sort(tmp_to_sort.begin(),tmp_to_sort.end(),compareByPointValue); 

			//copy the sorted elements into the lookup array
			for (int x=0; x<tmp_to_sort.size(); x++)
			{
				indexLookupArr[index[i].indexmin+x]=tmp_to_sort[x].pid;
			}

			tmp_to_sort.clear();	
	}	

	double tendsort=omp_get_wtime();
	printf("\nSORT cells time (on host): %f", tendsort - tstartsort);

	// printf("\nTotal length: %u",totalLength);
	#endif
	

	// return;






	unsigned int * dev_indexLookupArr;
	// dev_indexLookupArr=(unsigned int*)malloc(sizeof(unsigned int)*(*DBSIZE));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_indexLookupArr, sizeof(unsigned int)*(*DBSIZE));
	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array allocation -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy lookup array to the device:
	errCode=cudaMemcpy(dev_indexLookupArr, indexLookupArr, sizeof(unsigned int)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: copy lookup array to device -- error with code " << errCode << endl; 
	}	

	

	///////////////////////////////////
	//END COPY THE LOOKUP ARRAY TO THE DATA ELEMS TO THE GPU
	///////////////////////////////////



	///////////////////////////////////
	//COPY THE GRID CELL LOOKUP ARRAY 
	///////////////////////////////////

	
	
						
	struct gridCellLookup * dev_gridCellLookupArr;
	// dev_gridCellLookupArr=(struct gridCellLookup*)malloc(sizeof(struct gridCellLookup)*(*nNonEmptyCells));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_gridCellLookupArr, sizeof(struct gridCellLookup)*(*nNonEmptyCells));
	if(errCode != cudaSuccess) {
	cout << "\nError: copy grid cell lookup array allocation -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy lookup array to the device:
	errCode=cudaMemcpy(dev_gridCellLookupArr, gridCellLookupArr, sizeof(struct gridCellLookup)*(*nNonEmptyCells), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: copy grid cell lookup array to device -- error with code " << errCode << endl; 
	}	

	

	///////////////////////////////////
	//END COPY THE GRID CELL LOOKUP ARRAY 
	///////////////////////////////////





	
	///////////////////////////////////
	//COPY GRID DIMENSIONS TO THE GPU
	//THIS INCLUDES THE NUMBER OF CELLS IN EACH DIMENSION, 
	//AND THE STARTING POINT OF THE GRID IN THE DIMENSIONS 
	///////////////////////////////////

	//minimum boundary of the grid:
	DTYPE* dev_minArr;
	// dev_minArr=(DTYPE*)malloc(sizeof(DTYPE)*(NUMINDEXEDDIM));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_minArr, sizeof(DTYPE)*(NUMINDEXEDDIM));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc minArr -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_minArr, minArr, sizeof(DTYPE)*(NUMINDEXEDDIM), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: Copy minArr to device -- error with code " << errCode << endl; 
	}	


	//number of cells in each dimension
	unsigned int * dev_nCells;
	// dev_nCells=(unsigned int*)malloc(sizeof(unsigned int)*(NUMINDEXEDDIM));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_nCells, sizeof(unsigned int)*(NUMINDEXEDDIM));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc nCells -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_nCells, nCells, sizeof(unsigned int)*(NUMINDEXEDDIM), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: Copy nCells to device -- error with code " << errCode << endl; 
	}	


	///////////////////////////////////
	//END COPY GRID DIMENSIONS TO THE GPU
	///////////////////////////////////




	///////////////////////////////////
	//COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
	///////////////////////////////////

	//total size of the result set as it's batched
	//this isnt sent to the GPU
	unsigned int * totalResultSetCnt;
	totalResultSetCnt=(unsigned int*)malloc(sizeof(unsigned int));
	*totalResultSetCnt=0;

	//count values - for an individual kernel launch
	//need different count values for each stream
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*cnt=0;

	unsigned int * dev_cnt; 
	// dev_cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	// *dev_cnt=0;

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_cnt, sizeof(unsigned int)*GPUSTREAMS);	
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc cnt -- error with code " << errCode << endl; 
	}

	///////////////////////////////////
	//END COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
	///////////////////////////////////
	
	
	//////////////////////////////////
	//Copy the "threads per distance calculation" to the GPU 
	/////////////////////////////////

	unsigned int * threadsForDistanceCalc=(unsigned int *)malloc(sizeof(unsigned int));
	#if THREADMULTI==0
	*threadsForDistanceCalc=1;
	#endif


	//static
	#if THREADMULTI==-2
	*threadsForDistanceCalc=STATICTHREADSPERPOINT;
	#endif

	

	unsigned int * dev_threadsForDistanceCalc;

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_threadsForDistanceCalc, sizeof(unsigned int));		
	if(errCode != cudaSuccess) {
	cout << "\nError: threadsForDistanceCalc alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy threads to the device
	errCode=cudaMemcpy(dev_threadsForDistanceCalc, threadsForDistanceCalc, sizeof(unsigned int), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: threadsForDistanceCalc Got error with code " << errCode << endl; 
	}


	//////////////////////////////////
	//End Copy the query points to the GPU
	/////////////////////////////////


	

	///////////////////////////////////
	//EPSILON
	///////////////////////////////////
	DTYPE* dev_epsilon;
	// dev_epsilon=(DTYPE*)malloc(sizeof( DTYPE));
	

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(DTYPE));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc epsilon -- error with code " << errCode << endl; 
	}

	//copy to device
	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(DTYPE), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon copy to device -- error with code " << errCode << endl; 
	}		



	///////////////////////////////////
	//END EPSILON
	///////////////////////////////////


	///////////////////////////////////
	//NUMBER OF NON-EMPTY CELLS
	///////////////////////////////////
	unsigned int * dev_nNonEmptyCells;
	// dev_nNonEmptyCells=(unsigned int*)malloc(sizeof( unsigned int ));
	


	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_nNonEmptyCells, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc nNonEmptyCells -- error with code " << errCode << endl; 
	}

	//copy to device
	errCode=cudaMemcpy( dev_nNonEmptyCells, nNonEmptyCells, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: nNonEmptyCells copy to device -- error with code " << errCode << endl; 
	}		

	///////////////////////////////////
	//NUMBER OF NON-EMPTY CELLS
	///////////////////////////////////
	

	////////////////////////////////////
	//NUMBER OF THREADS PER GPU STREAM
	////////////////////////////////////

	//THE NUMBER OF THREADS THAT ARE LAUNCHED IN A SINGLE KERNEL INVOCATION
	//CAN BE FEWER THAN THE NUMBER OF ELEMENTS IN THE DATABASE IF MORE THAN 1 BATCH
	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	
	unsigned int * dev_N; 
	// dev_N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc dev_N -- error with code " << errCode << endl; 
	}	

	////////////////////////////////////
	//NUMBER OF THREADS PER GPU STREAM
	////////////////////////////////////


	////////////////////////////////////
	//OFFSET INTO THE DATABASE FOR BATCHING THE RESULTS
	//BATCH NUMBER 
	////////////////////////////////////
	unsigned int * batchOffset; 
	batchOffset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	
	unsigned int * dev_offset; 
	// dev_offset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_offset, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc offset -- error with code " << errCode << endl; 
	}

	//Batch number to calculate the point to process (in conjunction with the offset)
	//offset into the database when batching the results
	unsigned int * batchNumber; 
	batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	unsigned int * dev_batchNumber; 
	// dev_batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_batchNumber, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc batch number -- error with code " << errCode << endl; 
	}

	////////////////////////////////////
	//END OFFSET INTO THE DATABASE FOR BATCHING THE RESULTS
	//BATCH NUMBER
	////////////////////////////////////

				
	
	


	unsigned long long estimatedNeighbors=0;	
	unsigned int numBatches=0;
	unsigned int GPUBufferSize=0;


	


	double tstartbatchest=omp_get_wtime();
	//here out_epsilon is ignored, since we have already computed this
	// DTYPE unused_epsilon;
	//we compute the fraction of the total dataset processed
	//so that we don't overallocate the number of batches
	// double fracDB=(*QUERYSIZE*1.0)/(*DBSIZE*1.0);
	estimatedNeighbors=callGPUBatchEst(DBSIZE, *QUERYSIZE, dev_queryPts, k_neighbors, dev_database, epsilon, dev_epsilon, dev_grid, dev_indexLookupArr,dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_nNonEmptyCells, &numBatches, &GPUBufferSize);	
	double tendbatchest=omp_get_wtime();
	printf("\nTime to estimate batches: %f",tendbatchest - tstartbatchest);
	printf("\nIn Calling fn: Estimated neighbors: %llu, num. batches: %d, GPU Buffer size: %d",estimatedNeighbors, numBatches,GPUBufferSize);
	
	// printf("\n\n*******\nSETTING 100 BATCHES\n********\n\n");
	// numBatches=100;


	// printf("\nReturning after estimate batches");
	// return ;	


	/////////////////////////////////////////////////////////	
	//END BATCH ESTIMATOR	
	/////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////	
	//Dynamically select the number of threads per point for distance calculations
	/////////////////////////////////////////////////////////	

	//dynamic
	#if THREADMULTI==-1
	unsigned int pointsPerBatch=ceil((*QUERYSIZE*1.0)/(numBatches*1.0));
	printf("\n[THREADMULTI==-1 (Dynamic)] *QUERYSIZE: %u, numBatches: %u", *QUERYSIZE, numBatches);
	*threadsForDistanceCalc=ceil((DYNAMICTHRESHOLD*1.0)/(pointsPerBatch*1.0));
	
	//make sure that the number of threads per point doesn't exceed the maximum allowable (to prevent 1 thread with thousands of threads)	
	*threadsForDistanceCalc=min(MAXTHREADSPERPOINT,*threadsForDistanceCalc);
	printf("\n[THREADMULTI==-1 (Dynamic)] Points per batch: %u, Threads Per Point: %u",pointsPerBatch, *threadsForDistanceCalc);



	//copy threads to the device

	errCode=cudaMemcpy(dev_threadsForDistanceCalc, threadsForDistanceCalc, sizeof(unsigned int), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: threadsForDistanceCalc Got error with code " << errCode << endl; 
	}

	#endif
	

	/////////////////////////////////////////////////////////	
	//End Dynamically select the number of threads per point for distance calculations
	/////////////////////////////////////////////////////////	
	
	//this prints for all threads per point schemes
	//put here once instead of each time
	printf("\nThreads for distance calculations: %u", *threadsForDistanceCalc);

	////////////////////////////////////
	//TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	//debug values
	unsigned int * dev_debug1; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;

	unsigned int * dev_debug2; 
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;

	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;



	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 alloc -- error with code " << errCode << endl; 
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 alloc -- error with code " << errCode << endl; 
	}		

	//copy debug to device
	errCode=cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug1 copy to device -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug2 copy to device -- with code " << errCode << endl; 
	}


	////////////////////////////////////
	//END TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	

	///////////////////
	//ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////

	//THE NUMBER OF POINTERS IS EQUAL TO THE NUMBER OF BATCHES
	for (int i=0; i<numBatches; i++){
		int *ptr=NULL;
		struct neighborDataPtrs tmpStruct;
		tmpStruct.dataPtr=ptr;
		tmpStruct.sizeOfDataArr=0;
		
		pointersToNeighbors->push_back(tmpStruct);
	}

	///////////////////
	//END ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////



	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET USING THE BATCH ESTIMATOR
	///////////////////////////////////
	

	//NEED BUFFERS ON THE GPU AND THE HOST FOR THE NUMBER OF CONCURRENT STREAMS	
	//GPU BUFFER ON THE DEVICE
	//BUFFER ON THE HOST WITH PINNED MEMORY FOR FAST MEMCPY
	//BUFFER ON THE HOST TO DUMP THE RESULTS OF BATCHES SO THAT GPU THREADS CAN CONTINUE
	//EXECUTING STREAMS ON THE HOST



	//GPU MEMORY ALLOCATION: key/value pairs

	int * dev_pointIDKey[GPUSTREAMS]; //key
	int * dev_pointInDistValue[GPUSTREAMS]; //value
	DTYPE * dev_distancesKeyValue[GPUSTREAMS]; //distance between the key point and value point for kNN
	

	

	for (int i=0; i<GPUSTREAMS; i++)
	{

		//if we have allocated the maximum buffer size for pinned memory allocation, we need to make sure
		//that we create the same size buffer on the GPU
		unsigned long int bufferSizeDevice=GPUBufferSize;
		if (pinnedSavedFlag==1)
		{
			bufferSizeDevice=GPUBUFFERSIZE;
		}
		errCode=cudaMalloc((void **)&dev_pointIDKey[i], sizeof(int)*bufferSizeDevice);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}

		errCode=cudaMalloc((void **)&dev_pointInDistValue[i], sizeof(int)*bufferSizeDevice);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}

		errCode=cudaMalloc((void **)&dev_distancesKeyValue[i], sizeof(double)*bufferSizeDevice);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}

	}


	

	//HOST RESULT ALLOCATION FOR THE GPU TO COPY THE DATA INTO A PINNED MEMORY ALLOCATION
	//ON THE HOST
	//pinned result set memory for the host
	//the number of elements are recorded for that batch in resultElemCountPerBatch
	//NEED PINNED MEMORY ALSO BECAUSE YOU NEED IT TO USE STREAMS IN THRUST FOR THE MEMCOPY OF THE SORTED RESULTS	
	//can't do async copies without pinned memory

	//Two cases: 1) the maximum GPUBUFFERSIZE will be allocated for the first time (then we save pointers to the pinned memory)
	// 2) if the maximum amount of pinned memory GPUBUFFERSIZE has already been allocated then we don't allocate again
	//instead we just update pointers

	//PINNED MEMORY TO COPY FROM THE GPU	
	int * pointIDKey[GPUSTREAMS]; //key
	int * pointInDistValue[GPUSTREAMS]; //value
	DTYPE * distancesKeyValue[GPUSTREAMS]; //distance between the key point and value for kNN




	double tstartpinnedresults=omp_get_wtime();

	//if allocating max size for the first time
	if (GPUBufferSize==GPUBUFFERSIZE && pinnedSavedFlag==0)
	{	
		printf("\nPinned memory: allocating and saving pointers");
		//allocate
		for (int i=0; i<GPUSTREAMS; i++)
		{
		cudaMallocHost((void **) &pointIDKey[i], sizeof(int)*GPUBufferSize);
		cudaMallocHost((void **) &pointInDistValue[i], sizeof(int)*GPUBufferSize);
		cudaMallocHost((void **) &distancesKeyValue[i], sizeof(double)*GPUBufferSize);
		}

		//update pointers
		for (int i=0; i<GPUSTREAMS; i++)
		{
		pointIDKeySaved[i]=pointIDKey[i]; 
		pointInDistValueSaved[i]=pointInDistValue[i];
		distancesKeyValueSaved[i]=distancesKeyValue[i];
		}

		//update flag
		pinnedSavedFlag=1;
	}
	//if pinned memory has already been allocated
	else if (pinnedSavedFlag==1)
	{
		printf("\nPinned memory: using preexisting memory, setting pointers");
		//set pointers
		for (int i=0; i<GPUSTREAMS; i++)
		{
		pointIDKey[i]=pointIDKeySaved[i]; 
		pointInDistValue[i]=pointInDistValueSaved[i];
		distancesKeyValue[i]=distancesKeyValueSaved[i];
		}
	}
	//allocate as normal (for smaller pinned memory sizes that we don't save)
	else
	{
		printf("\nPinned memory small: allocating only");
		//allocate
		for (int i=0; i<GPUSTREAMS; i++)
		{
		cudaMallocHost((void **) &pointIDKey[i], sizeof(int)*GPUBufferSize);
		cudaMallocHost((void **) &pointInDistValue[i], sizeof(int)*GPUBufferSize);
		cudaMallocHost((void **) &distancesKeyValue[i], sizeof(double)*GPUBufferSize);
		}
	}


	double tendpinnedresults=omp_get_wtime();
	printf("\nTime to allocate pinned memory for results: %f", tendpinnedresults - tstartpinnedresults);
	

	// cudaMalloc((void **) &pointIDKey, sizeof(int)*GPUBufferSize*NUMBATCHES);
	// cudaMalloc((void **) &pointInDistValue, sizeof(int)*GPUBufferSize*NUMBATCHES);




	printf("\nmemory requested for results ON GPU (GiB): %f",(double)(sizeof(int)*3*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));
	printf("\nmemory requested for results in MAIN MEMORY (GiB): %f",(double)(sizeof(int)*3*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));

	
	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////











	



	/////////////////////////////////
	//SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////

	omp_set_num_threads(GPUSTREAMS);
	

	/////////////////////////////////
	//END SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////
	
	

	/////////////////////////////////
	//CREATE STREAMS
	////////////////////////////////

	cudaStream_t stream[GPUSTREAMS];
	
	for (int i=0; i<GPUSTREAMS; i++){
	cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
	}	

	/////////////////////////////////
	//END CREATE STREAMS
	////////////////////////////////
	
	

	///////////////////////////////////
	//LAUNCH KERNEL IN BATCHES
	///////////////////////////////////
		
	//since we use the strided scheme, some of the batch sizes
	//are off by 1 of each other, a first group of batches will
	//have 1 extra data point to process, and we calculate which batch numbers will 
	//have that.  The batchSize is the lower value (+1 is added to the first ones)

	CTYPE* dev_workCounts;
	cudaMalloc((void **)&dev_workCounts, sizeof(CTYPE)*2);
#if COUNTMETRICS == 1
        cudaMemcpy(dev_workCounts, workCounts, 2*sizeof(CTYPE), cudaMemcpyHostToDevice );
#endif

	unsigned int batchSize=(*QUERYSIZE)/numBatches;
	unsigned int batchesThatHaveOneMore=(*QUERYSIZE)-(batchSize*numBatches); //batch number 0- < this value have one more
	printf("\nBatches that have one more GPU thread: %u batchSize(N): %u, \n",batchesThatHaveOneMore,batchSize);

	uint64_t totalResultsLoop=0;


		
		
		
		//FOR LOOP OVER THE NUMBER OF BATCHES STARTS HERE
		//i=0...numBatches
		#pragma omp parallel for schedule(static,1) reduction(+:totalResultsLoop) num_threads(GPUSTREAMS)
		for (int i=0; i<numBatches; i++)
		// for (int i=0; i<1; i++)
		{	
			

			int tid=omp_get_thread_num();
			
			printf("\ntid: %d, starting iteration: %d",tid,i);

			//N NOW BECOMES THE NUMBER OF POINTS TO PROCESS PER BATCH
			//AS ONE GPU THREAD PROCESSES A SINGLE POINT
			


			
			if (i<batchesThatHaveOneMore)
			{
				N[tid]=batchSize+1;	
				printf("\nN (GPU threads): %d, tid: %d",N[tid], tid);
			}
			else
			{
				N[tid]=batchSize;	
				printf("\nN (1 less): %d tid: %d",N[tid], tid);
			}

			//if the number of points is 0, exit
			//can happen if only a few points for kNN
			if (N[tid]==0)
			{	
				continue;
			}

			//set relevant parameters for the batched execution that get reset
			
			//copy N to device 
			//N IS THE NUMBER OF THREADS
			errCode=cudaMemcpyAsync( &dev_N[tid], &N[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: N Got error with code " << errCode << endl; 
			}

			//the batched result set size (reset to 0):
			cnt[tid]=0;
			errCode=cudaMemcpyAsync( &dev_cnt[tid], &cnt[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
			}

			//the offset for batching, which keeps track of where to start processing at each batch
			batchOffset[tid]=numBatches; //for the strided
			errCode=cudaMemcpyAsync( &dev_offset[tid], &batchOffset[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_offset memcpy Got error with code " << errCode << endl; 
			}

			//the batch number for batching with strided
			batchNumber[tid]=i;
			errCode=cudaMemcpyAsync( &dev_batchNumber[tid], &batchNumber[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_batchNumber memcpy Got error with code " << errCode << endl; 
			}

			#if THREADMULTI==0
			const int TOTALBLOCKS=ceil((1.0*(N[tid]))/(1.0*BLOCKSIZE));	
			printf("\ntotal blocks: %d",TOTALBLOCKS);
			#endif

			#if THREADMULTI>=2
			const int TOTALBLOCKS=ceil((1.0*(N[tid])*(THREADMULTI*1.0)*(*iter+1.0))/(1.0*BLOCKSIZE));	
			

			printf("\ntotal blocks (THREADMULTI==%d, iteration: %u for more distance calculation threads per query point per iteration): %d",THREADMULTI, *iter, TOTALBLOCKS);
			#endif

			//number of threads per point
			//dynamic (-1) or static (-2)
			#if THREADMULTI==-1 || THREADMULTI==-2
			const int TOTALBLOCKS=ceil((1.0*(N[tid])*(*threadsForDistanceCalc))/(1.0*BLOCKSIZE));	
			
				#if THREADMULTI==-1
				printf("\ntotal blocks (THREADMULTI DYNAMIC, ThreadsPerPoint: %u, for more distance calculation threads per query point per iteration): %d",*threadsForDistanceCalc, TOTALBLOCKS);
				#endif

				#if THREADMULTI==-2
				printf("\ntotal blocks (THREADMULTI STATIC, ThreadsPerPoint: %u, for more distance calculation threads per query point per iteration): %d",*threadsForDistanceCalc, TOTALBLOCKS);
				#endif

			#endif

			//execute kernel -- normal kernel that gets executed using an index	
			//0 is shared memory pool
			kernelNDGridIndexGlobalkNN<<< TOTALBLOCKS, BLOCKSIZE, 0, stream[tid]>>>(dev_debug1, dev_debug2, dev_k_neighbors, &dev_N[tid], 
			&dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_epsilon, dev_grid, dev_indexLookupArr, 
			dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_pointIDKey[tid], dev_pointInDistValue[tid], dev_distancesKeyValue[tid], dev_queryPts, dev_threadsForDistanceCalc, dev_workCounts);


			// errCode=cudaDeviceSynchronize();
			// cout <<"\n\nError from device synchronize: "<<errCode;

			cout <<"\n\nKERNEL LAUNCH RETURN: "<<cudaGetLastError()<<endl<<endl;
			if ( cudaSuccess != cudaGetLastError() ){
		    	cout <<"\n\nERROR IN KERNEL LAUNCH. ERROR: "<<cudaSuccess<<endl<<endl;
		    }

		    

		   
			// find the size of the number of results
			errCode=cudaMemcpyAsync( &cnt[tid], &dev_cnt[tid], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
			}
			else{
				printf("\nGPU: result set size within epsilon (GPU grid): %d",cnt[tid]);cout.flush();
				fprintf(stderr,"\nGPU: result set size within epsilon (GPU grid): %d",cnt[tid]);
			}


			// optional debug values
			/*
				errCode=cudaMemcpyAsync( debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] );
				if(errCode != cudaSuccess) {
				cout << "\nError: getting debug from GPU Got error with code " << errCode << endl; 
				}
				else{
					printf("\nNum query points that have at least 5 points in the origin cell: %u",*debug1);cout.flush();
					
				}

				errCode=cudaMemcpyAsync( debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] );
				if(errCode != cudaSuccess) {
				cout << "\nError: getting debug from GPU Got error with code " << errCode << endl; 
				}
				else{
					printf("\nNum query points that do not have at least 5 points in the origin cell: %u",*debug2);cout.flush();
					
				}
			*/

			
			////////////////////////////////////
			//SORT THE TABLE DATA ON THE GPU
			//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
			////////////////////////////////////

			/////////////////////////////
			//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
			//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
			//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
			/////////////////////////////

			//sort by key with the data already on the device:
			//wrap raw pointer with a device_ptr to use with Thrust functions
			thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey[tid]);
			thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue[tid]);
			thrust::device_ptr<DTYPE> dev_distancesKeyValPtr(dev_distancesKeyValue[tid]); // dist for knn

			// ZipIterator iterBegin(thrust::make_tuple(dev_data_ptr, dev_distancesKeyValPtr));  

			// thrust::make_zip_iterator(thrust::make_tuple(dev_pointInDistValue[tid], dev_distancesKeyValue[tid]));
			// thrust::make_zip_iterator(thrust::make_tuple(dev_pointInDistValue[tid]+ cnt[tid], dev_distancesKeyValue[tid]+ cnt[tid]));


			//XXXXXXXXXXXXXXXX
			//THRUST USING STREAMS REQUIRES THRUST V1.8 
			//XXXXXXXXXXXXXXXX
			
			//original first step
			

			
			
			//k-NN
			//Need the 3 arrays: keys, values, distances
			//back-to-back sort - keys: query points, values: points wihtin the distance of the keys
			//first sort the key/values by the distance
			//then sort the distances/values by the keys
			//uses the tuple to move the values in the arrays around when sorting
			try{
			
				//first sort by the distance array (not the key)
				thrust::stable_sort_by_key(thrust::cuda::par.on(stream[tid]), dev_distancesKeyValPtr, dev_distancesKeyValPtr + cnt[tid], 
					thrust::make_zip_iterator(thrust::make_tuple(dev_keys_ptr, dev_data_ptr)));

				//next sort by the key array 
				thrust::stable_sort_by_key(thrust::cuda::par.on(stream[tid]), dev_keys_ptr, dev_keys_ptr + cnt[tid], 
					thrust::make_zip_iterator(thrust::make_tuple(dev_data_ptr, dev_distancesKeyValPtr)));

			}
			// catch(std::bad_alloc &e)
			//   {
			//     std::cerr << "Ran out of memory while sorting, " << GPUBufferSize << std::endl;
			//     exit(-1);
			//   }
			catch(thrust::system_error e)
  			{
    		std::cerr << "Error inside sort: " << e.what() << std::endl;
    		exit(-1);
  			}



  			//Here we test only copying back the elements that have at least k+1 elements
  			//Steps:
  			//1) count unique keys
  			//2) use thrust count to get the frequency of each
  			//3) make a list of all of the keys that have at least k neighbors 
  			//3) copy using thrust set_intersection_by_key 

  			//1) count unique keys
  	// 		thrust::device_vector<int> values(cnt[tid]);
			// thrust::unique_copy( dev_keys_ptr, dev_keys_ptr+cnt[tid], values.begin() );
  	// 		thrust::device_vector<int> d_unique_keys(N[tid]);
  	// 		thrust::reduce_by_key(dev_keys_ptr, dev_keys_ptr+cnt[tid], thrust::constant_iterator<int>(1), dev_keys_ptr, d_unique_keys.begin());

  	// 		for (int x=0; x<cnt[tid]; x++)
  	// 		{
			// printf("\nUnique keys: %d, ",d_unique_keys[x]);
			// }


			//original
	  		//thrust with streams into individual buffers for each batch
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey[tid]), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue[tid]), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);	
			cudaMemcpyAsync(thrust::raw_pointer_cast(distancesKeyValue[tid]), thrust::raw_pointer_cast(dev_distancesKeyValPtr), cnt[tid]*sizeof(double), cudaMemcpyDeviceToHost, stream[tid]);	

			//need to make sure the data is copied before constructing portion of the neighbor table
			cudaStreamSynchronize(stream[tid]);

			




			
			
			//////////////////////////////////////////////
			//original table construction
				
			/*	
			double tableconstuctstart=omp_get_wtime();
			//set the number of neighbors in the pointer struct:
			(*pointersToNeighbors)[i].sizeOfDataArr=cnt[tid];    
			(*pointersToNeighbors)[i].dataPtr=new int[cnt[tid]]; 
			(*pointersToNeighbors)[i].distancePtr=new DTYPE[cnt[tid]];
			
			ValsPtr.insert((*pointersToNeighbors)[i].dataPtr); //used to free memory later
			DistancePtr.insert((*pointersToNeighbors)[i].distancePtr); //used to free memory later
			
			

			constructNeighborTableKeyValueWithPtrskNN(pointIDKey[tid], pointInDistValue[tid], distancesKeyValue[tid], neighborTable, (*pointersToNeighbors)[i].dataPtr, (*pointersToNeighbors)[i].distancePtr, &cnt[tid]);

			
			

			double tableconstuctend=omp_get_wtime();	
			
			printf("\nTable construct time: %f", tableconstuctend - tableconstuctstart);
			*/
			//////////////////////////////////////////////
			//end original table construction
			//////////////////////////////////////////////
			


			//directly store the kNN from the pinned memory buffers
			
			
			double totaldisttmp=0;
			double knnStoreStart=omp_get_wtime();
			storeNeighborTableForkNNOnTheFly(pointIDKey[tid], pointInDistValue[tid], distancesKeyValue[tid], 
				&cnt[tid], NDdataPoints, k_neighbors, queryPtsVect, nearestNeighborTable, nearestNeighborTableDistances, &totaldisttmp);

			#pragma omp critical
			*totaldistance+=totaldisttmp;

			double knnStoreEnd=omp_get_wtime();
			printf("\nkNN store time: %f", knnStoreEnd - knnStoreStart);

			//add the batched result set size to the total count
			totalResultsLoop+=cnt[tid];

			printf("\nRunning total of total size of result array, tid: %d: %lu", tid, totalResultsLoop);
			
			
	
			


		

		} //END LOOP OVER THE GPU BATCHES

		// cudaDeviceSynchronize();

		


#if COUNTMETRICS == 1
        cudaMemcpy(workCounts, dev_workCounts, 2*sizeof(CTYPE), cudaMemcpyDeviceToHost );
#endif

	
	
	printf("\nTOTAL RESULT SET SIZE ON HOST:  %lu", totalResultsLoop);
	*totalNeighbors=totalResultsLoop;



	double tKernelResultsEnd=omp_get_wtime();
	
	printf("\nTime to launch kernel and execute everything (get results etc.) except freeing memory: %f",tKernelResultsEnd-tKernelResultsStart);


	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////



	///////////////////////////////////	
	//OPTIONAL DEBUG VALUES
	///////////////////////////////////
	
	// double tStartdebug=omp_get_wtime();

	// errCode=cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	// }
	// else
	// {
	// 	printf("\nDebug1 value: %u",*debug1);
	// }

	// errCode=cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	// }
	// else
	// {
	// 	printf("\nDebug2 value: %u",*debug2);
	// }	

	// double tEnddebug=omp_get_wtime();
	// printf("\nTime to retrieve debug values: %f", tEnddebug - tStartdebug);
	

	///////////////////////////////////	
	//END OPTIONAL DEBUG VALUES
	///////////////////////////////////
	

	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
	// if (NUM_TRIALS>1)
	// {

	double tFreeStart=omp_get_wtime();

	for (int i=0; i<GPUSTREAMS; i++){
		errCode=cudaStreamDestroy(stream[i]);
		if(errCode != cudaSuccess) {
		cout << "\nError: destroying stream" << errCode << endl; 
		}
	}



	//free on the heap
	free(DBSIZE);
	free(database);
	free(queryPts);
	free(totalResultSetCnt);
	free(cnt);
	free(threadsForDistanceCalc);
	free(N);
	free(batchOffset);
	free(batchNumber);
	free(debug1);
	free(debug2);



	//free the data on the device
	// cudaFree(dev_pointIDKey);
	// cudaFree(dev_pointInDistValue);
	// cudaFree(dev_distancesKeyValue);
	

	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	cudaFree(dev_epsilon);
	cudaFree(dev_grid);
	cudaFree(dev_gridCellLookupArr);
	cudaFree(dev_indexLookupArr);
	cudaFree(dev_minArr);
	cudaFree(dev_nCells);
	cudaFree(dev_nNonEmptyCells);
	cudaFree(dev_N); 	
	cudaFree(dev_cnt); 
	cudaFree(dev_offset); 
	cudaFree(dev_batchNumber); 
	cudaFree(dev_threadsForDistanceCalc);
	cudaFree(dev_queryPts);
	cudaFree(dev_workCounts);
	
	//free data related to the individual streams for each batch
	for (int i=0; i<GPUSTREAMS; i++){
		//free the data on the device
		cudaFree(dev_pointIDKey[i]);
		cudaFree(dev_pointInDistValue[i]);
		cudaFree(dev_distancesKeyValue[i]);


		//free on the host
		if (pinnedSavedFlag==0)
		{
		cudaFreeHost(pointIDKey[i]);
		cudaFreeHost(pointInDistValue[i]);
		cudaFreeHost(distancesKeyValue[i]);
		}

	}

	
	

	double tFreeEnd=omp_get_wtime();

	printf("\nTime freeing memory: %f", tFreeEnd - tFreeStart);
	cout<<"\n** last error at end of fn batches (could be from freeing memory): "<<cudaGetLastError();

}








//similar to the original, except that we give it a list of query points to process
//the kernel takes this list of query points

//This is used when the index no longer provides any selectivity
//Don't call the batch estimator anymore, and do a brute force comparison
//Return all of the neighbors of each point
void distanceTableNDBruteForce(std::vector<std::vector<DTYPE> > * NDdataPoints, int * nearestNeighborTable, DTYPE * nearestNeighborTableDistances,
	std::vector<unsigned int> *queryPtsVect, DTYPE* epsilon,  unsigned int k_neighbors, 
	uint64_t * totalNeighbors, struct neighborTableLookup * neighborTable)
{

	

	double tKernelResultsStart=omp_get_wtime();

	//CUDA error code:
	cudaError_t errCode;


	cout<<"\n** Sometimes the GPU will error on a previous execution and you won't know. \nLast error start of function: "<<cudaGetLastError();


	
		



	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	unsigned int * DBSIZE;
	DBSIZE=(unsigned int*)malloc(sizeof(unsigned int));
	*DBSIZE=NDdataPoints->size();
	
	printf("\nIn main GPU method: DBSIZE is: %u",*DBSIZE);cout.flush();

	
	DTYPE* database= (DTYPE*)malloc(sizeof(DTYPE)*(*DBSIZE)*(GPUNUMDIM));  
	DTYPE* dev_database= (DTYPE*)malloc(sizeof(DTYPE)*(*DBSIZE)*(GPUNUMDIM));  
	
		
	
	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE));		
	if(errCode != cudaSuccess) {
	cout << "\nError: database alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	
	


	//copy the database from the ND vector to the array:
	for (int i=0; i<(*DBSIZE); i++){
		std::copy((*NDdataPoints)[i].begin(), (*NDdataPoints)[i].end(), database+(i*(GPUNUMDIM)));
	}


	//copy database to the device
	errCode=cudaMemcpy(dev_database, database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database2 Got error with code " << errCode << endl; 
	}


	//size of the database
	unsigned int * dev_DBSIZE;
	errCode=cudaMalloc( (void**)&dev_DBSIZE, sizeof(unsigned int));		
	if(errCode != cudaSuccess) {
	cout << "\nError: database size -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy database to the device
	errCode=cudaMemcpy(dev_DBSIZE, DBSIZE, sizeof(unsigned int), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database size Got error with code " << errCode << endl; 
	}

	//printf("\n size of database: %d",N);



	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	//////////////////////////////////
	//Copy the query points to the GPU
	/////////////////////////////////
	unsigned int * QUERYSIZE;
	QUERYSIZE=(unsigned int*)malloc(sizeof(unsigned int));
	*QUERYSIZE=queryPtsVect->size();






	printf("\nIn subsequent execution. Num. query points is: %u",*QUERYSIZE);cout.flush();

	unsigned int * queryPts= (unsigned int *)malloc(sizeof(unsigned int )*(*QUERYSIZE));  
	// unsigned int * dev_queryPts= (unsigned int *)malloc(sizeof(unsigned int )*(*QUERYSIZE));  
	unsigned int * dev_queryPts;

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_queryPts, sizeof(unsigned int)*(*QUERYSIZE));		
	if(errCode != cudaSuccess) {
	cout << "\nError: queryPts alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy the query point ids from the vector to the array:
	std::copy((*queryPtsVect).begin(), (*queryPtsVect).end(), queryPts);
	

	//copy database to the device
	errCode=cudaMemcpy(dev_queryPts, queryPts, sizeof(unsigned int)*(*QUERYSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: queryPts Got error with code " << errCode << endl; 
	}


	


	//number of query points
	unsigned int * dev_sizequerypts;
	// dev_sizequerypts=(unsigned int*)malloc(sizeof(unsigned int));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_sizequerypts, sizeof(unsigned int));		
	if(errCode != cudaSuccess) {
	cout << "\nError: size query pts alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy the number of query points to the devid
	errCode=cudaMemcpy(dev_sizequerypts, QUERYSIZE, sizeof(unsigned int), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: size query pts got error with code " << errCode << endl; 
	}


	//////////////////////////////////
	//END Copy the query points to the GPU
	/////////////////////////////////



	///////////////////////////////////
	//COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
	///////////////////////////////////

	//total size of the result set as it's batched
	//this isnt sent to the GPU
	unsigned int * totalResultSetCnt;
	totalResultSetCnt=(unsigned int*)malloc(sizeof(unsigned int));
	*totalResultSetCnt=0;

	//count values - for an individual kernel launch
	//need different count values for each stream
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*cnt=0;

	unsigned int * dev_cnt; 
	dev_cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*dev_cnt=0;

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_cnt, sizeof(unsigned int)*GPUSTREAMS);	
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc cnt -- error with code " << errCode << endl; 
	}

	///////////////////////////////////
	//END COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
	///////////////////////////////////
	
	
	//////////////////////////////////
	//Copy the "threads per distance calculation" to the GPU 
	/////////////////////////////////

	
	// unsigned int * threadsForDistanceCalc=(unsigned int *)malloc(sizeof(unsigned int));
	// #if THREADMULTI==0
	// *threadsForDistanceCalc=1;
	// #endif

	// #if THREADMULTI>1
	// *threadsForDistanceCalc=(*iter+1)*THREADMULTI;
	// #endif

	// //static
	// #if THREADMULTI==-2
	// *threadsForDistanceCalc=STATICTHREADSPERPOINT;
	// #endif

	//for the brute force, since we don't know the number of query points
	//we use the dynamic number of threads
	//See below after we compute the number of batches


	
	

	/////////////////////////////////////////////////////////	
	//End Dynamically select the number of threads per point for distance calculations
	/////////////////////////////////////////////////////////	
	

	

	




	

	///////////////////////////////////
	//EPSILON
	///////////////////////////////////
	// DTYPE* dev_epsilon;
	// dev_epsilon=(DTYPE*)malloc(sizeof( DTYPE));
	

	// //Allocate on the device
	// errCode=cudaMalloc((void**)&dev_epsilon, sizeof(DTYPE));
	// if(errCode != cudaSuccess) {
	// cout << "\nError: Alloc epsilon -- error with code " << errCode << endl; 
	// }

	// //copy to device
	// errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(DTYPE), cudaMemcpyHostToDevice );
	// if(errCode != cudaSuccess) {
	// cout << "\nError: epsilon copy to device -- error with code " << errCode << endl; 
	// }		



	///////////////////////////////////
	//END EPSILON
	///////////////////////////////////




	////////////////////////////////////
	//NUMBER OF THREADS PER GPU STREAM
	////////////////////////////////////

	//THE NUMBER OF THREADS THAT ARE LAUNCHED IN A SINGLE KERNEL INVOCATION
	//CAN BE FEWER THAN THE NUMBER OF ELEMENTS IN THE DATABASE IF MORE THAN 1 BATCH
	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	
	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc dev_N -- error with code " << errCode << endl; 
	}	

	////////////////////////////////////
	//NUMBER OF THREADS PER GPU STREAM
	////////////////////////////////////


	////////////////////////////////////
	//OFFSET INTO THE DATABASE FOR BATCHING THE RESULTS
	//BATCH NUMBER 
	////////////////////////////////////
	unsigned int * batchOffset; 
	batchOffset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	
	unsigned int * dev_offset; 
	dev_offset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_offset, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc offset -- error with code " << errCode << endl; 
	}

	//Batch number to calculate the point to process (in conjunction with the offset)
	//offset into the database when batching the results
	unsigned int * batchNumber; 
	batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	unsigned int * dev_batchNumber; 
	dev_batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_batchNumber, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc batch number -- error with code " << errCode << endl; 
	}

	////////////////////////////////////
	//END OFFSET INTO THE DATABASE FOR BATCHING THE RESULTS
	//BATCH NUMBER
	////////////////////////////////////

	//RE-WRITE BATCH ESTIMATOR HERE
	unsigned long long estimatedNeighbors=0;
	estimatedNeighbors=((*QUERYSIZE*1.0)*(*DBSIZE*1.0));
	
	printf("\n***Not estimating batches using the batch estimator kernel -- Brute Force");

	
	unsigned int GPUBufferSize=GPUBUFFERSIZE;		
	//because some batches may get more points than another batch, need to overestimate (*1.2)
	unsigned int numBatches=ceil((estimatedNeighbors*1.2)/(GPUBufferSize*1.0));
	
	printf("\nNumber of batches brute force: %u, GPUBufferSize: %u",numBatches, GPUBufferSize);
	

	


	/////////////////////////////////////////////////////////	
	//END BATCH ESTIMATOR	
	/////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////
	//Dynamically set the number of threads per point
	///////////////////////////////////////////////////////


	//dynamic
	unsigned int * threadsForDistanceCalc=(unsigned int *)malloc(sizeof(unsigned int));
	unsigned int pointsPerBatch=ceil((*QUERYSIZE*1.0)/(numBatches*1.0));
	printf("\n[THREADMULTI==-1 (Dynamic, Brute Force)] *QUERYSIZE: %u, numBatches: %u", *QUERYSIZE, numBatches);
	*threadsForDistanceCalc=ceil((DYNAMICTHRESHOLD*1.0)/(pointsPerBatch*1.0));
	
	//make sure that the number of threads per point doesn't exceed the maximum allowable (to prevent 1 thread with thousands of threads)	
	// *threadsForDistanceCalc=min(MAXTHREADSPERPOINT,*threadsForDistanceCalc);
	printf("\n[THREADMULTI==-1 (Dynamic, Brute Force)] Points per batch: %u, Threads Per Point: %u",pointsPerBatch, *threadsForDistanceCalc);



	//copy threads to the device
	unsigned int * dev_threadsForDistanceCalc;

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_threadsForDistanceCalc, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: threadsForDistanceCalc -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy(dev_threadsForDistanceCalc, threadsForDistanceCalc, sizeof(unsigned int), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: threadsForDistanceCalc Got error with code " << errCode << endl; 
	}



	////////////////////////////////////
	//TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	//debug values
	unsigned int * dev_debug1; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;

	unsigned int * dev_debug2; 
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;

	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;



	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 alloc -- error with code " << errCode << endl; 
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 alloc -- error with code " << errCode << endl; 
	}		

	//copy debug to device
	errCode=cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug1 copy to device -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug2 copy to device -- with code " << errCode << endl; 
	}


	////////////////////////////////////
	//END TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	

	///////////////////
	//ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////

	//THE NUMBER OF POINTERS IS EQUAL TO THE NUMBER OF BATCHES
	/*
	for (int i=0; i<numBatches; i++){
		int *ptr=NULL;
		struct neighborDataPtrs tmpStruct;
		tmpStruct.dataPtr=ptr;
		tmpStruct.sizeOfDataArr=0;
		
		pointersToNeighbors->push_back(tmpStruct);
	}*/

	///////////////////
	//END ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////



	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET USING THE BATCH ESTIMATOR
	///////////////////////////////////
	

	//NEED BUFFERS ON THE GPU AND THE HOST FOR THE NUMBER OF CONCURRENT STREAMS	
	//GPU BUFFER ON THE DEVICE
	//BUFFER ON THE HOST WITH PINNED MEMORY FOR FAST MEMCPY
	//BUFFER ON THE HOST TO DUMP THE RESULTS OF BATCHES SO THAT GPU THREADS CAN CONTINUE
	//EXECUTING STREAMS ON THE HOST



	//GPU MEMORY ALLOCATION: key/value pairs

	int * dev_pointIDKey[GPUSTREAMS]; //key
	int * dev_pointInDistValue[GPUSTREAMS]; //value
	DTYPE * dev_distancesKeyValue[GPUSTREAMS]; //distance between the key point and value point for kNN
	

	

	for (int i=0; i<GPUSTREAMS; i++)
	{

		//if we have allocated the maximum buffer size for pinned memory allocation, we need to make sure
		//that we create the same size buffer on the GPU

		unsigned long int bufferSizeDevice=GPUBufferSize;
		if (pinnedSavedFlag==1)
		{
			bufferSizeDevice=GPUBUFFERSIZE;
		}
		printf("\nBuffer size device: %lu",bufferSizeDevice);
		errCode=cudaMalloc((void **)&dev_pointIDKey[i], sizeof(int)*bufferSizeDevice);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}

		errCode=cudaMalloc((void **)&dev_pointInDistValue[i], sizeof(int)*bufferSizeDevice);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}

		errCode=cudaMalloc((void **)&dev_distancesKeyValue[i], sizeof(double)*bufferSizeDevice);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}

	}


	

	//HOST RESULT ALLOCATION FOR THE GPU TO COPY THE DATA INTO A PINNED MEMORY ALLOCATION
	//ON THE HOST
	//pinned result set memory for the host
	//the number of elements are recorded for that batch in resultElemCountPerBatch
	//NEED PINNED MEMORY ALSO BECAUSE YOU NEED IT TO USE STREAMS IN THRUST FOR THE MEMCOPY OF THE SORTED RESULTS	
	//can't do async copies without pinned memory

	//Two cases: 1) the maximum GPUBUFFERSIZE will be allocated for the first time (then we save pointers to the pinned memory)
	// 2) if the maximum amount of pinned memory GPUBUFFERSIZE has already been allocated then we don't allocate again
	//instead we just update pointers

	//PINNED MEMORY TO COPY FROM THE GPU	
	int * pointIDKey[GPUSTREAMS]; //key
	int * pointInDistValue[GPUSTREAMS]; //value
	DTYPE * distancesKeyValue[GPUSTREAMS]; //distance between the key point and value for kNN




	double tstartpinnedresults=omp_get_wtime();

	//if allocating max size for the first time
	if (GPUBufferSize==GPUBUFFERSIZE && pinnedSavedFlag==0)
	{	
		printf("\nPinned memory: allocating and saving pointers");
		//allocate
		for (int i=0; i<GPUSTREAMS; i++)
		{
		cudaMallocHost((void **) &pointIDKey[i], sizeof(int)*GPUBufferSize);
		cudaMallocHost((void **) &pointInDistValue[i], sizeof(int)*GPUBufferSize);
		cudaMallocHost((void **) &distancesKeyValue[i], sizeof(double)*GPUBufferSize);
		}

		//update pointers
		for (int i=0; i<GPUSTREAMS; i++)
		{
		pointIDKeySaved[i]=pointIDKey[i]; 
		pointInDistValueSaved[i]=pointInDistValue[i];
		distancesKeyValueSaved[i]=distancesKeyValue[i];
		}

		//update flag
		pinnedSavedFlag=1;
	}
	//if pinned memory has already been allocated
	else if (pinnedSavedFlag==1)
	{
		printf("\nPinned memory: using preexisting memory, setting pointers");
		//set pointers
		for (int i=0; i<GPUSTREAMS; i++)
		{
		pointIDKey[i]=pointIDKeySaved[i]; 
		pointInDistValue[i]=pointInDistValueSaved[i];
		distancesKeyValue[i]=distancesKeyValueSaved[i];
		}
	}
	//allocate as normal (for smaller pinned memory sizes that we don't save)
	else
	{
		printf("\nPinned memory small: allocating only");
		//allocate
		for (int i=0; i<GPUSTREAMS; i++)
		{
		cudaMallocHost((void **) &pointIDKey[i], sizeof(int)*GPUBufferSize);
		cudaMallocHost((void **) &pointInDistValue[i], sizeof(int)*GPUBufferSize);
		cudaMallocHost((void **) &distancesKeyValue[i], sizeof(double)*GPUBufferSize);
		}
	}


	double tendpinnedresults=omp_get_wtime();
	printf("\nTime to allocate pinned memory for results: %f", tendpinnedresults - tstartpinnedresults);
	

	// cudaMalloc((void **) &pointIDKey, sizeof(int)*GPUBufferSize*NUMBATCHES);
	// cudaMalloc((void **) &pointInDistValue, sizeof(int)*GPUBufferSize*NUMBATCHES);




	printf("\nmemory requested for results ON GPU (GiB): %f",(double)(sizeof(int)*3*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));
	printf("\nmemory requested for results in MAIN MEMORY (GiB): %f",(double)(sizeof(int)*3*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));

	
	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////











	



	/////////////////////////////////
	//SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////

	omp_set_num_threads(GPUSTREAMS);
	

	/////////////////////////////////
	//END SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////
	
	

	/////////////////////////////////
	//CREATE STREAMS
	////////////////////////////////

	cudaStream_t stream[GPUSTREAMS];
	
	for (int i=0; i<GPUSTREAMS; i++){
	cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
	}	

	/////////////////////////////////
	//END CREATE STREAMS
	////////////////////////////////
	
	

	///////////////////////////////////
	//LAUNCH KERNEL IN BATCHES
	///////////////////////////////////
		
	//since we use the strided scheme, some of the batch sizes
	//are off by 1 of each other, a first group of batches will
	//have 1 extra data point to process, and we calculate which batch numbers will 
	//have that.  The batchSize is the lower value (+1 is added to the first ones)

	CTYPE* dev_workCounts;
	cudaMalloc((void **)&dev_workCounts, sizeof(CTYPE)*2);
#if COUNTMETRICS == 1
        cudaMemcpy(dev_workCounts, workCounts, 2*sizeof(CTYPE), cudaMemcpyHostToDevice );
#endif

	unsigned int batchSize=(*QUERYSIZE)/numBatches;
	unsigned int batchesThatHaveOneMore=(*QUERYSIZE)-(batchSize*numBatches); //batch number 0- < this value have one more
	printf("\nBatches that have one more GPU thread: %u batchSize(N): %u, \n",batchesThatHaveOneMore,batchSize);

	uint64_t totalResultsLoop=0;


		
		
		//FOR LOOP OVER THE NUMBER OF BATCHES STARTS HERE
		//i=0...numBatches
		#pragma omp parallel for schedule(static,1) reduction(+:totalResultsLoop) num_threads(GPUSTREAMS)
		for (int i=0; i<numBatches; i++)
		{	
			

			int tid=omp_get_thread_num();
			
			printf("\ntid: %d, starting iteration: %d",tid,i);

			//N NOW BECOMES THE NUMBER OF POINTS TO PROCESS PER BATCH
			//AS ONE GPU THREAD PROCESSES A SINGLE POINT
			


			
			if (i<batchesThatHaveOneMore)
			{
				N[tid]=batchSize+1;	
				printf("\nN (GPU threads): %d, tid: %d",N[tid], tid);
			}
			else
			{
				N[tid]=batchSize;	
				printf("\nN (1 less): %d tid: %d",N[tid], tid);
			}

			//if the number of points is 0, exit
			//can happen if only a few points for kNN
			if (N[tid]==0)
			{	
				continue;
			}

			//set relevant parameters for the batched execution that get reset
			
			//copy N to device 
			//N IS THE NUMBER OF THREADS
			errCode=cudaMemcpyAsync( &dev_N[tid], &N[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: N Got error with code " << errCode << endl; 
			}

			//the batched result set size (reset to 0):
			cnt[tid]=0;
			errCode=cudaMemcpyAsync( &dev_cnt[tid], &cnt[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
			}

			//the offset for batching, which keeps track of where to start processing at each batch
			batchOffset[tid]=numBatches; //for the strided
			errCode=cudaMemcpyAsync( &dev_offset[tid], &batchOffset[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_offset memcpy Got error with code " << errCode << endl; 
			}

			//the batch number for batching with strided
			batchNumber[tid]=i;
			errCode=cudaMemcpyAsync( &dev_batchNumber[tid], &batchNumber[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_batchNumber memcpy Got error with code " << errCode << endl; 
			}

			/*
			#if THREADMULTI==0
			const int TOTALBLOCKS=ceil((1.0*(N[tid]))/(1.0*BLOCKSIZE));	
			printf("\ntotal blocks: %d",TOTALBLOCKS);
			#endif

			#if THREADMULTI>=2
			const int TOTALBLOCKS=ceil((1.0*(N[tid])*(THREADMULTI*1.0)*(*iter+1.0))/(1.0*BLOCKSIZE));	
			

			printf("\ntotal blocks (THREADMULTI==%d, iteration: %u for more distance calculation threads per query point per iteration): %d",THREADMULTI, *iter, TOTALBLOCKS);
			#endif

			//number of threads per point
			//dynamic (-1) or static (-2)
			#if THREADMULTI==-1 || THREADMULTI==-2
			const int TOTALBLOCKS=ceil((1.0*(N[tid])*(*threadsForDistanceCalc))/(1.0*BLOCKSIZE));	
			
				#if THREADMULTI==-1
				printf("\ntotal blocks (THREADMULTI DYNAMIC, ThreadsPerPoint: %u, for more distance calculation threads per query point per iteration): %d",*threadsForDistanceCalc, TOTALBLOCKS);
				#endif

				#if THREADMULTI==-2
				printf("\ntotal blocks (THREADMULTI STATIC, ThreadsPerPoint: %u, for more distance calculation threads per query point per iteration): %d",*threadsForDistanceCalc, TOTALBLOCKS);
				#endif

			#endif
			*/

			//Dynamic for brute force
			const int TOTALBLOCKS=ceil((1.0*(N[tid])*(*threadsForDistanceCalc))/(1.0*BLOCKSIZE));	
			printf("\ntotal blocks (THREADMULTI DYNAMIC Brute Force, ThreadsPerPoint: %u, for more distance calculation threads per query point per iteration): %d",*threadsForDistanceCalc, TOTALBLOCKS);
			

			
			

			//execute kernel -- normal kernel that gets executed using an index	
			//0 is shared memory pool
			// kernelNDGridIndexGlobalkNNSubsequentExecution<<< TOTALBLOCKS, BLOCKSIZE, 0, stream[tid]>>>(dev_debug1, dev_debug2, &dev_N[tid], 
			// &dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_epsilon, dev_grid, dev_indexLookupArr, 
			// dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask, 
			// dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid], dev_distancesKeyValue[tid], dev_queryPts, dev_threadsForDistanceCalc, dev_workCounts);


			kernelNDBruteForce<<< TOTALBLOCKS, BLOCKSIZE, 0, stream[tid]>>>(dev_debug1, dev_debug2, &dev_N[tid], dev_DBSIZE,
			&dev_offset[tid], &dev_batchNumber[tid], dev_database, &dev_cnt[tid], dev_pointIDKey[tid], dev_pointInDistValue[tid], dev_distancesKeyValue[tid], 
			dev_queryPts, dev_threadsForDistanceCalc, dev_workCounts);

			// __global__ void kernelNDBruteForce(unsigned int *debug1, unsigned int *debug2, unsigned int *N, unsigned int * DBSIZE, 
			// unsigned int * offset, unsigned int *batchNum, DTYPE* database, unsigned int * cnt, 
			// int * pointIDKey, int * pointInDistVal, double * distancesKeyVal, unsigned int * queryPts, unsigned int * threadsForDistanceCalc, CTYPE* workCounts)		

			// errCode=cudaDeviceSynchronize();
			// cout <<"\n\nError from device synchronize: "<<errCode;

			cout <<"\n\nKERNEL LAUNCH RETURN: "<<cudaGetLastError()<<endl<<endl;
			if ( cudaSuccess != cudaGetLastError() ){
		    	cout <<"\n\nERROR IN KERNEL LAUNCH. ERROR: "<<cudaSuccess<<endl<<endl;
		    }

		    

		   
			// find the size of the number of results
			errCode=cudaMemcpyAsync( &cnt[tid], &dev_cnt[tid], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
			}
			else{
				printf("\nGPU: result set size within epsilon (GPU grid): %d",cnt[tid]);cout.flush();
				fprintf(stderr,"\nIter: Brute Force, GPU: result set size within epsilon (GPU grid): %d",cnt[tid]);
			}


			

			
			////////////////////////////////////
			//SORT THE TABLE DATA ON THE GPU
			//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
			////////////////////////////////////

			/////////////////////////////
			//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
			//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
			//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
			/////////////////////////////

			//sort by key with the data already on the device:
			//wrap raw pointer with a device_ptr to use with Thrust functions
			thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey[tid]);
			thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue[tid]);
			thrust::device_ptr<DTYPE> dev_distancesKeyValPtr(dev_distancesKeyValue[tid]); // dist for knn

			// ZipIterator iterBegin(thrust::make_tuple(dev_data_ptr, dev_distancesKeyValPtr));  

			// thrust::make_zip_iterator(thrust::make_tuple(dev_pointInDistValue[tid], dev_distancesKeyValue[tid]));
			// thrust::make_zip_iterator(thrust::make_tuple(dev_pointInDistValue[tid]+ cnt[tid], dev_distancesKeyValue[tid]+ cnt[tid]));


			//XXXXXXXXXXXXXXXX
			//THRUST USING STREAMS REQUIRES THRUST V1.8 
			//XXXXXXXXXXXXXXXX
			
			//original first step
			

			
			
			//k-NN
			//Need the 3 arrays: keys, values, distances
			//back-to-back sort - keys: query points, values: points wihtin the distance of the keys
			//first sort the key/values by the distance
			//then sort the distances/values by the keys
			//uses the tuple to move the values in the arrays around when sorting
			try{
			
				//first sort by the distance array (not the key)
				thrust::stable_sort_by_key(thrust::cuda::par.on(stream[tid]), dev_distancesKeyValPtr, dev_distancesKeyValPtr + cnt[tid], 
					thrust::make_zip_iterator(thrust::make_tuple(dev_keys_ptr, dev_data_ptr)));

				//next sort by the key array 
				thrust::stable_sort_by_key(thrust::cuda::par.on(stream[tid]), dev_keys_ptr, dev_keys_ptr + cnt[tid], 
					thrust::make_zip_iterator(thrust::make_tuple(dev_data_ptr, dev_distancesKeyValPtr)));

			}
			// catch(std::bad_alloc &e)
			//   {
			//     std::cerr << "Ran out of memory while sorting, " << GPUBufferSize << std::endl;
			//     exit(-1);
			//   }
			catch(thrust::system_error e)
  			{
    		std::cerr << "Error inside sort: " << e.what() << std::endl;
    		exit(-1);
  			}

  		

	  		//thrust with streams into individual buffers for each batch
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey[tid]), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue[tid]), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);	
			cudaMemcpyAsync(thrust::raw_pointer_cast(distancesKeyValue[tid]), thrust::raw_pointer_cast(dev_distancesKeyValPtr), cnt[tid]*sizeof(double), cudaMemcpyDeviceToHost, stream[tid]);	

			//need to make sure the data is copied before constructing portion of the neighbor table
			cudaStreamSynchronize(stream[tid]);

			////////////////////////////////////
			//ORIGINAL TABLE CONSTRUCTION
			/*
			double tableconstuctstart=omp_get_wtime();
			//set the number of neighbors in the pointer struct:
			(*pointersToNeighbors)[i].sizeOfDataArr=cnt[tid];    
			(*pointersToNeighbors)[i].dataPtr=new int[cnt[tid]]; 
			(*pointersToNeighbors)[i].distancePtr=new DTYPE[cnt[tid]];
			ValsPtr.insert((*pointersToNeighbors)[i].dataPtr); //used to free memory later
			DistancePtr.insert((*pointersToNeighbors)[i].distancePtr); //used to free memory later
			constructNeighborTableKeyValueWithPtrskNN(pointIDKey[tid], pointInDistValue[tid], distancesKeyValue[tid], neighborTable, (*pointersToNeighbors)[i].dataPtr, (*pointersToNeighbors)[i].distancePtr, &cnt[tid]);

			
			//cout <<"\nIn make neighbortable. Data array ptr: "<<(*pointersToNeighbors)[i].dataPtr<<" , size of data array: "<<(*pointersToNeighbors)[i].sizeOfDataArr;cout.flush();

			double tableconstuctend=omp_get_wtime();	
			
			printf("\nTable construct time: %f", tableconstuctend - tableconstuctstart);
			*/
			/////////////////////////
			//END ORIGINAL TABLE CONSTRUCTION
			////////////////////////

			//directly store the kNN from the pinned memory buffers

			//COMMENTING FOR TESTING

			
			double fillerargdist=0;

			double knnStoreStart=omp_get_wtime();
			storeNeighborTableForkNNOnTheFly(pointIDKey[tid], pointInDistValue[tid], distancesKeyValue[tid], 
				&cnt[tid], NDdataPoints, k_neighbors, queryPtsVect, nearestNeighborTable, nearestNeighborTableDistances, &fillerargdist);

			double knnStoreEnd=omp_get_wtime();
			printf("\nkNN store time: %f", knnStoreEnd - knnStoreStart);
			


			//add the batched result set size to the total count
			totalResultsLoop+=cnt[tid];

			
			//add the batched result set size to the total count
			totalResultsLoop+=cnt[tid];

			printf("\nRunning total of total size of result array, tid: %d: %lu", tid, totalResultsLoop);
			//}
			
	
			


		

		} //END LOOP OVER THE GPU BATCHES

		// cudaDeviceSynchronize();


#if COUNTMETRICS == 1
        cudaMemcpy(workCounts, dev_workCounts, 2*sizeof(CTYPE), cudaMemcpyDeviceToHost );
#endif

	
	
	printf("\nTOTAL RESULT SET SIZE ON HOST:  %lu", totalResultsLoop);
	*totalNeighbors=totalResultsLoop;


	double tKernelResultsEnd=omp_get_wtime();
	
	printf("\nTime to launch kernel and execute everything (get results etc.) except freeing memory: %f",tKernelResultsEnd-tKernelResultsStart);


	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////



	///////////////////////////////////	
	//OPTIONAL DEBUG VALUES
	///////////////////////////////////
	
	// double tStartdebug=omp_get_wtime();

	// errCode=cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	// }
	// else
	// {
	// 	printf("\nDebug1 value: %u",*debug1);
	// }

	// errCode=cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	// }
	// else
	// {
	// 	printf("\nDebug2 value: %u",*debug2);
	// }	

	// double tEnddebug=omp_get_wtime();
	// printf("\nTime to retrieve debug values: %f", tEnddebug - tStartdebug);
	

	///////////////////////////////////	
	//END OPTIONAL DEBUG VALUES
	///////////////////////////////////
	

	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
	// if (NUM_TRIALS>1)
	// {

	double tFreeStart=omp_get_wtime();

	for (int i=0; i<GPUSTREAMS; i++){
		errCode=cudaStreamDestroy(stream[i]);
		if(errCode != cudaSuccess) {
		cout << "\nError: destroying stream" << errCode << endl; 
		}
	}



	//free the data on the device
	// cudaFree(dev_pointIDKey);
	// cudaFree(dev_pointInDistValue);
	// cudaFree(dev_distancesKeyValue);
	

	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	// cudaFree(dev_epsilon);
	cudaFree(dev_N); 	
	cudaFree(dev_cnt); 
	cudaFree(dev_offset); 
	cudaFree(dev_batchNumber); 
	cudaFree(dev_threadsForDistanceCalc);
	cudaFree(dev_queryPts);
	cudaFree(dev_workCounts);
	
	//free data related to the individual streams for each batch
	for (int i=0; i<GPUSTREAMS; i++){
		//free the data on the device
		cudaFree(dev_pointIDKey[i]);
		cudaFree(dev_pointInDistValue[i]);
		cudaFree(dev_distancesKeyValue[i]);


		//free on the host
		if (pinnedSavedFlag==0)
		{
		cudaFreeHost(pointIDKey[i]);
		cudaFreeHost(pointInDistValue[i]);
		cudaFreeHost(distancesKeyValue[i]);
		}

	}

	
	

	double tFreeEnd=omp_get_wtime();

	printf("\nTime freeing memory: %f", tFreeEnd - tFreeStart);
	cout<<"\n** last error at end of fn batches (could be from freeing memory): "<<cudaGetLastError();

}







void warmUpGPU(){
// initialize all ten integers of a device_vector to 1 
thrust::device_vector<int> D(10, 1); 
// set the first seven elements of a vector to 9 
thrust::fill(D.begin(), D.begin() + 7, 9); 
// initialize a host_vector with the first five elements of D 
thrust::host_vector<int> H(D.begin(), D.begin() + 5); 
// set the elements of H to 0, 1, 2, 3, ... 
thrust::sequence(H.begin(), H.end()); // copy all of H back to the beginning of D 
thrust::copy(H.begin(), H.end(), D.begin()); 
// print D 
for(int i = 0; i < D.size(); i++) 
std::cout << " D[" << i << "] = " << D[i]; 


return;
}



//this function gets the key (point) values (points within the distance), and the distances
void constructNeighborTableKeyValueWithPtrskNN(int * pointIDKey, int * pointInDistValue, DTYPE * distancePointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, DTYPE * pointersToDistances, unsigned int * cnt)
{

	
	//copy the value data:
	std::copy(pointInDistValue, pointInDistValue+(*cnt), pointersToNeighbors);
	std::copy(distancePointInDistValue, distancePointInDistValue+(*cnt), pointersToDistances);



	//Step 1: find all of the unique keys and their positions in the key array
	unsigned int numUniqueKeys=0;

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key=pointIDKey[0];
	tmp.position=0;
	uniqueKeyData.push_back(tmp);

	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (int i=1; i<(*cnt)+1; i++){
		if (pointIDKey[i-1]!=pointIDKey[i]){
			numUniqueKeys++;
			tmp.key=pointIDKey[i];
			tmp.position=i;
			tmp.distance=distancePointInDistValue[i];
			uniqueKeyData.push_back(tmp);
		}
	}



	
	//insert into the neighbor table the values based on the positions of 
	//the unique keys obtained above. 
	for (int i=0; i<uniqueKeyData.size()-1; i++) {
		int keyElem=uniqueKeyData[i].key;
		neighborTable[keyElem].pointID=keyElem;
		neighborTable[keyElem].indexmin=uniqueKeyData[i].position;
		neighborTable[keyElem].indexmax=uniqueKeyData[i+1].position-1;
		//update the pointer to the data array for the values
		neighborTable[keyElem].dataPtr=pointersToNeighbors;	
		//update the pointer to the data array for the distances
		neighborTable[keyElem].distancePtr=pointersToDistances;
	}

}






void constructNeighborTableKeyValue(int * pointIDKey, int * pointInDistValue, struct table * neighborTable, unsigned int * cnt)
{
	
	//newer multithreaded way:
	//Step 1: find all of the unique keys and their positions in the key array
	
	//double tstart=omp_get_wtime();

	unsigned int numUniqueKeys=0;
	unsigned int count=0;

	

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key=pointIDKey[0];
	tmp.position=0;
	uniqueKeyData.push_back(tmp);



	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (int i=1; i<(*cnt)+1; i++)
	{
		if (pointIDKey[i-1]!=pointIDKey[i])
		{
			numUniqueKeys++;
			tmp.key=pointIDKey[i];
			tmp.position=i;
			uniqueKeyData.push_back(tmp);
		}
	}



	//Step 2: In parallel, insert into the neighbor table the values based on the positions of 
	//the unique keys obtained above. Since multiple threads access this function, we don't want to 
	//do too many memory operations while GPU memory transfers are occurring, or else we decrease the speed that we 
	//get data off of the GPU
	omp_set_nested(1);
	#pragma omp parallel for reduction(+:count) num_threads(2) schedule(static,1)
	for (int i=0; i<uniqueKeyData.size()-1; i++) 
	{
		int keyElem=uniqueKeyData[i].key;
		int valStart=uniqueKeyData[i].position;
		int valEnd=uniqueKeyData[i+1].position-1;
		int size=valEnd-valStart+1;
		
		//seg fault from here: is it neighbortable mem alloc?
		neighborTable[keyElem].pointID=keyElem;
		neighborTable[keyElem].neighbors.insert(neighborTable[keyElem].neighbors.begin(),&pointInDistValue[valStart],&pointInDistValue[valStart+size]);
		
		//printf("\nval: start:%d, end: %d", valStart,valEnd);
		//printf("\ni: %d, keyElem: %d, position start: %d, position end: %d, size: %d", i,keyElem,valStart, valEnd,size);	


		count+=size;

	}
	

}






//gets the average distance between points
//samples the dataset
//needed to estimate epsilon
//DTYPE * epsilon_guess -- original epsilon for iteration 2
void sampleNeighborsBruteForce(std::vector<std::vector <DTYPE> > * NDdataPoints, DTYPE * epsilon_guess, unsigned int * bucket, unsigned int k_neighbors)
{
	//CUDA error code:
	cudaError_t errCode;


	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int));
	*N=NDdataPoints->size();
	
	printf("\nIn main GPU method: Number of data points, (N), is: %u ",*N);cout.flush();




	
	//the database will just be a 1-D array, we access elemenets based on NDIM
	DTYPE* database= (DTYPE*)malloc(sizeof(DTYPE)*(*N)*GPUNUMDIM);  
	DTYPE* dev_database= (DTYPE*)malloc(sizeof(DTYPE)*(*N)*GPUNUMDIM);  
	

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_database, sizeof(DTYPE)*GPUNUMDIM*(*N));		
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; cout.flush(); 
	}


	



	//copy the database from the ND vector to the array:
	for (int i=0; i<*N; i++){
		std::copy((*NDdataPoints)[i].begin(), (*NDdataPoints)[i].end(), database+(i*GPUNUMDIM));
	}
	//test 
	// printf("\n\n");
	// int tmpcnt=0;
	// for (int i=0; i<NDdataPoints->size(); i++)
	// {
	// 	for (int j=0; j<(*NDdataPoints)[i].size(); j++)
	// 	{
	// 		database[tmpcnt]=(*NDdataPoints)[i][j];
	// 		tmpcnt++;
	// 	}
	// }
	// for (int i=0; i<(*N)*GPUNUMDIM; i++){
	// 	printf("%f,",database[i]);
	// }	



	
	//copy database to the device:
	errCode=cudaMemcpy(dev_database, database, sizeof(DTYPE)*(*N)*GPUNUMDIM, cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; 
	}	




	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////



	//copy total distance 
	double * total_distance=(double*)malloc(sizeof(double));
	*total_distance=0;
	double * dev_total_distance=(double*)malloc(sizeof(double));
	

	//allocate total distance on the device
	errCode=cudaMalloc((double**)&dev_total_distance, sizeof(double));	
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_total_distance Got error with code " << errCode << endl; 
	}

	//copy
	errCode=cudaMemcpy( dev_total_distance, total_distance, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_total_distance memcpy Got error with code " << errCode << endl; 
	}









	///////////////////////////////////
	//SET OTHER KERNEL PARAMETERS
	///////////////////////////////////

	

	//count values
	unsigned long long int * cnt;
	cnt=(unsigned long long int*)malloc(sizeof(unsigned long long int));
	*cnt=0;

	unsigned long long int * dev_cnt; 
	dev_cnt=(unsigned long long int*)malloc(sizeof(unsigned long long int));
	*dev_cnt=0;

	//allocate on the device
	errCode=cudaMalloc((unsigned long long int**)&dev_cnt, sizeof(unsigned long long int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: cnt Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_cnt, cnt, sizeof(unsigned long long int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
	}



		
	//size of the database:
	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof( unsigned int ));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}	



	//debug values
	unsigned int * dev_debug1; 
	unsigned int * dev_debug2; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;




	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 Got error with code " << errCode << endl; 
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 Got error with code " << errCode << endl; 
	}		



	//N (DATASET SIZE)
	errCode=cudaMemcpy( dev_N, N, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}		

	



	///////////////////////////////////
	//END SET OTHER KERNEL PARAMETERS
	///////////////////////////////////


	


	///////////////////////////////////
	//LAUNCH KERNEL
	///////////////////////////////////

	const int TOTALBLOCKS=ceil((0.001*(*N))/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks: %d",TOTALBLOCKS);


	//execute kernel	

	
	double tkernel_start=omp_get_wtime();
	kernelEstimateAvgDistBruteForce<<< TOTALBLOCKS, BLOCKSIZE >>>(dev_N, dev_debug1, dev_debug2, dev_cnt, dev_database, dev_total_distance);
	if ( cudaSuccess != cudaGetLastError() ){
    	printf( "Error in kernel launch!\n" );
    }


    cudaDeviceSynchronize();
    double tkernel_end=omp_get_wtime();
    printf("\nTime for kernel only (brute force average dist estimator): %f", tkernel_end - tkernel_start);
    ///////////////////////////////////
	//END LAUNCH KERNEL
	///////////////////////////////////
    


    ///////////////////////////////////
	//GET RESULT SET
	///////////////////////////////////

	//The total number of distance calculations
	errCode=cudaMemcpy( cnt, dev_cnt, sizeof(unsigned long long int), cudaMemcpyDeviceToHost );
	if(errCode != cudaSuccess) {
	cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nGPU: Number of distance calculations: %llu",*cnt);
	}




	//The total distance
	errCode=cudaMemcpy(total_distance, dev_total_distance, sizeof(double), cudaMemcpyDeviceToHost );
	if(errCode != cudaSuccess) {
	cout << "\nError: getting dev_total_distance from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nGPU: dev_total_distance: %f",*total_distance);
	}


	
	

	double avg_distance=*total_distance/(*cnt*1.0);
	printf("\nAvg distance: %f",avg_distance);
	// avg_distance=avg_distance*0.20;
	// printf("\nAvg distance (20 percent of it): %f",avg_distance);


	//can't estimate epsilon with arguments based on volumes. Need a k-dist histogram

	unsigned int * nbuckets;
	nbuckets=(unsigned int*)malloc(sizeof(unsigned int));
	// *nbuckets=10000; //original
	*nbuckets=10000;

	double * dev_avg_distance;
	

	double * bucketWidth;
	bucketWidth=(double*)malloc(sizeof(double));
	*bucketWidth=avg_distance/(*nbuckets*1.0);
	
	double * dev_bucketWidth;
	
	//histogram data
	unsigned int * histogram;
	histogram=(unsigned int*)calloc((*nbuckets),sizeof(unsigned int));

	unsigned int * dev_histogram;

	printf("\nNum buckets: %u, bucketWidth: %f",*nbuckets,*bucketWidth);

	//alloc on device: avg_distance, bucketWidth, histogram
	errCode=cudaMalloc( (double **)&dev_avg_distance, sizeof(double ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: avg_distance Got error with code " << errCode << endl; 
	}		

	errCode=cudaMalloc( (double **)&dev_bucketWidth, sizeof(double ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: bucket width Got error with code " << errCode << endl; 
	}		

	errCode=cudaMalloc( (unsigned int **)&dev_histogram, sizeof(unsigned int )*(*nbuckets) );
	if(errCode != cudaSuccess) {
	cout << "\nError: histogram Got error with code " << errCode << endl; 
	}	

	//memcpy to device: avg_distance, bucketWidth, histogram

	errCode=cudaMemcpy( dev_avg_distance, &avg_distance, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: avg_distance memcpy Got error with code " << errCode << endl; 
	}			


	errCode=cudaMemcpy( dev_bucketWidth, bucketWidth, sizeof(double), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: nbuckets memcpy Got error with code " << errCode << endl; 
	}			

	errCode=cudaMemcpy( dev_histogram, histogram, sizeof(unsigned int)*(*nbuckets), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: histogram memcpy Got error with code " << errCode << endl; 
	}	



	//offset for sampling the dataset:
	//Offset -- the point offset based on 1% of the data for 1M points
	//don't do 1% of all points otherwise takes too long on large datasets
	unsigned int * offset=(unsigned int*)malloc(sizeof(unsigned int));
	unsigned int * dev_offset;
	double offsetRate=0.01;

	if (*N>=50000)
	{
		double frac=(50000.0/(*N))*offsetRate;
		*offset=1.0/frac;
	}
	else
	{
		*offset=1.0/offsetRate; 
	}

	printf("\n[Epsilon estimator] Offset: %u", *offset);


	errCode=cudaMalloc( (unsigned int **)&dev_offset, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: offset Got error with code " << errCode << endl; 
	}	

	errCode=cudaMemcpy( dev_offset, offset, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: offset memcpy Got error with code " << errCode << endl; 
	}	


	// unsigned int *N, unsigned int *debug1, unsigned int *debug2, 
	// unsigned long long int * cnt, DTYPE* database, double * avg_distance, unsigned int * histogram, double * bucket_width

	// const int TOTALBLOCKS2=ceil((1*(*N))/(1.0*BLOCKSIZE));	
	// const int TOTALBLOCKS2=ceil((1*(*N))/(1.0*BLOCKSIZE));	

	const int TOTALBLOCKS2=ceil((*N)/(*offset));	
	// const int TOTALBLOCKS2=ceil((*N)/300);	
	printf("\ntotal blocks: %d",TOTALBLOCKS2);
	kernelKDistBruteForce<<< TOTALBLOCKS2, BLOCKSIZE >>>(dev_N, dev_offset, dev_debug1, dev_debug2, dev_cnt, dev_database, dev_avg_distance, dev_histogram, dev_bucketWidth);

	 ///////////////////////////////////
	//GET RESULT SET
	///////////////////////////////////
	unsigned int totalInBuckets=0;
	//The histogram
	errCode=cudaMemcpy( histogram, dev_histogram, sizeof(unsigned int)*(*nbuckets), cudaMemcpyDeviceToHost );
	if(errCode != cudaSuccess) {
	cout << "\nError: getting histogram from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		for (unsigned int i=0; i<(*nbuckets); i++)
		{
			totalInBuckets+=histogram[i];
		}
		printf("\nTotal in buckets: %u",totalInBuckets);

	}


		/*
		
		//we have the k-dist information, but we use all of the points in a cell, not within the hypersphere defined by epsilon	
		double volume_unit_ncube=powf(2*1,NUMINDEXEDDIM); //hypercube with radius 1 at the center (2* radius)

		// https://en.wikipedia.org/wiki/N-sphere
		// sanity check: https://keisan.casio.com/exec/system/1223381019
		double a=(powf(M_PI,(NUMINDEXEDDIM*0.5))*powf(1,NUMINDEXEDDIM));
		double b=(tgamma(((NUMINDEXEDDIM*0.5))+1.0));
		
		double volume_unit_nsphere=a/b; //n-sphere

		double ratio_ncube_nsphere=volume_unit_ncube/volume_unit_nsphere;

		printf("\nRatio ncube to nsphere: %f",ratio_ncube_nsphere);
		*/

	
		unsigned int cumulative=0;

		double bestEpsilon=0;
		int bucketNumFound=-1;
		bool flag=0;
		bool flag2=0;	

		double beta=BETA;

		for (unsigned int i=0; i<(*nbuckets); i++)
		{

			cumulative+=histogram[i];
			double cumulative_per_point_sampled=(cumulative/(1.0*TOTALBLOCKS2));
			
			//epsilon distance is the bucket that gives k-NN number of cumulative neighbors 
			//e.g., if k is 5, then we use the epsilon that gives 5 cumulative neighbors on average
			
			//uncomment to print the buckets
			// printf("\nbucket: %d, val: %u, epsilon-distance: %f, Cumulative neighbors: %u, Cumulative neighbors/point sampled: %f",i,histogram[i],(*bucketWidth*(i+1)), cumulative, cumulative_per_point_sampled);


			

			unsigned long int min_pts=k_neighbors*1.0;
			unsigned long int max_pts=k_neighbors*100.0;
			unsigned long int point_threshold_beta=min_pts+(max_pts-min_pts)*beta;

			//output the epsilon if using the min value (gamma=0)
			if ((cumulative_per_point_sampled>(k_neighbors*1.0)) && flag2==0)
			{
				printf("\nBest epsilon if beta=0: %f", (*bucketWidth*(i+1.0)));
				flag2=1;
			}

			if ((cumulative_per_point_sampled>(point_threshold_beta)) && flag==0)
			{
				bestEpsilon=(*bucketWidth*(i+1.0));
				bucketNumFound=i;
				flag=1;
			}

			//if the last iteration and the bucket has not been found, set to the last bucket
			//modified for outlier detection for potentially high values of K
			if ((i==(*nbuckets)-1) && (bucketNumFound==-1))
			{
				
				bestEpsilon=(*bucketWidth*(i+1.0));
				bucketNumFound=i;
				printf("\nThe bucket was not found even at the last bucket. Setting epsilon to be estimated based on the last bucket.\nBest epsilon: %f", bestEpsilon);
			}

		}

	
	printf("\nThe epsilon is the range query distance needed to find at least k neighbors (at least k beta=0)");
	printf("\nThis epsilon value needs to be multiplied by 2 to correspond to a cell that contains the n-sphere with radius epsilon. \nI.e., in 2-D, the cell circumscribing the circle is of length 2xEpsilon");	



	
	*epsilon_guess=bestEpsilon;
	*bucket=bucketNumFound;
	
	printf("\nBest Epsilon Found (Use cumulative neighbors per point, beta=%f) %f, In bucket: %d",beta,*epsilon_guess,bucketNumFound);

	
	
	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////


	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
    //free:
	cudaFree(dev_database);
	cudaFree(dev_total_distance);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	cudaFree(dev_cnt);
	cudaFree(dev_histogram);
	cudaFree(dev_avg_distance);
	cudaFree(dev_bucketWidth);

	//cudaFree(dev_results);

	////////////////////////////////////


}



void cleanUpPinned()
{
	//free host pinned memory that was saved

	if (pinnedSavedFlag==1)
	{
	//free data related to the individual streams for each batch
	for (int i=0; i<GPUSTREAMS; i++){		
		cudaFreeHost(pointIDKeySaved[i]);
		cudaFreeHost(pointInDistValueSaved[i]);
		cudaFreeHost(distancesKeyValueSaved[i]);
		}

	}
}


void cleanUpNeighborTable()
{
	for (auto it = ValsPtr.begin(); it != ValsPtr.end(); ++it)
	{
		delete(*it);
	}

	for (auto it = DistancePtr.begin(); it != DistancePtr.end(); ++it)
	{
		delete(*it);
	}

	ValsPtr.clear();
	DistancePtr.clear();

}




//We use the threads used for batching to store the neighbortable into the "final" kNN table
//This is so that we can free buffer memory as we compute the batches
//And don't need to do this between kNN iteration rounds
void storeNeighborTableForkNNOnTheFly(int * pointIDKey, int * pointInDistValue, DTYPE * distancePointInDistValue, unsigned int * cnt, std::vector<std::vector<DTYPE> > * NDdataPoints,
	int k_Neighbors, std::vector<unsigned int> *queryPts, int * nearestNeighborTable, DTYPE * nearestNeighborTableDistances, double * totaldistance)
{

	uint64_t KNN=k_Neighbors+1;

	struct keysForKNN
	{
		uint64_t key; //the key (a point)
		uint64_t position_min; //position in the array where it is found
		uint64_t position_max; //maximum position in the array
		//the distance between position min and max determines if there are at least k neighbors 
	};

	

	//Step 1: find all of the unique keys and their positions in the key array
	uint64_t numUniqueKeys=0;

	std::vector<keysForKNN> uniqueKeyData;

	keysForKNN tmp;
	tmp.key=pointIDKey[0];
	tmp.position_min=0;
	uniqueKeyData.push_back(tmp);

	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (uint64_t i=1; i<(uint64_t)(*cnt)+1; i++){
		
		if (pointIDKey[i-1]!=pointIDKey[i]){
			
			//update the maximum value for key-1
			uniqueKeyData[numUniqueKeys].position_max=i-1;

			numUniqueKeys++;
			tmp.key=pointIDKey[i];
			tmp.position_min=i;

			uniqueKeyData.push_back(tmp);
		}
	}

	//the last maximum that needs to be updated afterwards
	uniqueKeyData[numUniqueKeys].position_max=(uint64_t)(*cnt)-1;


	//we now go through the position min/max values and get the unique keys with at least k neighbors
	std::vector<keysForKNN> uniqueKeyDataSufficientNeighbors;
	for (uint64_t i=0; i<uniqueKeyData.size(); i++)
	{
		uint64_t num_neighbors=uniqueKeyData[i].position_max-uniqueKeyData[i].position_min+1;
		if (num_neighbors>=KNN)
		{
			tmp.key=uniqueKeyData[i].key;
			tmp.position_min=uniqueKeyData[i].position_min;
			tmp.position_max=uniqueKeyData[i].position_min+KNN;
			uniqueKeyDataSufficientNeighbors.push_back(tmp);
		}
	}

	//we now copy the points with sufficient neighbors to the final kNN array
	//3 threads can saturate memory bandwidth for memcpy
	
	


	double totaldisttmp=0;

	#pragma omp parallel for num_threads(3) reduction(+:totaldisttmp)
	for (uint64_t i=0; i<uniqueKeyDataSufficientNeighbors.size(); i++)
	{
		uint64_t pointID=uniqueKeyDataSufficientNeighbors[i].key;
		uint64_t minIdx=uniqueKeyDataSufficientNeighbors[i].position_min;
		uint64_t maxIdx=uniqueKeyDataSufficientNeighbors[i].position_max;
		//offset into the 1-D rray of results
		uint64_t offset=pointID*KNN;
		
		std::copy(&pointInDistValue[minIdx], &pointInDistValue[maxIdx], nearestNeighborTable+offset);	

		for (uint64_t j=minIdx; j<maxIdx; j++)
		{
			totaldisttmp+=(distancePointInDistValue[j]*distancePointInDistValue[j]); //unsquare rooted
		}

		std::copy(&distancePointInDistValue[minIdx], &distancePointInDistValue[maxIdx], nearestNeighborTableDistances+offset);	
		
	}


	*totaldistance=totaldisttmp;
	


}





//New: index on the GPU
//output:
//struct gridCellLookup ** gridCellLookupArr
//struct grid ** index
//unsigned int * indexLookupArr
void populateNDGridIndexAndLookupArrayGPU(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE *epsilon, DTYPE* minArr,  uint64_t totalCells, unsigned int * nCells, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr, unsigned int *nNonEmptyCells)
{

	printf("\nIndexing on the GPU");

	//CUDA error code:
	cudaError_t errCode;


	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////

	unsigned int * DBSIZE;
	DBSIZE=(unsigned int*)malloc(sizeof(unsigned int));
	*DBSIZE=NDdataPoints->size();
	
	printf("\nDBSIZE is: %u",*DBSIZE);cout.flush();

	unsigned int * dev_DBSIZE;

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_DBSIZE, sizeof(unsigned int));		
	if(errCode != cudaSuccess) {
	cout << "\nError: database N -- error with code " << errCode << endl; cout.flush(); 
	}


	//copy database size to the device
	errCode=cudaMemcpy(dev_DBSIZE, DBSIZE, sizeof(unsigned int), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database size Got error with code " << errCode << endl; 
	}
	
	DTYPE* database= (DTYPE*)malloc(sizeof(DTYPE)*(*DBSIZE)*(GPUNUMDIM));  
	
	DTYPE* dev_database;  
		
	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE));		
	if(errCode != cudaSuccess) {
	cout << "\nError: database alloc -- error with code " << errCode << endl; cout.flush(); 
	}


	//copy the database from the ND vector to the array:
	for (int i=0; i<(*DBSIZE); i++){
		std::copy((*NDdataPoints)[i].begin(), (*NDdataPoints)[i].end(), database+(i*(GPUNUMDIM)));
	}


	//copy database to the device
	errCode=cudaMemcpy(dev_database, database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database2 Got error with code " << errCode << endl; 
	}




	//printf("\n size of database: %d",N);



	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY GRID DIMENSIONS TO THE GPU
	//THIS INCLUDES THE NUMBER OF CELLS IN EACH DIMENSION, 
	//AND THE STARTING POINT OF THE GRID IN THE DIMENSIONS 
	///////////////////////////////////

	//minimum boundary of the grid:
	DTYPE* dev_minArr;

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_minArr, sizeof(DTYPE)*(NUMINDEXEDDIM));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc minArr -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_minArr, minArr, sizeof(DTYPE)*(NUMINDEXEDDIM), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: Copy minArr to device -- error with code " << errCode << endl; 
	}	


	//number of cells in each dimension
	unsigned int * dev_nCells;
	

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_nCells, sizeof(unsigned int)*(NUMINDEXEDDIM));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc nCells -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_nCells, nCells, sizeof(unsigned int)*(NUMINDEXEDDIM), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: Copy nCells to device -- error with code " << errCode << endl; 
	}	


	///////////////////////////////////
	//END COPY GRID DIMENSIONS TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//EPSILON
	///////////////////////////////////
	DTYPE* dev_epsilon;
	// dev_epsilon=(DTYPE*)malloc(sizeof( DTYPE));
	

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(DTYPE));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc epsilon -- error with code " << errCode << endl; 
	}

	//copy to device
	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(DTYPE), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon copy to device -- error with code " << errCode << endl; 
	}		

	///////////////////////////////////
	//END EPSILON
	///////////////////////////////////


	///////////////////////////////////
	//Array for each point's cell
	///////////////////////////////////

	uint64_t * dev_pointCellArr;  
		
	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_pointCellArr, sizeof(uint64_t)*(*DBSIZE));		
	if(errCode != cudaSuccess) {
	cout << "\nError: point cell array alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	///////////////////////////////////
	//End array for each point's cell
	///////////////////////////////////


	//First: we get the number of non-empty grid cells


	const int TOTALBLOCKS=ceil((1.0*(*DBSIZE))/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks: %d",TOTALBLOCKS);

	kernelIndexComputeNonemptyCells<<< TOTALBLOCKS, BLOCKSIZE>>>(dev_database, dev_DBSIZE, dev_epsilon, dev_minArr, dev_nCells, dev_pointCellArr);


	cudaDeviceSynchronize();

	thrust::device_ptr<uint64_t> dev_pointCellArr_ptr(dev_pointCellArr);

	thrust::device_ptr<uint64_t> dev_new_end;

	try{
		//first sort
		thrust::sort(thrust::device, dev_pointCellArr_ptr, dev_pointCellArr_ptr + (*DBSIZE)); //, thrust::greater<uint64_t>()
		//then unique
		dev_new_end=thrust::unique(thrust::device, dev_pointCellArr_ptr, dev_pointCellArr_ptr + (*DBSIZE));
	}
	catch(std::bad_alloc &e)
	{
	 	std::cerr << "Ran out of memory while sorting, "<< std::endl;
	    exit(-1);
    }



    uint64_t * new_end = thrust::raw_pointer_cast(dev_new_end);

    uint64_t numNonEmptyCells=std::distance(dev_pointCellArr_ptr,dev_new_end);

    printf("\nNumber of full cells (non-empty): %lu",numNonEmptyCells);

    *nNonEmptyCells=numNonEmptyCells;

    ////////////////////////
    //populate grid cell lookup array- its the uniqued array above (dev_pointCellArr)
    *gridCellLookupArr= new struct gridCellLookup[numNonEmptyCells];

    uint64_t * pointCellArrTmp;  
    pointCellArrTmp=(uint64_t *)malloc((sizeof(uint64_t)*numNonEmptyCells));


    errCode=cudaMemcpy(pointCellArrTmp, dev_pointCellArr, sizeof(uint64_t)*numNonEmptyCells, cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: pointCellArrTmp memcpy Got error with code " << errCode << endl; 
	}

	for (uint64_t i=0; i<numNonEmptyCells; i++)
	{
		(*gridCellLookupArr)[i].idx=i;
		(*gridCellLookupArr)[i].gridLinearID=pointCellArrTmp[i];	
	}

	

	// for (uint64_t i=0; i<numNonEmptyCells; i++)
	// {
	// 	printf("\nGrid idx: %lu, linearID: %lu",(*gridCellLookupArr)[i].idx,(*gridCellLookupArr)[i].gridLinearID);
	// }


	
    //Second: we execute the same kernel again to get each point's cell ID-- we ruined this array by uniquing it
    //So that we don't run out of memory on the GPU for larger datasets, we redo the kernel and sort		
    // const int TOTALBLOCKS=ceil((1.0*(*DBSIZE))/(1.0*BLOCKSIZE));	
	// printf("\ntotal blocks: %d",TOTALBLOCKS);

    //Compute again pointCellArr-- the first one was invalidated because of the unique
	kernelIndexComputeNonemptyCells<<< TOTALBLOCKS, BLOCKSIZE>>>(dev_database, dev_DBSIZE, dev_epsilon, dev_minArr, dev_nCells, dev_pointCellArr);




	//Create "values" for key/value pairs, that are the point idx of the database
	unsigned int * dev_databaseVal;

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_databaseVal, sizeof(unsigned int)*(*DBSIZE));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc databaseVal -- error with code " << errCode << endl; 
	}

	kernelInitEnumerateDB<<< TOTALBLOCKS, BLOCKSIZE>>>(dev_databaseVal, dev_DBSIZE);

	//Sort the point ids by cell ids, using key/value pairs, where 
	//key-cell id, value- point id

	try
	{
	thrust::sort_by_key(thrust::device, dev_pointCellArr, dev_pointCellArr+(*DBSIZE),dev_databaseVal);
	}
	catch(std::bad_alloc &e)
	{
		std::cerr << "Ran out of memory while sorting key/value pairs "<< std::endl;
	    exit(-1);	
	}

	uint64_t * cellKey=(uint64_t *)malloc(sizeof(uint64_t)*(*DBSIZE));
	// unsigned int * databaseIDValue=(unsigned int *)malloc(sizeof(unsigned int)*(*DBSIZE));

	//Sorted keys by cell, aligning with the database point IDs below
	//keys
	errCode=cudaMemcpy(cellKey, dev_pointCellArr, sizeof(uint64_t)*(*DBSIZE), cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: pointCellArr memcpy Got error with code " << errCode << endl; 
	}
	//Point ids
	// errCode=cudaMemcpy(databaseIDValue, dev_databaseVal, sizeof(unsigned int)*(*DBSIZE), cudaMemcpyDeviceToHost);
	// if(errCode != cudaSuccess) {
	// cout << "\nError: databaseIDValue memcpy Got error with code " << errCode << endl; 
	// }

	//Point ids
	//indexLookupArr
	errCode=cudaMemcpy(indexLookupArr, dev_databaseVal, sizeof(unsigned int)*(*DBSIZE), cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: databaseIDValue memcpy Got error with code " << errCode << endl; 
	}


	//populate lookup array from grid cells into database (databaseIDValue above)


	//indexLookupArr
	// std::copy(databaseIDValue, databaseIDValue+(*DBSIZE), indexLookupArr);




	//Populate grid index
	//allocate memory for the index that will be sent to the GPU
	*index=new grid[numNonEmptyCells];
	


	//populate grid index

	//the first index
	(*index)[0].indexmin=0;

	uint64_t cnt=0;
	
	for (uint64_t i=1; i<(*DBSIZE); i++){
		
		if (cellKey[i-1]!=cellKey[i])
		{
			//grid index
			cnt++;
			(*index)[cnt].indexmin=i;			
			(*index)[cnt-1].indexmax=i-1;			

			
		}


		// (*index)[i].indexmin=tmpIndex[i].indexmin;
		// (*index)[i].indexmax=tmpIndex[i].indexmax;
		// (*gridCellLookupArr)[i].idx=i;
		// (*gridCellLookupArr)[i].gridLinearID=uniqueGridCellLinearIdsVect[i];
	}
	
	//the last index
	(*index)[numNonEmptyCells-1].indexmax=(*DBSIZE)-1;

	// for (int i=0; i<(*DBSIZE); i++)
	// {
	// 	printf("\npoint: %u, Cell: %llu",databaseIDValue[i],cellKey[i]);
	// }






	printf("\nFull cells: %d (%f, fraction full)",(unsigned int)numNonEmptyCells, numNonEmptyCells*1.0/double(totalCells));
	printf("\nEmpty cells: %ld (%f, fraction empty)",totalCells-(unsigned int)numNonEmptyCells, (totalCells-numNonEmptyCells*1.0)/double(totalCells));
	printf("\nSize of index that would be sent to GPU (GiB) -- (if full index sent), excluding the data lookup arr: %f", (double)sizeof(struct grid)*(totalCells)/(1024.0*1024.0*1024.0));
	printf("\nSize of compressed index to be sent to GPU (GiB) , excluding the data and grid lookup arr: %f", (double)sizeof(struct grid)*(numNonEmptyCells*1.0)/(1024.0*1024.0*1024.0));
	printf("\nWhen copying from entire index to compressed index: number of non-empty cells: %lu",numNonEmptyCells);


	
	free(DBSIZE);
	free(database);
	free(pointCellArrTmp);
	free(cellKey);
	// free(databaseIDValue);
	cudaFree(dev_DBSIZE);
	cudaFree(dev_database);
	cudaFree(dev_minArr);
	cudaFree(dev_nCells);
	cudaFree(dev_epsilon);
	cudaFree(dev_pointCellArr);
	cudaFree(dev_databaseVal);



}


