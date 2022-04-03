#include "structs.h"
#include "params.h"

//functions for index on the GPU
__global__ void kernelIndexComputeNonemptyCells(DTYPE* database, unsigned int *N, DTYPE* epsilon, DTYPE* minArr, unsigned int * nCells, uint64_t * pointCellArr);
__global__ void kernelInitEnumerateDB(unsigned int * databaseVal, unsigned int *N);


//original when we just counted results
// __global__ void kernelBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, unsigned long long int * cnt, DTYPE* database, double * totalDistance);
__global__ void kernelBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, DTYPE* epsilon, unsigned long long int * cnt, DTYPE* database);

__global__ void kernelEstimateAvgDistBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, 
	unsigned long long int * cnt, DTYPE* database, double * total_distance);

__global__ void kernelKDistBruteForce(unsigned int *N, unsigned int * offset, unsigned int *debug1, unsigned int *debug2, 
	unsigned long long int * cnt, DTYPE* database, double * avg_distance, unsigned int * histogram, double * bucket_width);

//used when the index no longer provides any selectivity 
__global__ void kernelNDBruteForce(unsigned int *debug1, unsigned int *debug2, unsigned int *N, unsigned int * DBSIZE, 
	unsigned int * offset, unsigned int *batchNum, DTYPE* database, unsigned int * cnt, 
	int * pointIDKey, int * pointInDistVal, DTYPE * distancesKeyVal, unsigned int * queryPts, unsigned int * threadsForDistanceCalc, CTYPE* workCounts);

//used when the index no longer provides any selectivity -- database sorted on first dimension
__global__ void kernelNDBruteForceSortAndSearch(unsigned int *debug1, unsigned int *debug2, unsigned int *N, unsigned int * DBSIZE, unsigned int * DBmapping, DTYPE* epsilon, unsigned int * k_neighbors,
	unsigned int * offset, unsigned int *batchNum, DTYPE* database, unsigned int * cnt, 
	int * pointIDKey, int * pointInDistVal, DTYPE * distancesKeyVal, unsigned int * queryPts, unsigned int * threadsForDistanceCalc, CTYPE* workCounts);

__global__ void kernelNDGridIndexBatchEstimatorOLD(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
	unsigned int * sampleOffset, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
	unsigned int * nNonEmptyCells,  unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets);

//with expansion of the neighboring cells
__global__ void kernelNDGridIndexBatchEstimatorWithExpansion(unsigned int *debug1, unsigned int *debug2, unsigned int * queryPts, unsigned int * iteration, unsigned int *N,  
	unsigned int * sampleOffset, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
	unsigned int * nNonEmptyCells,  unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets);

//same as the original but with a query set
__global__ void kernelNDGridIndexBatchEstimatorQuerySet(unsigned int *debug1, unsigned int *debug2, unsigned int * threadsForDistanceCalc, unsigned int * queryPts,unsigned int *N,  
	unsigned int * sampleOffset, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
	unsigned int * nNonEmptyCells);

// __global__ void kernelNDGridIndexGlobalkNN(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
// 	unsigned int * offset, unsigned int *batchNum, DTYPE * database, DTYPE *epsilon, struct grid * index, unsigned int * indexLookupArr, 
// 	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
// 	unsigned int * nNonEmptyCells,  unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets,
// 	int * pointIDKey, int * pointInDistVal, double * distancesKeyVal, CTYPE* workCounts);

// __global__ void kernelNDGridIndexGlobalkNNSubsequentExecution(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
// 	unsigned int * offset, unsigned int *batchNum, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
// 	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
// 	unsigned int * nNonEmptyCells,  unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets,
// 	int * pointIDKey, int * pointInDistVal, DTYPE * distancesKeyVal, unsigned int * queryPts, unsigned int * threadsForDistanceCalc, CTYPE* workCounts);


__global__ void kernelNDGridIndexGlobalkNN(unsigned int *debug1, unsigned int *debug2, unsigned int * k_neighbors, unsigned int *N,  
	unsigned int * offset, unsigned int *batchNum, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
	unsigned int * nNonEmptyCells, int * pointIDKey, int * pointInDistVal, DTYPE * distancesKeyVal, unsigned int * queryPts, unsigned int * threadsForDistanceCalc, CTYPE* workCounts);

__device__ uint64_t getLinearID_nDimensionsGPU(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions);

__global__ void kernelSortPointsInCells(DTYPE* database, struct grid * index, unsigned int* indexLookupArr, unsigned int nNonEmptyCells);


__device__ unsigned long int evaluateNumPointsPerCell(unsigned int* nCells, unsigned int* indexes, struct gridCellLookup * gridCellLookupArr, 
	unsigned int* nNonEmptyCells, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, DTYPE* point, 
	unsigned int* cnt, int* pointIDKey, int* pointInDistVal, DTYPE * distancesKeyVal, int pointIdx, bool differentCell, unsigned int* nDCellIDs, 
	CTYPE* workCounts, unsigned int * threadsForDistanceCalc, unsigned int * tid);