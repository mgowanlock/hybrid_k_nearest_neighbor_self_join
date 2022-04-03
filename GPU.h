#include "structs.h"
#include "params.h"


//index on the GPU
void populateNDGridIndexAndLookupArrayGPU(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE *epsilon, DTYPE* minArr, uint64_t totalCells, unsigned int * nCells, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr, unsigned int *nNonEmptyCells);

void distanceTableNDSortAndSearch(std::vector<std::vector<DTYPE> > * NDdataPoints, int * nearestNeighborTable, DTYPE * nearestNeighborTableDistances,
	std::vector<unsigned int> *queryPtsVect, DTYPE* epsilon,  unsigned int k_neighbors, 
	uint64_t * totalNeighbors, struct neighborTableLookup * neighborTable);

void distanceTableNDGridBatcheskNN(std::vector<std::vector<DTYPE> > * NDdataPoints, int * nearestNeighborTable, 
	DTYPE * nearestNeighborTableDistances,  double * totaldistance, 
	std::vector<unsigned int> *queryPtsVect, DTYPE* epsilon,  unsigned int k_neighbors, struct grid * index, 
	struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, DTYPE* minArr, unsigned int * nCells, 
	unsigned int * indexLookupArr, struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors, 
	uint64_t * totalNeighbors, CTYPE* workCounts);

//used when the index no longer provides enough selectivity
void distanceTableNDBruteForce(std::vector<std::vector<DTYPE> > * NDdataPoints, int * nearestNeighborTable, DTYPE * nearestNeighborTableDistances, 
	std::vector<unsigned int> *queryPtsVect, DTYPE* epsilon,  unsigned int k_neighbors, 
	uint64_t * totalNeighbors, struct neighborTableLookup * neighborTable);

unsigned long long callGPUBatchEst(unsigned int * DBSIZE, unsigned int N_QueryPts, unsigned int * dev_queryPts, unsigned int k_Neighbors, DTYPE* dev_database,  DTYPE* in_epsilon, DTYPE* dev_epsilon, struct grid * dev_grid, 
	unsigned int * dev_indexLookupArr, struct gridCellLookup * dev_gridCellLookupArr, DTYPE* dev_minArr, 
	unsigned int * dev_nCells, unsigned int * dev_nNonEmptyCells, unsigned int * retNumBatches, unsigned int * retGPUBufferSize);


// void constructNeighborTableKeyValueWithPtrskNN(int * pointIDKey, int * pointInDistValue, double * distancePointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, double * pointersToDistances, unsigned int * cnt);
void constructNeighborTableKeyValueWithPtrskNN(int * pointIDKey, int * pointInDistValue, DTYPE * distancePointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, DTYPE * pointersToDistances, unsigned int * cnt);

void warmUpGPU();

//for the brute force version without batches
void constructNeighborTableKeyValue(int * pointIDKey, int * pointInDistValue, struct table * neighborTable, unsigned int * cnt);



//gets average distance between points in the dataset
void sampleNeighborsBruteForce(std::vector<std::vector <DTYPE> > * NDdataPoints, DTYPE * epsilon_guess, unsigned int * bucket, unsigned int k_neighbors);

//frees pinned memory that has been reused
void cleanUpPinned();

//frees neighbortable data
void cleanUpNeighborTable();


void storeNeighborTableForkNNOnTheFly(int * pointIDKey, int * pointInDistValue, DTYPE * distancePointInDistValue, unsigned int * cnt, std::vector<std::vector<DTYPE> > * NDdataPoints,
			int k_Neighbors, std::vector<unsigned int> *queryPts, int * nearestNeighborTable, DTYPE * nearestNeighborTableDistances, double * totaldistance);