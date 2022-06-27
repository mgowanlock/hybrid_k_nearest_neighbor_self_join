
#ifndef STRUCTS_H
#define STRUCTS_H
#include <vector>
#include <stdio.h>
#include <iostream>
#include <semaphore.h>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "params.h"


//for outlier detection
//for each of the sum of distances we need to sort by distance to obtain the outlier scores
struct keyValPointDistStruct
{
	int pointID;
	DTYPE distance;
};

//for outlier detection
//for each point, we count the number of times it appears in another point's kNN set
struct keyValPointFrequencyKNNGraphStruct
{
	int pointID;
	uint64_t numTimesInOtherSet;
};

//used for sort-and-search brute force to sort in first dimension and map to original element id
struct databaseSortMap
{
	DTYPE data[GPUNUMDIM];
	unsigned int idx;
}; 

struct key_val_sort
{
		unsigned int pid; //point id
		DTYPE value_at_dim;
};




struct dim_reorder_sort
{
		unsigned int dim; //point dimension
		DTYPE variance; //variance of the points in this dimension
};

//tmp struct for splitting work between CPU and GPU
struct GPUQueryNumPts{
		unsigned int queriesGPU;
		unsigned long int pntsInCell;
};

struct workArray{
		unsigned int queryPntID;
		unsigned long int pntsInCell;
		
};



struct keyData{
		int key;
		int position;
		double distance;
};


//need to pass in the neighbortable thats an array of the dataset size.
//carry around a pointer to the array that has the points within epsilon though
struct neighborTableLookup
{
	int pointID;
	int indexmin;
	int indexmax;
	int * dataPtr;
	DTYPE * distancePtr;
};



//a struct that points to the arrays of individual data points within epsilon
//and the size of each of these arrays (needed to construct a subsequent neighbor table)
//will be used inside a vector.
struct neighborDataPtrs{
	int * dataPtr;
	DTYPE * distancePtr;
	int sizeOfDataArr;
};


//the result set:
// struct structresults{
// int pointID;
// int pointInDist;
// };


//the neighbortable.  The index is the point ID, each contains a vector
//only for the GPU Brute force implementation
struct table{
int pointID;
std::vector<int> neighbors;
};

//index lookup table for the GPU. Contains the indices for each point in an array
//where the array stores the direct neighbours of all of the points
struct gpulookuptable{
int indexmin;
int indexmax;
};

struct grid{	
int indexmin; //Contains the indices for each point in an array where the array stores the ids of the points in the grid
int indexmax;
};

//key/value pair for the gridCellLookup -- maps the location in an array of non-empty cells
struct gridCellLookup{	
unsigned int idx; //idx in the "grid" struct array
uint64_t gridLinearID; //The linear ID of the grid cell
//compare function for linearID
__host__ __device__ bool operator<(const gridCellLookup & other) const
  {
    return gridLinearID < other.gridLinearID;
  }
};

//neighbortable CPU -- indexmin and indexmax point to a single vector
struct neighborTableLookupCPU
{
	int pointID;
	int indexmin;
	int indexmax;
};



 



// struct compareThrust
// {
//   __host__ __device__
//   bool operator()(structresults const& lhs, structresults const& rhs)
//   {
//     if (lhs.pointID != rhs.pointID)
//     {
//         return (lhs.pointID < rhs.pointID);
//     }
//         return (lhs.pointInDist < rhs.pointInDist);
//   }
// };

//semaphores for index construction while kNN searching
//declared in main
// extern sem_t semIndex;
// extern sem_t semkNN;


//concurrent vector to store the memory addresses of
//the neighbortables that neeed to be stored and then freed


#endif
