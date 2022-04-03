

#include "kernel.h"
#include "structs.h"
#include <math.h>	
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

#include "params.h"







__device__ void evaluateCell(unsigned int* nCells, unsigned int* indexes, struct gridCellLookup * gridCellLookupArr, unsigned int* nNonEmptyCells, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, unsigned int* cnt,int* pointIDKey, int* pointInDistVal, DTYPE* distancesKeyVal, int pointIdx, unsigned int* nDCellIDs, CTYPE* workCounts, unsigned int * threadsForDistanceCalc, unsigned int * tid);
__forceinline__ __device__ void evalPointWithoutEpsilon(unsigned int* indexLookupArr, int k, DTYPE* database, DTYPE* point, unsigned int* cnt, int* pointIDKey, int* pointInDistVal, DTYPE * distancesKeyVal, int pointIdx, bool differentCell);



/////////////////////////////////////////
//THE RESULTS GET GENERATED AS KEY/VALUE PAIRS IN TWO ARRAYS
//KEY- THE POINT ID BEING SEARCHED
//VALUE- A POINT ID WITHIN EPSILON OF THE KEY POINT THAT WAS SEARCHED
//THE RESULTS ARE SORTED IN SITU ON THE DEVICE BY THRUST AFTER THE KERNEL FINISHES
/////////////////////////////////////////



__device__ uint64_t getLinearID_nDimensionsGPU(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions) {
    // int i;
    // uint64_t offset = 0;
    // for( i = 0; i < nDimensions; i++ ) {
    //     offset += (uint64_t)pow(dimLen[i],i) * (uint64_t)indexes[nDimensions - (i + 1)];
    // }
    // return offset;

    uint64_t offset = 0;
	uint64_t multiplier = 1;
	for (int i = 0; i<nDimensions; i++){
  	offset += (uint64_t)indexes[i] * multiplier;
  	multiplier *= dimLen[i];
	}

	return offset;
}






//kNN:
//double * distancesKeyVal - distances stored with the key value pairs
//unsigned int * queryPts - query point ids to be processed
//N is now the number of query points per batch

//We now attempt to compute the kNN for each point without restarting the ones that don't
//have k neighbors, by looking at the other adjacent cells
__global__ void kernelNDGridIndexGlobalkNN(unsigned int *debug1, unsigned int *debug2, unsigned int * k_neighbors, unsigned int *N,  
	unsigned int * offset, unsigned int *batchNum, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
	unsigned int * nNonEmptyCells,  int * pointIDKey, int * pointInDistVal, DTYPE * distancesKeyVal, unsigned int * queryPts, unsigned int * threadsForDistanceCalc, CTYPE* workCounts)
{


//query id
unsigned int threadsDistCalcReg=(*threadsForDistanceCalc);
unsigned int qid=(threadIdx.x+ (blockIdx.x*BLOCKSIZE))/threadsDistCalcReg; //each additional iteration we add a 
																					//number of threads for doing distance 						
																					//calculations	
if (qid>=*N){
	return;
}


//thread id
unsigned int tid=(threadIdx.x+ (blockIdx.x*BLOCKSIZE)); 
unsigned int pointIdx=queryPts[qid*(*offset)+(*batchNum)]; 
unsigned int pointOffset=pointIdx*(GPUNUMDIM); 

unsigned int nDCellIDs[NUMINDEXEDDIM];
unsigned int nDMinCellIDs[NUMINDEXEDDIM];
unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
unsigned int indexes[NUMINDEXEDDIM];
unsigned int loopRng[NUMINDEXEDDIM];


for (int i=0; i<NUMINDEXEDDIM; i++){
	nDCellIDs[i]=(database[pointOffset+i]-minArr[i])/(*epsilon);
	nDMinCellIDs[i]=max(0,(int)nDCellIDs[i]-(int)1); //boundary conditions (don't go beyond cell 0) 
																		//cast to int so we don't roll over into a high positive unsigned int value
	nDMaxCellIDs[i]=min(nCells[i]-1,nDCellIDs[i]+1); //boundary conditions (don't go beyond the maximum number of cells)

}

    //computation
	for (loopRng[0]=nDMinCellIDs[0]; loopRng[0]<=nDMaxCellIDs[0]; loopRng[0]++)
	for (loopRng[1]=nDMinCellIDs[1]; loopRng[1]<=nDMaxCellIDs[1]; loopRng[1]++)
	#include "kernelloops.h"					
	{ //beginning of loop body
	
	for (int x=0; x<NUMINDEXEDDIM; x++){
	indexes[x]=loopRng[x];	
	}
		evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr, cnt, pointIDKey, pointInDistVal, distancesKeyVal, pointIdx, nDCellIDs, workCounts, &threadsDistCalcReg, &tid);
	
	} //end loop body

	

}


__forceinline__ __device__ void evalPoint(unsigned int* indexLookupArr, int k, DTYPE* database, DTYPE* epsilon, 
	unsigned int* cnt, int* pointIDKey, int* pointInDistVal, DTYPE * distancesKeyVal, int pointIdx) {


	DTYPE runningTotalDist=0;
	unsigned int dataIdx=indexLookupArr[k];

	#if SHORTCIRCUIT==1
	bool exitFlag=0;
	#endif

        for (int l=0; l<GPUNUMDIM; l++){
         runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-database[pointIdx*GPUNUMDIM+l])*
         (database[dataIdx*GPUNUMDIM+l]-database[pointIdx*GPUNUMDIM+l]);
          #if SHORTCIRCUIT==1
          if ((runningTotalDist>(((*epsilon)*(*epsilon))))) { 
          	  exitFlag=1;                                                 
              break;													  
          }
          #endif
        }
       
        
        #if SHORTCIRCUIT==1
        if (exitFlag==0)
        {
        #endif
        	
        runningTotalDist=sqrt(runningTotalDist);

        
	        if (runningTotalDist<=(*epsilon)){
	          unsigned int idx=atomicAdd(cnt,int(1));
	          pointIDKey[idx]=pointIdx;
	          pointInDistVal[idx]=dataIdx;
	          distancesKeyVal[idx]=runningTotalDist;
			}

		#if SHORTCIRCUIT==1	
		}
		#endif
	



}

__forceinline__ __device__ void evalPointWithoutEpsilon(unsigned int* indexLookupArr, int k, DTYPE* database, DTYPE* point, unsigned int* cnt, int* pointIDKey, int* pointInDistVal, DTYPE * distancesKeyVal, int pointIdx, bool differentCell) {


	DTYPE runningTotalDist=0;
	// unsigned int dataIdx=indexLookupArr[k];

    for (int l=0; l<GPUNUMDIM; l++){
      runningTotalDist+=(database[indexLookupArr[k]*GPUNUMDIM+l]-point[l])*(database[indexLookupArr[k]*GPUNUMDIM+l]-point[l]);
    }
    unsigned int idx=atomicAdd(cnt,int(1));
    pointIDKey[idx]=pointIdx;
    pointInDistVal[idx]=indexLookupArr[k];
    distancesKeyVal[idx]=sqrt(runningTotalDist);

}


__device__ void evaluateCell(unsigned int* nCells, unsigned int* indexes, struct gridCellLookup * gridCellLookupArr, 
	unsigned int* nNonEmptyCells, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	 unsigned int* cnt, int* pointIDKey, int* pointInDistVal, DTYPE * distancesKeyVal, int pointIdx, 
	unsigned int* nDCellIDs, CTYPE* workCounts, unsigned int * threadsForDistanceCalc, unsigned int * tid) {


#if COUNTMETRICS == 1
			atomicAdd(&workCounts[1],int(1));
#endif

    uint64_t calcLinearID=getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
    //compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says 
    //a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

    struct gridCellLookup tmp;
    tmp.gridLinearID=calcLinearID;
    //find if the cell is non-empty
    if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr+ (*nNonEmptyCells), gridCellLookup(tmp))){


    	//compute the neighbors for the adjacent non-empty cell
    	struct gridCellLookup * resultBinSearch=thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
    	unsigned int GridIndex=resultBinSearch->idx;
	




		//here, we have multiple threads perform evalPoint as a function of the iteration
		//more threads for decreasing workload sizes
		for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k+=(*threadsForDistanceCalc)){ 
		
		int pntid=k+(*tid%(*threadsForDistanceCalc));

		if (pntid<=index[GridIndex].indexmax)
		{
		evalPoint(indexLookupArr, pntid, database, epsilon, cnt, pointIDKey, pointInDistVal, distancesKeyVal, pointIdx);
		}
		#if COUNTMETRICS == 1
			atomicAdd(&workCounts[0],1);
		#endif
    	} // end for loop

	} //end if statement

}







__device__ unsigned long int evaluateNumPointsPerCell(unsigned int* nCells, unsigned int* indexes, struct gridCellLookup * gridCellLookupArr, unsigned int* nNonEmptyCells, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, DTYPE* point, unsigned int* cnt, int* pointIDKey, int* pointInDistVal, DTYPE * distancesKeyVal, int pointIdx, bool differentCell, unsigned int* nDCellIDs, CTYPE* workCounts, unsigned int * threadsForDistanceCalc, unsigned int * tid) 
{

        uint64_t calcLinearID=getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
        //compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says 
        //a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

        struct gridCellLookup tmp;
        tmp.gridLinearID=calcLinearID;
        //find if the cell is non-empty
        if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr+ (*nNonEmptyCells), gridCellLookup(tmp))){

                //compute the neighbors for the adjacent non-empty cell
                struct gridCellLookup * resultBinSearch=thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
				return (unsigned long int)((index[resultBinSearch->idx].indexmax-index[resultBinSearch->idx].indexmin)+1);
		}

		return 0;

}


//Kernel brute forces to generate the neighbor table for each point in the database
__global__ void kernelBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, DTYPE* epsilon, 
	unsigned long long int * cnt, DTYPE* database)
{
unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

if (tid>=*N){
	return;
}


int dataOffset=tid*GPUNUMDIM;
DTYPE runningDist=0;
//compare my point to every other point
for (int i=0; i<(*N); i++)
{
	runningDist=0;
	for (int j=0; j<GPUNUMDIM; j++){
		runningDist+=(database[(i*GPUNUMDIM)+j]-database[dataOffset+j])*(database[(i*GPUNUMDIM)+j]-database[dataOffset+j]);
	}

	//if within epsilon:
	if ((sqrt(runningDist))<=(*epsilon)){
		atomicAdd(cnt, (unsigned long long int)1);
	}
}


return;
}

//Kernel brute forces to generate the neighbor table for each point in the database
//return the total distance to prevent the compiler from optimizing out the distance calculation

//original brute force when we only counted the results
/*
__global__ void kernelBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, unsigned long long int * cnt, DTYPE* database, double * totalDistance) {

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

if (tid>=*N){
	return;
}

unsigned long long int cntDistanceCalculations=0;
double totalDistanceThread=0;


int dataOffset=tid*GPUNUMDIM;
double runningDist=0;
//compare my point to every other point
for (int i=0; i<(*N); i++)
{
	runningDist=0;
	for (int j=0; j<GPUNUMDIM; j++){
		runningDist+=(database[(i*GPUNUMDIM)+j]-database[dataOffset+j])*(database[(i*GPUNUMDIM)+j]-database[dataOffset+j]);
	}	

	totalDistanceThread+=runningDist;		
	cntDistanceCalculations++;	
}

// each thread adds its total number of distance calculations

atomicAdd(cnt, (unsigned long long int)cntDistanceCalculations);
atomicAdd(totalDistance, (double)totalDistanceThread);


return;
}
*/


// __device__ double atomicAdd(double* address, double val)
// {
//     unsigned long long int* address_as_ull =
//                           (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;
//     do {
//         assumed = old;
//             old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +
//                                __longlong_as_double(assumed)));
//     } while (assumed != old);
//     return __longlong_as_double(old);
// }


//Kernel brute forces to estimate the average distance between points
__global__ void kernelEstimateAvgDistBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, 
	unsigned long long int * cnt, DTYPE* database, double * total_distance)
{
//1% of the points for searching	
unsigned int tid=(threadIdx.x+ (blockIdx.x*BLOCKSIZE))*1000; 

if (tid>=*N){
	return;
}



int dataOffset=tid*GPUNUMDIM;
//compare my point to every other point sampled - 0.1%
for (int i=0+threadIdx.x; i<(*N); i+=1000)
{
	double runningDist=0;
	for (int j=0; j<GPUNUMDIM; j++){
		runningDist+=(database[(i*GPUNUMDIM)+j]-database[dataOffset+j])*(database[(i*GPUNUMDIM)+j]-database[dataOffset+j]);
	}

	runningDist=sqrt(runningDist);

	atomicAdd(cnt, (unsigned long long int)1);
	atomicAdd(total_distance, (double)runningDist);


	// //if within epsilon:
	// if ((sqrt(runningDist))<=(*epsilon)){
		
		// unsigned int idx=atomicInc(cnt, uint64_t(1));
		// pointIDKey[idx]=tid;
		// pointInDistVal[idx]=i;
		// }
}


return;
}


/*

//Kernel brute force to get the neighbors that are tough to get with the index
//I.e., when there's less than 20% of the points left, we execute this
//We execute 1 warp per point, where each warp compares its point to all other points in the dataset
//Inflate the number of threads per point since the batch size will be large and we don't want small block sizes 
__global__ void kernelRemainderBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, 
	unsigned long long int * cnt, DTYPE* database, double * epsilon, unsigned int * queryPts, int * pointIDKey, int * pointInDistVal, double * distancesKeyVal)
{

unsigned int tid=((blockIdx.x*BLOCKSIZE)+threadIdx.x)/32;

if (tid>=*N){
	return;
}



unsigned int pointIdx=queryPts[tid];
unsigned int dataOffset=pointIdx*GPUNUMDIM;


//compare my point to every other point
for (int i=0; i<(*N); i+=32)
{
	double runningDist=0;
	unsigned int dataIdx=i+threadIdx.x;
	for (int j=0; j<GPUNUMDIM; j++){
		runningDist+=(database[(dataIdx*GPUNUMDIM)+j]-database[dataOffset+j])*(database[(dataIdx*GPUNUMDIM)+j]-database[dataOffset+j]);
	
	 		#if SHORTCIRCUIT==1
	          if (sqrt(runningDist)>(*epsilon)) {
	              break;
	          }
	          #endif
	}

	runningDist=sqrt(runningDist);

	
	if (runningDist<*epsilon)
	{

		  unsigned int idx=atomicAdd(cnt,int(1));
          pointIDKey[idx]=pointIdx;
          pointInDistVal[idx]=dataIdx;
          distancesKeyVal[idx]=runningDist;
	}

}


return;
}
*/


//Kernel brute force to estimate the epsilon value needed to get the k points 
//makes a histogram of neighbors vs distance
//Gets all the neighbors for a selected few points

//avg_distance -- do not include distances passed this one, which is the average distance between two
//points in a dataset (much too large to be useful)
//histogram -- array data in buckets
//bucket_width -- determines which index the distance falls into
//offset- sample each dataset with a variable number of points based on the offset
__global__ void kernelKDistBruteForce(unsigned int *N, unsigned int * offset, unsigned int *debug1, unsigned int *debug2, 
	unsigned long long int * cnt, DTYPE* database, double * avg_distance, unsigned int * histogram, double * bucket_width)
{
//1% of the points for searching	
// unsigned int tid=(threadIdx.x+ (blockIdx.x*BLOCKSIZE))*100; 

//each block works on the same point
// unsigned int tid=(blockIdx.x*BLOCKSIZE);
// unsigned int tid=(blockIdx.x)*(300);
unsigned int tid=(blockIdx.x)*(*offset);

if (tid>=*N){
	return;
}




int dataOffset=tid*GPUNUMDIM;

//used for sampling when comparing the points to all other points in the database
int samplePnts=8;

//compare my point to every other point sampled - 0.1%
// for (int i=0+threadIdx.x; i<(*N); i+=1000)

// unsigned int numpts=*N/1000;

// unsigned int minRng=max(0,tid-(numpts/2));
// unsigned int maxRng=min((*N)-1,tid+(numpts/2));

// for (int i=minRng; i<maxRng; i++)


//compare my point to every samplePnts block of points
// for (int i=0; i<(*N); i+=BLOCKSIZE)
for (int i=0; i<(*N); i+=BLOCKSIZE*samplePnts)
{
	double runningDist=0;
	// int pntID=i+threadIdx.x;
	int pntID=i+threadIdx.x;
	for (int j=0; j<GPUNUMDIM; j++){
		runningDist+=(database[(pntID*GPUNUMDIM)+j]-database[dataOffset+j])*(database[(pntID*GPUNUMDIM)+j]-database[dataOffset+j]);
	
		if (sqrt(runningDist)>*avg_distance)
		{
			break;
		}
	}

	runningDist=sqrt(runningDist);

	//don't let a point count itself 
	if (runningDist<*avg_distance && pntID!=tid)
	{
	// atomicAdd(cnt, (unsigned long long int)1);
	// atomicAdd(total_distance, (double)runningDist);

		unsigned int bucket=runningDist/(*bucket_width);
		atomicAdd(histogram+bucket, (unsigned int)1*samplePnts);
	}

	// //if within epsilon:
	// if ((sqrt(runningDist))<=(*epsilon)){
		
		// unsigned int idx=atomicInc(cnt, uint64_t(1));
		// pointIDKey[idx]=tid;
		// pointInDistVal[idx]=i;
		// }
}


return;
}


/*
__global__ void kernelNDGridIndexBatchEstimatorWithExpansion(unsigned int *debug1, unsigned int *debug2, unsigned int * queryPts, unsigned int * iteration, unsigned int *N,  
	unsigned int * sampleOffset, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
	unsigned int * nNonEmptyCells,  unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets)
{

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

if (tid>=*N){
	return;
}


//from the actual kernel
//unsigned int pointIdx=queryPts[tid*(*offset)+(*batchNum)];

//original without the query point 
// unsigned int pointID=tid*(*sampleOffset)*(GPUNUMDIM);

//modified to account for query points not total dataset

unsigned int queryPntIdx=queryPts[tid*(*sampleOffset)];
unsigned int pointID=queryPntIdx*(GPUNUMDIM);

// if (tid<512)
// printf("\ntid: %u, queryPntIdx: %u, Array idx: %u",tid, queryPntIdx, pointID);

//make a local copy of the point
DTYPE point[GPUNUMDIM];
for (int i=0; i<GPUNUMDIM; i++){
	point[i]=database[pointID+i];	
}


unsigned int regIter=*iteration+1; //need to add 1 because otherwise we won't look at the neighboring cells

DTYPE epsilon_w_iter=(*epsilon)*regIter;


//calculate the coords of the Cell for the point
//and the min/max ranges in each dimension
unsigned int nDCellIDs[NUMINDEXEDDIM];
unsigned int nDMinCellIDs[NUMINDEXEDDIM];
unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
for (int i=0; i<NUMINDEXEDDIM; i++){
	nDCellIDs[i]=(point[i]-minArr[i])/(*epsilon);
	nDMinCellIDs[i]=max(0,int(nDCellIDs[i]-regIter)); //boundary conditions (don't go beyond cell 0) // set to int so it can be evaluated against a negative number 
														//(otherwise unsigned int wrap around)
	nDMaxCellIDs[i]=min(nCells[i]-1,nDCellIDs[i]+regIter); //boundary conditions (don't go beyond the maximum number of cells)

	// if (tid==0 && regIter==1)
	// {
	// printf("\n min/max cells: %u,%u",nDMinCellIDs[i], nDMaxCellIDs[i]);
	// }

}




///////////////////////////
//Take the intersection of the ranges for each dimension between
//the point and the filtered set of cells in each dimension 
//Ranges in a given dimension that have points in them that are non-empty in a dimension will be tested 
///////////////////////////

unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];
//compare the point's range of cell IDs in each dimension to the filter mask
//only 2 possible values (you always find the middle point in the range), because that's the cell of the point itself
bool foundMin=0;
bool foundMax=0;

//we go through each dimension and compare the range of the query points min/max cell ids to the filtered ones
	//find out which ones in the range exist based on the min/max
	//then determine the appropriate ranges
	for (int i=0; i<NUMINDEXEDDIM; i++)
	{
		foundMin=0;
		foundMax=0;
		
		// cases:
		// found the min and max
		// found the min and not max
		//found the max and not the min
		//you don't find the min or max -- then only check the mid
		//you always find the mid because it's in the cell of the point you're looking for
		
		//for expanding the number of cells searched, need to find the left and right one that might exist
		for (int k=0; k<regIter; k++)
		{	
			if (foundMin==0)
			{
				if(thrust::binary_search(thrust::seq, gridCellNDMask+ gridCellNDMaskOffsets[(i*2)],
					gridCellNDMask+ gridCellNDMaskOffsets[(i*2)+1]+1,nDMinCellIDs[i]+k)){ //extra +1 here is because we include the upper bound
					foundMin=1;
					rangeFilteredCellIdsMin[i]=nDMinCellIDs[i];
				}
				else{
				rangeFilteredCellIdsMin[i]=nDMinCellIDs[i]+1;	
				}
			}
			

			if (foundMax==0)
			{
				if(thrust::binary_search(thrust::seq, gridCellNDMask+ gridCellNDMaskOffsets[(i*2)],
					gridCellNDMask+ gridCellNDMaskOffsets[(i*2)+1]+1,nDMaxCellIDs[i]-k)){ //extra +1 here is because we include the upper bound
					foundMax=1;
					rangeFilteredCellIdsMax[i]=nDMaxCellIDs[i];
				}
				else{
				rangeFilteredCellIdsMax[i]=nDMinCellIDs[i]+1;
				}
			}
			

				
		}

		// if (tid==0 && regIter==1)
		// 		{
		// 		printf("\n min/max cells filtered: %u,%u",rangeFilteredCellIdsMin[i], rangeFilteredCellIdsMax[i]);
		// 		}



		
	}		


	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////	
	



	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0]=rangeFilteredCellIdsMin[0]; loopRng[0]<=rangeFilteredCellIdsMax[0]; loopRng[0]++)
	for (loopRng[1]=rangeFilteredCellIdsMin[1]; loopRng[1]<=rangeFilteredCellIdsMax[1]; loopRng[1]++)
	#include "kernelloops.h"						
	{ //beginning of loop body
	

	#if COUNTMETRICS == 1
			atomicAdd(debug1,int(1));
	#endif	

	for (int x=0; x<NUMINDEXEDDIM; x++){
	indexes[x]=loopRng[x];	
	// if (tid==0)
	// 	printf("\ndim: %d, indexes: %d",x, indexes[x]);
	}
	

	
	uint64_t calcLinearID=getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
	//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says 
	//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

	//CHANGE THIS TO A BINARY SEARCH LATER	

	// for (int x=0; x<(*nNonEmptyCells);x++){
	// 	if (calcLinearID==gridCellLookupArr[x].gridLinearID){
	// 		cellsToCheck->push_back(calcLinearID); 
	// 	}
	// }
	
	// printf("\ntid: %d, Linear id: %d",tid,calcLinearID);

	struct gridCellLookup tmp;
	tmp.gridLinearID=calcLinearID;
	if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr+ (*nNonEmptyCells), gridCellLookup(tmp))){
		//in the GPU implementation we go directly to computing neighbors so that we don't need to
		//store a buffer of the cells to check 
		//cellsToCheck->push_back(calcLinearID); 

		//HERE WE COMPUTE THE NEIGHBORS FOR THE CELL
		//XXXXXXXXXXXXXXXXXXXXXXXXX
		


		
		struct gridCellLookup * resultBinSearch=thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex=resultBinSearch->idx;



		//original when estimating the neighbors within epsilon
		for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
				DTYPE runningTotalDist=0;
				unsigned int dataIdx=indexLookupArr[k];

				

				for (int l=0; l<GPUNUMDIM; l++){
				runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
					
					#if SHORTCIRCUIT==1
					if ((sqrt(runningTotalDist))>epsilon_w_iter)
					{
						break;
					}
					#endif
				}


				if (sqrt(runningTotalDist)<=epsilon_w_iter){
					unsigned int idx=atomicAdd(cnt,int(1));
				}
			}
		

		//just return the number of point in each cell
		// unsigned int pointsInCell=index[GridIndex].indexmax-index[GridIndex].indexmin+1;
		// atomicAdd(cnt,pointsInCell);

	}

	
	//printf("\nLinear id: %d",calcLinearID);
	} //end loop body

} //end function
*/


//CONVERTING THIS FOR MULTIPLE THREADS PER POINT


__global__ void kernelNDGridIndexBatchEstimatorQuerySet(unsigned int *debug1, unsigned int *debug2, unsigned int * threadsForDistanceCalc, unsigned int * queryPts,unsigned int *N,  
	unsigned int * sampleOffset, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
	unsigned int * nNonEmptyCells)
{

// unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

//query id	
unsigned int qid=(threadIdx.x+ (blockIdx.x*BLOCKSIZE))/(*threadsForDistanceCalc);



if (qid>=*N){
	return;
}



//thread id
unsigned int tid=(threadIdx.x+ (blockIdx.x*BLOCKSIZE)); 



//original
// unsigned int pointID=tid*(*sampleOffset)*(GPUNUMDIM);



//original
// unsigned int queryPntIdx=queryPts[tid*(*sampleOffset)];
// unsigned int pointID=queryPntIdx*(GPUNUMDIM);

//with a number of threads for distance calculations
unsigned int queryPntIdx=queryPts[qid*(*sampleOffset)];
unsigned int pointID=queryPntIdx*(GPUNUMDIM);

//make a local copy of the point
DTYPE point[GPUNUMDIM];
for (int i=0; i<GPUNUMDIM; i++){
	point[i]=database[pointID+i];	
}

//calculate the coords of the Cell for the point
//and the min/max ranges in each dimension
unsigned int nDCellIDs[NUMINDEXEDDIM];
unsigned int nDMinCellIDs[NUMINDEXEDDIM];
unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
unsigned int indexes[NUMINDEXEDDIM];
unsigned int loopRng[NUMINDEXEDDIM];
for (int i=0; i<NUMINDEXEDDIM; i++){
	nDCellIDs[i]=(point[i]-minArr[i])/(*epsilon);
	nDMinCellIDs[i]=max(0,nDCellIDs[i]-1); //boundary conditions (don't go beyond cell 0)
	nDMaxCellIDs[i]=min(nCells[i]-1,nDCellIDs[i]+1); //boundary conditions (don't go beyond the maximum number of cells)

}


	//computation
	for (loopRng[0]=nDMinCellIDs[0]; loopRng[0]<=nDMaxCellIDs[0]; loopRng[0]++)
	for (loopRng[1]=nDMinCellIDs[1]; loopRng[1]<=nDMaxCellIDs[1]; loopRng[1]++)
	#include "kernelloops.h"						
	{ //beginning of loop body
	

	#if COUNTMETRICS == 1
			atomicAdd(debug1,int(1));
	#endif	

	for (int x=0; x<NUMINDEXEDDIM; x++){
	indexes[x]=loopRng[x];	
	// if (tid==0)
	// 	printf("\ndim: %d, indexes: %d",x, indexes[x]);
	}
	

	
	uint64_t calcLinearID=getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);


	struct gridCellLookup tmp;
	tmp.gridLinearID=calcLinearID;
	if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr+ (*nNonEmptyCells), gridCellLookup(tmp))){
		//in the GPU implementation we go directly to computing neighbors so that we don't need to
		//store a buffer of the cells to check 
		//cellsToCheck->push_back(calcLinearID); 

		//HERE WE COMPUTE THE NEIGHBORS FOR THE CELL
		//XXXXXXXXXXXXXXXXXXXXXXXXX
		


		
		struct gridCellLookup * resultBinSearch=thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex=resultBinSearch->idx;

		for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k+=(*threadsForDistanceCalc)){
				DTYPE runningTotalDist=0;

				int pntid=k+(tid%(*threadsForDistanceCalc));
				unsigned int dataIdx=indexLookupArr[pntid];

				
				if (pntid<=index[GridIndex].indexmax)
				{
					for (int l=0; l<GPUNUMDIM; l++){
					runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
					
					  #if SHORTCIRCUIT==1
			          if (sqrt(runningTotalDist)>(*epsilon)) {
			              break;
			          }
			          #endif

					}


					if (sqrt(runningTotalDist)<=(*epsilon)){
						unsigned int idx=atomicAdd(cnt,int(1));
						// pointIDKey[idx]=tid;
						// pointInDistVal[idx]=i;
						//neighborTableCPUPrototype[queryPoint].neighbors.push_back(dataIdx);

					}
				}
			}



	}

	} //end loop body

}





//N- query set size for the batch

//used when the index no longer provides any selectivity
__global__ void kernelNDBruteForce(unsigned int *debug1, unsigned int *debug2, unsigned int *N, unsigned int * DBSIZE, 
	unsigned int * offset, unsigned int *batchNum, DTYPE* database, unsigned int * cnt, 
	int * pointIDKey, int * pointInDistVal, DTYPE * distancesKeyVal, unsigned int * queryPts, unsigned int * threadsForDistanceCalc, CTYPE* workCounts)
{



//query id
unsigned int threadsDistCalcReg=(*threadsForDistanceCalc);
unsigned int qid=(threadIdx.x+ (blockIdx.x*BLOCKSIZE))/threadsDistCalcReg; //each additional iteration we add a 



		


if (qid>=*N){
	return;
}

//thread id
unsigned int tid=(threadIdx.x+ (blockIdx.x*BLOCKSIZE)); 
unsigned int pointIdx=queryPts[qid*(*offset)+(*batchNum)];  



//make a local copy of the point
DTYPE point[GPUNUMDIM];
for (int i=0; i<GPUNUMDIM; i++){
	point[i]=database[pointIdx*(GPUNUMDIM)+i];	
}



	DTYPE runningTotalDist=0;


	for (unsigned int i=0; i<(*DBSIZE); i+=threadsDistCalcReg)
	{		
		runningTotalDist=0;

		//original
		// unsigned int dataIdx=i+tid;

		//modified
		unsigned int dataIdx=i+(tid%threadsDistCalcReg);		
		
		

		//to ensure that the thread doesn't go past the database size
		if (dataIdx<(*DBSIZE))
		{
	        for (int l=0; l<GPUNUMDIM; l++){
	          runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
	        }
	        

	        	
	        runningTotalDist=sqrt(runningTotalDist);

	    
		          unsigned int idx=atomicAdd(cnt,int(1));
		          pointIDKey[idx]=pointIdx;
		          pointInDistVal[idx]=dataIdx;
		          distancesKeyVal[idx]=runningTotalDist;
	
	    }      
	}





}



//used when the index no longer provides any selectivity
//not quite brute force: sorts dataset in first dimension
//DBmapping- contains the mapping of the original data id and the sorted data id
__global__ void kernelNDBruteForceSortAndSearch(unsigned int *debug1, unsigned int *debug2, unsigned int *N, unsigned int * DBSIZE, unsigned int * DBmapping, DTYPE* epsilon, unsigned int * k_neighbors, 
	unsigned int * offset, unsigned int *batchNum, DTYPE* database, unsigned int * cnt, 
	int * pointIDKey, int * pointInDistVal, DTYPE * distancesKeyVal, unsigned int * queryPts, unsigned int * threadsForDistanceCalc, CTYPE* workCounts)
{



//query id
unsigned int threadsDistCalcReg=(*threadsForDistanceCalc);
unsigned int qid=(threadIdx.x+ (blockIdx.x*BLOCKSIZE))/threadsDistCalcReg; //QUERY ID



		


if (qid>=*N){
	return;
}

//thread id
unsigned int tid=(threadIdx.x+ (blockIdx.x*BLOCKSIZE)); 
unsigned int pointIdx=queryPts[qid*(*offset)+(*batchNum)];  



//make a local copy of the point
DTYPE point[GPUNUMDIM];
for (int i=0; i<GPUNUMDIM; i++){
	// point[i]=database[pointIdx*(GPUNUMDIM)+i]; //original before DB mapping
	point[i]=database[DBmapping[pointIdx]*(GPUNUMDIM)+i];	
}

	//to start,
	//want to use the data in the database that's +/- 2*epsilon (original epsilon estimation yielded too few points)
	DTYPE distmin=point[0];
	DTYPE distmax=point[0];


	DTYPE runningTotalDist=0;
	DTYPE eps=(*epsilon);

	//counter that counts the numbero f results within epsilon
	__shared__ unsigned int cntResults;

	cntResults=0;



	//////////////////
	//First, we keep expanding epsilon until we get at least kNN within the distance
	//We compute distmin and distmax here
	//Then we store the results
	/////////////////

	//keep expanding epsilon until we get at least kNN results
	while (cntResults<(*k_neighbors))
	{
		eps=eps*2.0;
		distmin=distmin-(eps); //start by trying 2x the epsilon that failed to find the knn from the kernel with the index
		distmax=distmax+(eps); //start by trying 2x the epsilon that failed to find the knn from the kernel with the index
		cntResults=0;


		__syncthreads();
		

		//count increasing to dist max
		for (unsigned int i=DBmapping[pointIdx]; i<(*DBSIZE); i+=threadsDistCalcReg)
		{

			//modified
			unsigned int dataIdx=i+(tid%threadsDistCalcReg);		



			//break out of the loop if we exceed distmax or the data id>=DBsize
			if (((database[dataIdx*GPUNUMDIM])>distmax) || (dataIdx>=(*DBSIZE)))
			{
				break;
			}

			runningTotalDist=0;

			#pragma unroll
			for (int l=0; l<GPUNUMDIM; l++){
		          runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
		        }
		        runningTotalDist=sqrt(runningTotalDist);	

		        if (runningTotalDist<=eps)
		        {
		        	// cntResults++;
		        	atomicInc(&cntResults,(unsigned int)1);
		        }
		}


		//count decreasing to dist min
		for (int i=DBmapping[pointIdx]; i>=0; i-=threadsDistCalcReg)
		{

			//modified
			unsigned int dataIdx=(unsigned int)i+(tid%threadsDistCalcReg);		

			//break out of the loop if we exceed distmin or the data id>=DBsize
			if (database[dataIdx*GPUNUMDIM]<distmin || (dataIdx>=(*DBSIZE)))
			{
				break;
			}

			runningTotalDist=0;

			#pragma unroll
			for (int l=0; l<GPUNUMDIM; l++){
		          runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
		        }
		        runningTotalDist=sqrt(runningTotalDist);	

		        if (runningTotalDist<=eps)
		        {
		        	// cntResults++;
		        	atomicInc(&cntResults,(unsigned int)1);
		        }
		}

	
		__syncthreads();

	}

	


/*

	//Now that we have computed distmin/distmax that gives us at least knn, we store the results


		//count increasing to dist max
		for (unsigned int i=DBmapping[pointIdx]; i<(*DBSIZE); i+=threadsDistCalcReg)
		{

			
			unsigned int dataIdx=i+(tid%threadsDistCalcReg);		



			//break out of the loop if we exceed distmax or the dataid>=DBsize
			if (((database[dataIdx*GPUNUMDIM])>distmax) || (dataIdx>=(*DBSIZE)))
			{
				break;
			}

			runningTotalDist=0;

			#pragma unroll
			for (int l=0; l<GPUNUMDIM; l++){
		          runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
		        }
		        runningTotalDist=sqrt(runningTotalDist);	

		        if (runningTotalDist<=eps)
		        {
		        	unsigned int idx=atomicAdd(cnt,int(1));
		          	pointIDKey[idx]=pointIdx;
		          	pointInDistVal[idx]=DBmapping[dataIdx];
		          	distancesKeyVal[idx]=runningTotalDist;
		        }
		}


		//count decreasing to dist min
		for (unsigned int i=DBmapping[pointIdx]; i>=0; i-=threadsDistCalcReg)
		{

			//modified
			unsigned int dataIdx=i+(tid%threadsDistCalcReg);		

			//break out of the loop if we exceed distmin or the data id>=DBsize
			if (database[dataIdx*GPUNUMDIM]<distmin || (dataIdx>=(*DBSIZE)))
			{
				break;
			}

			runningTotalDist=0;

			#pragma unroll
			for (int l=0; l<GPUNUMDIM; l++){
		          runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
		        }
		        runningTotalDist=sqrt(runningTotalDist);	

		        if (runningTotalDist<=eps)
		        {
		        	unsigned int idx=atomicAdd(cnt,int(1));
		          	pointIDKey[idx]=pointIdx;
		          	pointInDistVal[idx]=DBmapping[dataIdx];
		          	distancesKeyVal[idx]=runningTotalDist;
		        }
		}


*/











	//ORIGINAL BRUTE FORCE
	/*
	DTYPE runningTotalDist=0;


	for (unsigned int i=0; i<(*DBSIZE); i+=threadsDistCalcReg)
	{		
		runningTotalDist=0;

		//original
		// unsigned int dataIdx=i+tid;

		//modified
		unsigned int dataIdx=i+(tid%threadsDistCalcReg);		
		
		

		//to ensure that the thread doesn't go past the database size due to multiple threads per point
		if (dataIdx<(*DBSIZE))
		{
	        for (int l=0; l<GPUNUMDIM; l++){
	          runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
	        }
	        

	        	
	        runningTotalDist=sqrt(runningTotalDist);

	    
		          unsigned int idx=atomicAdd(cnt,int(1));
		          pointIDKey[idx]=pointIdx;
		          pointInDistVal[idx]=dataIdx;
		          distancesKeyVal[idx]=runningTotalDist;
	
	    }      
	}

	*/





}




/*
__global__ void kernelNDGridIndexBatchEstimatorQuerySet(unsigned int *debug1, unsigned int *debug2, unsigned int * threadsForDistanceCalc, unsigned int * queryPts,unsigned int *N,  
	unsigned int * sampleOffset, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
	unsigned int * nNonEmptyCells,  unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets)
{

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 



if (tid>=*N){
	return;
}

// if (queryPntIdx>=*N){
// 	return;
// }




//original
// unsigned int pointID=tid*(*sampleOffset)*(GPUNUMDIM);




unsigned int queryPntIdx=queryPts[tid*(*sampleOffset)];
unsigned int pointID=queryPntIdx*(GPUNUMDIM);

//make a local copy of the point
DTYPE point[GPUNUMDIM];
for (int i=0; i<GPUNUMDIM; i++){
	point[i]=database[pointID+i];	
}

//calculate the coords of the Cell for the point
//and the min/max ranges in each dimension
unsigned int nDCellIDs[NUMINDEXEDDIM];
unsigned int nDMinCellIDs[NUMINDEXEDDIM];
unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
for (int i=0; i<NUMINDEXEDDIM; i++){
	nDCellIDs[i]=(point[i]-minArr[i])/(*epsilon);
	nDMinCellIDs[i]=max(0,nDCellIDs[i]-1); //boundary conditions (don't go beyond cell 0)
	nDMaxCellIDs[i]=min(nCells[i]-1,nDCellIDs[i]+1); //boundary conditions (don't go beyond the maximum number of cells)

}



///////////////////////////
//Take the intersection of the ranges for each dimension between
//the point and the filtered set of cells in each dimension 
//Ranges in a given dimension that have points in them that are non-empty in a dimension will be tested 
///////////////////////////

unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];
//compare the point's range of cell IDs in each dimension to the filter mask
//only 2 possible values (you always find the middle point in the range), because that's the cell of the point itself
bool foundMin=0;
bool foundMax=0;

//we go through each dimension and compare the range of the query points min/max cell ids to the filtered ones
	//find out which ones in the range exist based on the min/max
	//then determine the appropriate ranges
	for (int i=0; i<NUMINDEXEDDIM; i++)
	{
		foundMin=0;
		foundMax=0;
		//for each dimension
		//OPTIMIZE: WITH BINARY SEARCH LATER
		// for (int dimFilterRng=gridCellNDMaskOffsets[(i*2)]; dimFilterRng<=gridCellNDMaskOffsets[(i*2)+1]; dimFilterRng++){
		// 	if (gridCellNDMask[dimFilterRng]==nDMinCellIDs[i])
		// 		foundMin=1;
		// 	if (gridCellNDMask[dimFilterRng]==nDMaxCellIDs[i])
		// 		foundMax=1;
		// }
		
		
		
		if(thrust::binary_search(thrust::seq, gridCellNDMask+ gridCellNDMaskOffsets[(i*2)],
			gridCellNDMask+ gridCellNDMaskOffsets[(i*2)+1]+1,nDMinCellIDs[i])){ //extra +1 here is because we include the upper bound
			foundMin=1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask+ gridCellNDMaskOffsets[(i*2)],
			gridCellNDMask+ gridCellNDMaskOffsets[(i*2)+1]+1,nDMaxCellIDs[i])){ //extra +1 here is because we include the upper bound
			foundMax=1;
		}

		

		// cases:
		// found the min and max
		// found the min and not max
		//found the max and not the min
		//you don't find the min or max -- then only check the mid
		//you always find the mid because it's in the cell of the point you're looking for

		//NEED TO OPTIMIZE STILL
		if (foundMin==1 && foundMax==1){
			rangeFilteredCellIdsMin[i]=nDMinCellIDs[i];
			rangeFilteredCellIdsMax[i]=nDMaxCellIDs[i];
			//printf("\nmin and max");
		}
		else if (foundMin==1 && foundMax==0){
			rangeFilteredCellIdsMin[i]=nDMinCellIDs[i];
			rangeFilteredCellIdsMax[i]=nDMinCellIDs[i]+1;
			//printf("\nmin not max");
		}
		else if (foundMin==0 && foundMax==1){
			rangeFilteredCellIdsMin[i]=nDMinCellIDs[i]+1;	
			rangeFilteredCellIdsMax[i]=nDMaxCellIDs[i];
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i]=nDMinCellIDs[i]+1;	
			rangeFilteredCellIdsMax[i]=nDMinCellIDs[i]+1;	
		}

		
		
	}		


	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////	
	



	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0]=rangeFilteredCellIdsMin[0]; loopRng[0]<=rangeFilteredCellIdsMax[0]; loopRng[0]++)
	for (loopRng[1]=rangeFilteredCellIdsMin[1]; loopRng[1]<=rangeFilteredCellIdsMax[1]; loopRng[1]++)
	#include "kernelloops.h"						
	{ //beginning of loop body
	

	#if COUNTMETRICS == 1
			atomicAdd(debug1,int(1));
	#endif	

	for (int x=0; x<NUMINDEXEDDIM; x++){
	indexes[x]=loopRng[x];	
	// if (tid==0)
	// 	printf("\ndim: %d, indexes: %d",x, indexes[x]);
	}
	

	
	uint64_t calcLinearID=getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
	//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says 
	//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

	//CHANGE THIS TO A BINARY SEARCH LATER	

	// for (int x=0; x<(*nNonEmptyCells);x++){
	// 	if (calcLinearID==gridCellLookupArr[x].gridLinearID){
	// 		cellsToCheck->push_back(calcLinearID); 
	// 	}
	// }
	
	// printf("\ntid: %d, Linear id: %d",tid,calcLinearID);

	struct gridCellLookup tmp;
	tmp.gridLinearID=calcLinearID;
	if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr+ (*nNonEmptyCells), gridCellLookup(tmp))){
		//in the GPU implementation we go directly to computing neighbors so that we don't need to
		//store a buffer of the cells to check 
		//cellsToCheck->push_back(calcLinearID); 

		//HERE WE COMPUTE THE NEIGHBORS FOR THE CELL
		//XXXXXXXXXXXXXXXXXXXXXXXXX
		


		
		struct gridCellLookup * resultBinSearch=thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex=resultBinSearch->idx;

		for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
				DTYPE runningTotalDist=0;
				unsigned int dataIdx=indexLookupArr[k];

				

				for (int l=0; l<GPUNUMDIM; l++){
				runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
				
				  #if SHORTCIRCUIT==1
		          if (sqrt(runningTotalDist)>(*epsilon)) {
		              break;
		          }
		          #endif

				}


				if (sqrt(runningTotalDist)<=(*epsilon)){
					unsigned int idx=atomicAdd(cnt,int(1));
					// pointIDKey[idx]=tid;
					// pointInDistVal[idx]=i;
					//neighborTableCPUPrototype[queryPoint].neighbors.push_back(dataIdx);

				}
			}



	}

	
	//printf("\nLinear id: %d",calcLinearID);
	} //end loop body

}

*/

__global__ void kernelNDGridIndexBatchEstimatorOLD(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
	unsigned int * sampleOffset, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
	unsigned int * nNonEmptyCells,  unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets)
{

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

if (tid>=*N){
	return;
}


unsigned int pointID=tid*(*sampleOffset)*(GPUNUMDIM);

//make a local copy of the point
DTYPE point[GPUNUMDIM];
for (int i=0; i<GPUNUMDIM; i++){
	point[i]=database[pointID+i];	
}

//calculate the coords of the Cell for the point
//and the min/max ranges in each dimension
unsigned int nDCellIDs[NUMINDEXEDDIM];
unsigned int nDMinCellIDs[NUMINDEXEDDIM];
unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
for (int i=0; i<NUMINDEXEDDIM; i++){
	nDCellIDs[i]=(point[i]-minArr[i])/(*epsilon);
	nDMinCellIDs[i]=max(0,nDCellIDs[i]-1); //boundary conditions (don't go beyond cell 0)
	nDMaxCellIDs[i]=min(nCells[i]-1,nDCellIDs[i]+1); //boundary conditions (don't go beyond the maximum number of cells)

}



///////////////////////////
//Take the intersection of the ranges for each dimension between
//the point and the filtered set of cells in each dimension 
//Ranges in a given dimension that have points in them that are non-empty in a dimension will be tested 
///////////////////////////

unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];
//compare the point's range of cell IDs in each dimension to the filter mask
//only 2 possible values (you always find the middle point in the range), because that's the cell of the point itself
bool foundMin=0;
bool foundMax=0;

//we go through each dimension and compare the range of the query points min/max cell ids to the filtered ones
	//find out which ones in the range exist based on the min/max
	//then determine the appropriate ranges
	for (int i=0; i<NUMINDEXEDDIM; i++)
	{
		foundMin=0;
		foundMax=0;
		//for each dimension
		//OPTIMIZE: WITH BINARY SEARCH LATER
		// for (int dimFilterRng=gridCellNDMaskOffsets[(i*2)]; dimFilterRng<=gridCellNDMaskOffsets[(i*2)+1]; dimFilterRng++){
		// 	if (gridCellNDMask[dimFilterRng]==nDMinCellIDs[i])
		// 		foundMin=1;
		// 	if (gridCellNDMask[dimFilterRng]==nDMaxCellIDs[i])
		// 		foundMax=1;
		// }
		
		
		
		if(thrust::binary_search(thrust::seq, gridCellNDMask+ gridCellNDMaskOffsets[(i*2)],
			gridCellNDMask+ gridCellNDMaskOffsets[(i*2)+1]+1,nDMinCellIDs[i])){ //extra +1 here is because we include the upper bound
			foundMin=1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask+ gridCellNDMaskOffsets[(i*2)],
			gridCellNDMask+ gridCellNDMaskOffsets[(i*2)+1]+1,nDMaxCellIDs[i])){ //extra +1 here is because we include the upper bound
			foundMax=1;
		}

		

		// cases:
		// found the min and max
		// found the min and not max
		//found the max and not the min
		//you don't find the min or max -- then only check the mid
		//you always find the mid because it's in the cell of the point you're looking for

		//NEED TO OPTIMIZE STILL
		if (foundMin==1 && foundMax==1){
			rangeFilteredCellIdsMin[i]=nDMinCellIDs[i];
			rangeFilteredCellIdsMax[i]=nDMaxCellIDs[i];
			//printf("\nmin and max");
		}
		else if (foundMin==1 && foundMax==0){
			rangeFilteredCellIdsMin[i]=nDMinCellIDs[i];
			rangeFilteredCellIdsMax[i]=nDMinCellIDs[i]+1;
			//printf("\nmin not max");
		}
		else if (foundMin==0 && foundMax==1){
			rangeFilteredCellIdsMin[i]=nDMinCellIDs[i]+1;	
			rangeFilteredCellIdsMax[i]=nDMaxCellIDs[i];
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i]=nDMinCellIDs[i]+1;	
			rangeFilteredCellIdsMax[i]=nDMinCellIDs[i]+1;	
		}

		
		
	}		


	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////	
	



	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0]=rangeFilteredCellIdsMin[0]; loopRng[0]<=rangeFilteredCellIdsMax[0]; loopRng[0]++)
	for (loopRng[1]=rangeFilteredCellIdsMin[1]; loopRng[1]<=rangeFilteredCellIdsMax[1]; loopRng[1]++)
	#include "kernelloops.h"						
	{ //beginning of loop body
	

	#if COUNTMETRICS == 1
			atomicAdd(debug1,int(1));
	#endif	

	for (int x=0; x<NUMINDEXEDDIM; x++){
	indexes[x]=loopRng[x];	
	// if (tid==0)
	// 	printf("\ndim: %d, indexes: %d",x, indexes[x]);
	}
	

	
	uint64_t calcLinearID=getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
	//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says 
	//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

	//CHANGE THIS TO A BINARY SEARCH LATER	

	// for (int x=0; x<(*nNonEmptyCells);x++){
	// 	if (calcLinearID==gridCellLookupArr[x].gridLinearID){
	// 		cellsToCheck->push_back(calcLinearID); 
	// 	}
	// }
	
	// printf("\ntid: %d, Linear id: %d",tid,calcLinearID);

	struct gridCellLookup tmp;
	tmp.gridLinearID=calcLinearID;
	if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr+ (*nNonEmptyCells), gridCellLookup(tmp))){
		//in the GPU implementation we go directly to computing neighbors so that we don't need to
		//store a buffer of the cells to check 
		//cellsToCheck->push_back(calcLinearID); 

		//HERE WE COMPUTE THE NEIGHBORS FOR THE CELL
		//XXXXXXXXXXXXXXXXXXXXXXXXX
		


		
		struct gridCellLookup * resultBinSearch=thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex=resultBinSearch->idx;

		for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
				DTYPE runningTotalDist=0;
				unsigned int dataIdx=indexLookupArr[k];

				

				for (int l=0; l<GPUNUMDIM; l++){
				runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
				}


				if (sqrt(runningTotalDist)<=(*epsilon)){
					unsigned int idx=atomicAdd(cnt,int(1));
					// pointIDKey[idx]=tid;
					// pointInDistVal[idx]=i;
					//neighborTableCPUPrototype[queryPoint].neighbors.push_back(dataIdx);

				}
			}



	}

	
	//printf("\nLinear id: %d",calcLinearID);
	} //end loop body

}


__global__ void kernelIndexComputeNonemptyCells(DTYPE* database, unsigned int *N, DTYPE* epsilon, DTYPE* minArr, unsigned int * nCells, uint64_t * pointCellArr)
{


	unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 


	if (tid>=*N){
		return;
	}

	// printf("\n%u",tid); 

	unsigned int pointID=tid*(GPUNUMDIM);

	unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
	for (int j=0; j<NUMINDEXEDDIM; j++){
		tmpNDCellIdx[j]=((database[pointID+j]-minArr[j])/(*epsilon));
	}
	uint64_t linearID=getLinearID_nDimensionsGPU(tmpNDCellIdx, nCells, NUMINDEXEDDIM);

	pointCellArr[tid]=linearID;	

		
}

__global__ void kernelInitEnumerateDB(unsigned int * databaseVal, unsigned int *N)
{


	unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 


	if (tid>=*N){
		return;
	}

	databaseVal[tid]=tid;	

		
}