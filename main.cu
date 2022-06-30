
#include <cstring>						// string manipulation
#include <string>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <stdio.h>
#include <random>
#include "omp.h"
#include <algorithm> 
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include "GPU.h"
#include "kernel.h"
// #include "tree_index.h"
#include "par_sort.h"
#include <math.h>
#include <queue>
#include <iomanip>
#include <set>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <semaphore.h>




//for printing defines as strings
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// #define kNN 5 //finds the self-point + the kNN here

//semaphores for index construction while kNN searching
// sem_t semIndex;
// sem_t semkNN;

//sort descending
bool compareByDimVariance(const dim_reorder_sort &a, const dim_reorder_sort &b)
{
    return a.variance > b.variance;
}

//sort ascending
bool compareByNumPointsInCell(const GPUQueryNumPts &a, const GPUQueryNumPts &b)
{
    return a.pntsInCell < b.pntsInCell;
}


//sort descending
bool compareWorkArrayByNumPointsInCell(const workArray &a, const workArray &b)
{
    return a.pntsInCell > b.pntsInCell;
}

//sort descending
bool compareByPointValue(const keyValPointDistStruct &a, const keyValPointDistStruct &b)
{
    return a.distance > b.distance;
}

//sort ascending
bool compareByPointFrequencyInOtherSets(const keyValPointFrequencyKNNGraphStruct &a, const keyValPointFrequencyKNNGraphStruct &b)
{
    return a.numTimesInOtherSet < b.numTimesInOtherSet;
}

using namespace std;

//function prototypes


void printNeighborTable(unsigned int databaseSize, int k_Neighbors, int * nearestNeighborTable, DTYPE * nearestNeighborTableDistances);
void splitWork(unsigned int k_neighbors, std::vector<unsigned int> * queriesCPU, std::vector<unsigned int> * queriesGPU, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, struct grid * index);
void generateNeighborTableCPUPrototype(std::vector<std::vector <DTYPE> > *NDdataPoints, unsigned int queryPoint, DTYPE epsilon, grid * index, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, std::vector<uint64_t> * cellsToCheck, table * neighborTableCPUPrototype);
void findNonEmptyCellsPrototype(DTYPE * point, DTYPE* epsilon, grid * index, DTYPE* minArr, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * nCells, unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets, std::vector<uint64_t> * cellsToCheck);
uint64_t getLinearID_nDimensions(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions);
void populateNDGridIndexAndLookupArray(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr,  DTYPE* minArr, unsigned int * nCells, uint64_t totalCells, unsigned int * nNonEmptyCells);
void populateNDGridIndexAndLookupArrayParallel(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr,  DTYPE* minArr, unsigned int * nCells, uint64_t totalCells, unsigned int * nNonEmptyCells);
void generateNDGridDimensions(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, DTYPE* minArr, DTYPE* maxArr, unsigned int * nCells, uint64_t * totalCells);
void importNDDataset(std::vector<std::vector <DTYPE> > *dataPoints, char * fname);
void CPUBruteForceTable(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, table * neighborTable, unsigned int * totalNeighbors);
void sortInNDBins(std::vector<std::vector <DTYPE> > *dataPoints);
void ReorderByDimension(std::vector<std::vector <DTYPE> > *NDdataPoints);
void procNeighborTableForkNN(std::vector<std::vector<DTYPE> > * NDdataPoints, neighborTableLookup * neighborTable, int k_Neighbors, std::vector<unsigned int> *queryPts);
void storeNeighborTableForkNN(std::vector<std::vector<DTYPE> > * NDdataPoints, neighborTableLookup * neighborTable, int k_Neighbors, std::vector<unsigned int> *queryPts, int ** nearestNeighborTable, DTYPE ** nearestNeighborTableDistances);
double estimateEpsilon(std::vector<std::vector<DTYPE> > * NDdataPoints, unsigned int k_neighbors);
bool checkIndexSelectivity(unsigned int * nCells);
void procNeighborTableForkNNUsingDirectTable(std::vector<std::vector<DTYPE> > * NDdataPoints, int k_Neighbors, std::vector<unsigned int> *queryPts, int * nearestNeighborTable);
//only the GPU queries are considered
void procNeighborTableForkNNUsingDirectTableGPUOnly(std::vector<std::vector<DTYPE> > * NDdataPoints, int k_Neighbors, std::vector<unsigned int> *queryPts, int * nearestNeighborTable);

//To be used in loop condition
int criticalCheckEquality(unsigned int * totalQueriesCompleted, unsigned long int TotalQueries);


void computeWorkDifficulty(unsigned int * outputOrderedQueryPntIDs, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, struct grid * index);

//for outlier scores:
void printOutlierScores(unsigned int databaseSize, int k_Neighbors, int * nearestNeighborTable, DTYPE * nearestNeighborTableDistances);

#ifndef PYTHON //standard C version
int main(int argc, char *argv[])
{

	

	//nested so we can parallelize some of the GPU tasks
	omp_set_nested(2);
	/////////////////////////
	// Get information from command line
	//1) the dataset, 2) epsilon, 3) number of dimensions
	/////////////////////////

	//Read in parameters from file:
	//dataset filename and cluster instance file
	if (argc!=4)
	{
	cout <<"\n\nIncorrect number of input parameters.  \nShould be dataset file, number of dimensions, kNN\n";
	return 0;
	}
	
	//copy parameters from commandline:
	//char inputFname[]="data/test_data_removed_nan.txt";	
	char inputFname[500];
	char inputnumdim[500];


	strcpy(inputFname,argv[1]);
	strcpy(inputnumdim,argv[2]);

    unsigned int kNN=atoi(argv[3]);
	unsigned int NDIM=atoi(inputnumdim);

	if (GPUNUMDIM!=NDIM){
		printf("\nERROR: The number of dimensions defined for the GPU is not the same as the number of dimensions\n \
		 passed into the computer program on the command line. GPUNUMDIM=%d, NDIM=%d Exiting!!!",GPUNUMDIM,NDIM);
		
		return 0;
	}

	

	//////////////////////////////
	//import the dataset:
	/////////////////////////////
	
	std::vector<std::vector <DTYPE> > NDdataPoints;	
	importNDDataset(&NDdataPoints, inputFname);


	int * nearestNeighborTable;
	// DTYPE * nearestNeighborTableDistances;
	//size allocated- only allocate on master rank (rank nprocs-1)
	unsigned long int elemBuffer=NDdataPoints.size()*(kNN+1);
	printf("\nNumber of data points: %lu",NDdataPoints.size());
	nearestNeighborTable=(int *)malloc(sizeof(int)*elemBuffer);
	
	
	
	
	
	int * ptr_to_neighbortable; //points to the neighbortable starting at elem 0 (in rank nprocs-1's memory)
	ptr_to_neighbortable=nearestNeighborTable;
	
	DTYPE * ptr_to_neighbortable_distances=(DTYPE *)malloc(sizeof(DTYPE)*elemBuffer); //updated this for outlier detection; was NULL, but needs to be allocated
	

  	printf("\nSize of result set (GiB): %f", ((sizeof(int)*elemBuffer))/(1024.0*1024*1024));

	

	char fname[]="gpu_stats.txt";
	ofstream gpu_stats;
	gpu_stats.open(fname,ios::app);	

	printf("\n*****************\nWarming up GPU:\n*****************\n");
	warmUpGPU();
	printf("\n*****************\n");

	DTYPE * minArr= new DTYPE[NUMINDEXEDDIM];
	DTYPE * maxArr= new DTYPE[NUMINDEXEDDIM];
	unsigned int * nCells= new unsigned int[NUMINDEXEDDIM];
	uint64_t totalCells=0;
	unsigned int nNonEmptyCells=0;
	uint64_t totalNeighbors =0;
	double totalTime=0;
	double timeReorderByDimVariance=0;	

	//for conssitency
	double totalDistance=0;
	

	#if REORDER==1
	double reorder_start=omp_get_wtime();
	ReorderByDimension(&NDdataPoints);
	double reorder_end=omp_get_wtime();
	timeReorderByDimVariance= reorder_end - reorder_start;
	#endif

    
    DTYPE eps_est=0;
    DTYPE eps_est_initial=0;	
    unsigned int bucket=0;
    double tstartEstEps=omp_get_wtime();
    sampleNeighborsBruteForce(&NDdataPoints, &eps_est, &bucket, kNN);
    double tendEstEps=omp_get_wtime();
    double timeEstEps=tendEstEps - tstartEstEps;
    printf("\nTime to estimate epsilon: %f", timeEstEps);

    

    printf("\nMain: estimate of epsilon: %0.9f",eps_est);
    eps_est_initial=eps_est;
	

	generateNDGridDimensions(&NDdataPoints,eps_est, minArr, maxArr, nCells, &totalCells);
	printf("\nGrid: total cells (including empty) %lu",totalCells);

	


	// allocate memory for index now that we know the number of cells
	//the grid struct itself
	//the grid lookup array that accompanys the grid -- so we only send the non-empty cells
	struct grid * index; //allocate in the populateDNGridIndexAndLookupArray -- only index the non-empty cells
	struct gridCellLookup * gridCellLookupArr; //allocate in the populateDNGridIndexAndLookupArray -- list of non-empty cells

	//the grid cell mask tells you what cells are non-empty in each dimension
	//used for finding the non-empty cells that you want
	// unsigned int * gridCellNDMask; //allocate in the populateDNGridIndexAndLookupArray -- list of cells in each n-dimension that have elements in them
	// unsigned int * nNDMaskElems= new unsigned int; //size of the above array
	// unsigned int * gridCellNDMaskOffsets=new unsigned int [NUMINDEXEDDIM*2]; //offsets into the above array for each dimension
																	//as [min,max,min,max,min,max] (for 3-D)	

	//ids of the elements in the database that are found in each grid cell
	unsigned int * indexLookupArr=new unsigned int[NDdataPoints.size()]; 
	
	//CPU indexing- GPGPU paper
	// populateNDGridIndexAndLookupArray(&NDdataPoints, eps_est, &gridCellLookupArr, &index, indexLookupArr, minArr,  nCells, totalCells, &nNonEmptyCells);
	
	//GPU indexing
	populateNDGridIndexAndLookupArrayGPU(&NDdataPoints, &eps_est, minArr, totalCells, nCells, &gridCellLookupArr, &index, indexLookupArr, &nNonEmptyCells);
	


	//Neighbortable storage -- the result
	neighborTableLookup * neighborTable= new neighborTableLookup[NDdataPoints.size()];
	std::vector<struct neighborDataPtrs> pointersToNeighbors;

	//initialize all of the neighbors to -1, indicating that the neighbors for a point haven't been found yet
	

	uint64_t cnt=0;
	for (uint64_t i=0; i<(uint64_t)NDdataPoints.size(); i++)
	{
		for (uint64_t j=0; j<(uint64_t)kNN+1; j++)
		{
		ptr_to_neighbortable[cnt]=-1;	
		cnt++;
		}
	}


	
	unsigned int * outputOrderedQueryPntIDs=new unsigned int[NDdataPoints.size()];
	//Order the work based on points in each cell
	double tstartOrderWork=omp_get_wtime();
	computeWorkDifficulty(outputOrderedQueryPntIDs, gridCellLookupArr, &nNonEmptyCells, indexLookupArr, index);


	//store the number of queries so we can comput the fail rate of the batch (used for increasing epsilon)
	std::vector<unsigned int>queriesGPU;
	queriesGPU.insert(queriesGPU.end(), &outputOrderedQueryPntIDs[0], &outputOrderedQueryPntIDs[NDdataPoints.size()]);



	double tendOrderWork=omp_get_wtime();
	double timeOrderWork=tendOrderWork - tstartOrderWork;
	printf("\nTime to order the work: %f",timeOrderWork);

	delete [] outputOrderedQueryPntIDs;

	//Start join computation here
	double tstart=omp_get_wtime();		


	unsigned int * queriesGPUBuffer=new unsigned int[NDdataPoints.size()];
	unsigned int * queriesIncompleteGPUBuffer=new unsigned int[NDdataPoints.size()];

	//Get queries to work on from the producer rank
	unsigned int numQueries=NDdataPoints.size();
	

	




	int numIter=0;	

	double fractionFailuresPrevIter=0;

	//XXX
	// while (numQueries!=-1)
	while (numQueries!=0)
	{

		CTYPE* workCounts = (CTYPE*)malloc(2*sizeof(CTYPE));
		workCounts[0]=0;
		workCounts[1]=0;

		
		pointersToNeighbors.clear();		

	
		///////////////////////////////////////////////
		//Only increase epsilon if reaches threshold failures

		//If not the first iteration, we re-index with the increased epsilon
		//If the number of failures on the previous iteration is >25%

		
		if (numIter!=0 && (fractionFailuresPrevIter>0.25) )
		{

			//free previously allocated memory
			delete [] gridCellLookupArr;
			delete [] index;

			double tstartindex=omp_get_wtime();
			//Increase epsilon by 0.5 epsilon:
			eps_est+=(eps_est_initial*0.5);

			if(eps_est==0.0)
			{
				printf("\n\n******************\nError: Epsilon is 0!\nExiting.\n******************\n");
				return(0);
			}
			
			generateNDGridDimensions(&NDdataPoints,eps_est, minArr, maxArr, nCells, &totalCells);

			//CPU Index construction- used in GPGPU paper
			// populateNDGridIndexAndLookupArrayParallel(&NDdataPoints, eps_est, &gridCellLookupArr, &index, indexLookupArr, minArr,  nCells, totalCells, &nNonEmptyCells);
			
			//GPU index construction:
			populateNDGridIndexAndLookupArrayGPU(&NDdataPoints, &eps_est, minArr, totalCells, nCells, &gridCellLookupArr, &index, indexLookupArr, &nNonEmptyCells);
			double tendindex=omp_get_wtime();
			printf("\nGPU: Time to reindex: %f", tendindex - tstartindex);
		}	
		
		
		
		printf("\nIteration: %d, epsilon: %f",numIter, eps_est);


		//End only increase if failures on previous iteration
		/////////////////////////////////////

		
		
		//Run GPU
		//with pointers to windowed memory
		//only if there are queries for the GPU
		
		
		unsigned int numQueriesForFailRate=queriesGPU.size();

		printf("\nNumber of GPU queries: %lu", queriesGPU.size());	
		if (queriesGPU.size()>0)
		{
		double totaldisttmp=0.0; //for consistency
		double tstartGPU=omp_get_wtime();	
		distanceTableNDGridBatcheskNN(&NDdataPoints, ptr_to_neighbortable, ptr_to_neighbortable_distances, &totaldisttmp, &queriesGPU, &eps_est, kNN, index, gridCellLookupArr, &nNonEmptyCells,  minArr, nCells, indexLookupArr, neighborTable, &pointersToNeighbors, &totalNeighbors, workCounts);	
		double tendGPU=omp_get_wtime();
		printf("\n[GRID] Time to compute distance table for KNN (epsilon=%f): %f,", eps_est, tendGPU - tstartGPU);

		// for consistency
		totalDistance+=totaldisttmp; 

		printf("\nTotal Distance: %f", totaldisttmp);

		//kNN find points with insufficient neighbors from the GPU execution (the ANN execution always finds enough neighbors)
		double tstartprocNeighborTable=omp_get_wtime();
		procNeighborTableForkNNUsingDirectTableGPUOnly(&NDdataPoints, kNN, &queriesGPU, ptr_to_neighbortable);
		double tendprocNeighborTable=omp_get_wtime();
		printf("\nTime proc queries from neighbor table (GPU only): %f", tendprocNeighborTable- tstartprocNeighborTable);
		}

		//The load imbalance may be due to the work-queue and not the CPU/GPU end times
		//Print the time since the start that the GPU finished computing its batch
		printf("\nGPU finished batch at: %f",omp_get_wtime()-tstart);

		
		
		fractionFailuresPrevIter=(queriesGPU.size()*1.0)/(numQueriesForFailRate*1.0);
		unsigned long int numGPUFailures=queriesGPU.size();
		printf("\nNumber of GPU failures: %lu, Fraction failures: %f",numGPUFailures,fractionFailuresPrevIter);

		//copy the failures to array
		for (int x=0; x<numGPUFailures; x++){
			queriesIncompleteGPUBuffer[x]=queriesGPU[x];
		}



		//Send the workmaster the failed queries
		// unsigned int cntIncomplete=numGPUFailures;

		
		//get the next batch of query points:
		queriesGPU.clear();

		//XXX feed the failures back into the query set
		queriesGPU.insert(queriesGPU.end(), &queriesIncompleteGPUBuffer[0], &queriesIncompleteGPUBuffer[numGPUFailures]);		

		
		numIter++;

		//set the number of queries for the while loop
		numQueries=numGPUFailures;


	} //end looping over GPU queries

	

	double tendGPU=omp_get_wtime();
	printf("\nFinished all GPU queries. Time: %f", tendGPU - tstart);fflush(stdout);

	
	cleanUpPinned();

	double tend=omp_get_wtime();
	printf("\nTime to join only: %f",tend-tstart);



	totalTime=(tend-tstart)+timeReorderByDimVariance+timeEstEps+timeOrderWork;

	



	
	printf("\n[Verification] Total distance (without square root): %f",totalDistance);

	//verification, check entire neighbortable
	procNeighborTableForkNNUsingDirectTable(&NDdataPoints, kNN, &queriesGPU, ptr_to_neighbortable);
	

	double gputime=tendGPU-tstart;
	
	#if PRINTNEIGHBORTABLE==1
	printNeighborTable(NDdataPoints.size(), kNN+1, ptr_to_neighbortable, ptr_to_neighbortable_distances);
	#endif


	//for outlier detection with SNAPS
	#if PRINTOUTLIERSCORES==1
	printOutlierScores(NDdataPoints.size(), kNN+1, ptr_to_neighbortable, ptr_to_neighbortable_distances);
	#endif
	
	//Output stats to gpu_stats.txt
	//dynamic	
	#if THREADMULTI==-1
	gpu_stats<<totalTime<<", "<< inputFname<<", KNN: "<<kNN<<nprocs<<", Eps bucket: "<<bucket<<", Total dist: "<<setprecision(9)<<
	totalDistance<<", "<<
	"GPUNUMDIM/NUMINDEXEDDIM/REORDER/SHORTCIRCUIT/THREADMULTI/MAXTHREADSPERPOINT/DYNAMICTHRESHOLD/STATICTHREADSPERPOINT/BETA/DTYPE(float/double): "
	<<GPUNUMDIM<<", "<<NUMINDEXEDDIM<<", "<<REORDER<< ", "<<SHORTCIRCUIT<<", "<<THREADMULTI<<", "<<MAXTHREADSPERPOINT<<", "
	<<DYNAMICTHRESHOLD<<", N/A, "<<BETA<<", "
	<< STR(DTYPE)<<endl;
	#endif

	#if THREADMULTI==-2
	gpu_stats<<totalTime<<", "<< inputFname<<", KNN: "<<kNN<<", Eps bucket: "<<bucket<<", Total dist: "<<setprecision(9)
	<<totalDistance<<", "<<
	"GPUNUMDIM/NUMINDEXEDDIM/REORDER/SHORTCIRCUIT/THREADMULTI/MAXTHREADSPERPOINT/DYNAMICTHRESHOLD/STATICTHREADSPERPOINT/BETA/DTYPE(float/double): "
	<<GPUNUMDIM<<", "<<NUMINDEXEDDIM<<", "<<REORDER<< ", "<<SHORTCIRCUIT<<", "<<THREADMULTI<<", "<<MAXTHREADSPERPOINT<<", N/A, "<<STATICTHREADSPERPOINT<<", "
	<<BETA<<", "
	<<STR(DTYPE)<<endl;
	#endif
	gpu_stats.close();

	//TESTING: Print NeighborTable:
		//new neighbortable in arrays
		
	/*
	for (int i=0; i<NDdataPoints.size(); i++)
	{
		printf("\nPoint id: %d, Neighbors: ",i);
		// printf("\nPoint id: %d (coords: %f, %f), Neighbors: ",i, NDdataPoints[i][0],NDdataPoints[i][1]);
		for (int j=0; j<kNN+1; j++){
			printf(" ValIdx: %d, dist: %f", nearestNeighborTable[(i*(kNN+1))+j],nearestNeighborTableDistances[(i*(kNN+1))+j]);
			// printf(" ValIdx: %u (coords: %f, %f), dist: %f", nearestNeighborTable[(i*(kNN+1))+j],NDdataPoints[nearestNeighborTable[(i*(kNN+1))+j]][0],NDdataPoints[nearestNeighborTable[(i*(kNN+1))+j]][1],nearestNeighborTableDistances[(i*(kNN+1))+j]);
			
		}	
	}
	*/

	printf("\n\n");
	return 0;

}
#endif //end #if not Python (standard C version)

#ifdef PYTHON
extern "C" void KNNJoinPy(DTYPE * dataset, unsigned int NUMPOINTS, unsigned int kNN, unsigned int NDIM, int * outNearestNeighborTable, DTYPE * outNearestNeighborTableDistances)
{

	

	//nested so we can parallelize some of the GPU tasks
	omp_set_nested(2);
	/////////////////////////
	// Get information from command line
	//1) the dataset, 2) epsilon, 3) number of dimensions
	/////////////////////////

	if (GPUNUMDIM!=NDIM){
		printf("\nERROR: The number of dimensions defined for the GPU is not the same as the number of dimensions\n \
		 passed from the Python Interface. GPUNUMDIM=%d, NDIM=%d Exiting!!!",GPUNUMDIM,NDIM);
		return;
	}

	// fprintf(stderr, "\n************************");
	// fprintf(stderr, "Python parameters to shared library\n");
	// fprintf(stderr, "Number of points: %d\n", NUMPOINTS);
	// fprintf(stderr, "kNN: %d\n", kNN);
	// fprintf(stderr, "NDIM: %d\n", NDIM);
	// fprintf(stderr, "************************\n");
	

	//////////////////////////////
	//import the dataset:
	/////////////////////////////
	
	std::vector<std::vector <DTYPE> > NDdataPoints;	

	//copy data into the dataset vector
	for (unsigned int i=0; i<NUMPOINTS; i++){
  		unsigned int idxMin=i*GPUNUMDIM;
  		unsigned int idxMax=(i+1)*GPUNUMDIM;
		std::vector<DTYPE>tmpPoint(dataset+idxMin, dataset+idxMax);
		NDdataPoints.push_back(tmpPoint);
	}
	


	// int * nearestNeighborTable;
	// DTYPE * nearestNeighborTableDistances;
	//size allocated- only allocate on master rank (rank nprocs-1)
	unsigned long int elemBuffer=NDdataPoints.size()*(kNN+1);
	// printf("\nNumber of data points: %lu",NDdataPoints.size());
	// nearestNeighborTable=(int *)malloc(sizeof(int)*elemBuffer);
	
	 
	int * ptr_to_neighbortable=outNearestNeighborTable;
	
	// DTYPE * outNearestNeighborTableDistances=(DTYPE *)malloc(sizeof(DTYPE)*elemBuffer); //updated this for outlier detection; was NULL, but needs to be allocated
	

  	printf("\nSize of result set (GiB): %f", ((sizeof(int)*elemBuffer))/(1024.0*1024*1024));

	

	char fname[]="gpu_stats.txt";
	ofstream gpu_stats;
	gpu_stats.open(fname,ios::app);	

	printf("\n*****************\nWarming up GPU:\n*****************\n");
	warmUpGPU();
	printf("\n*****************\n");

	DTYPE * minArr= new DTYPE[NUMINDEXEDDIM];
	DTYPE * maxArr= new DTYPE[NUMINDEXEDDIM];
	unsigned int * nCells= new unsigned int[NUMINDEXEDDIM];
	uint64_t totalCells=0;
	unsigned int nNonEmptyCells=0;
	uint64_t totalNeighbors =0;
	double totalTime=0;
	double timeReorderByDimVariance=0;	

	//for conssitency
	double totalDistance=0;
	

	#if REORDER==1
	double reorder_start=omp_get_wtime();
	ReorderByDimension(&NDdataPoints);
	double reorder_end=omp_get_wtime();
	timeReorderByDimVariance= reorder_end - reorder_start;
	#endif

    
    DTYPE eps_est=0;
    DTYPE eps_est_initial=0;	
    unsigned int bucket=0;
    double tstartEstEps=omp_get_wtime();
    sampleNeighborsBruteForce(&NDdataPoints, &eps_est, &bucket, kNN);
    double tendEstEps=omp_get_wtime();
    double timeEstEps=tendEstEps - tstartEstEps;
    printf("\nTime to estimate epsilon: %f", timeEstEps);

    

    printf("\nMain: estimate of epsilon: %0.9f",eps_est);
    eps_est_initial=eps_est;
	

	generateNDGridDimensions(&NDdataPoints,eps_est, minArr, maxArr, nCells, &totalCells);
	printf("\nGrid: total cells (including empty) %lu",totalCells);

	


	// allocate memory for index now that we know the number of cells
	//the grid struct itself
	//the grid lookup array that accompanys the grid -- so we only send the non-empty cells
	struct grid * index; //allocate in the populateDNGridIndexAndLookupArray -- only index the non-empty cells
	struct gridCellLookup * gridCellLookupArr; //allocate in the populateDNGridIndexAndLookupArray -- list of non-empty cells

	//the grid cell mask tells you what cells are non-empty in each dimension
	//used for finding the non-empty cells that you want
	// unsigned int * gridCellNDMask; //allocate in the populateDNGridIndexAndLookupArray -- list of cells in each n-dimension that have elements in them
	// unsigned int * nNDMaskElems= new unsigned int; //size of the above array
	// unsigned int * gridCellNDMaskOffsets=new unsigned int [NUMINDEXEDDIM*2]; //offsets into the above array for each dimension
																	//as [min,max,min,max,min,max] (for 3-D)	

	//ids of the elements in the database that are found in each grid cell
	unsigned int * indexLookupArr=new unsigned int[NDdataPoints.size()]; 
	
	//CPU indexing- GPGPU paper
	// populateNDGridIndexAndLookupArray(&NDdataPoints, eps_est, &gridCellLookupArr, &index, indexLookupArr, minArr,  nCells, totalCells, &nNonEmptyCells);
	
	//GPU indexing
	populateNDGridIndexAndLookupArrayGPU(&NDdataPoints, &eps_est, minArr, totalCells, nCells, &gridCellLookupArr, &index, indexLookupArr, &nNonEmptyCells);
	


	//Neighbortable storage -- the result
	neighborTableLookup * neighborTable= new neighborTableLookup[NDdataPoints.size()];
	std::vector<struct neighborDataPtrs> pointersToNeighbors;

	//initialize all of the neighbors to -1, indicating that the neighbors for a point haven't been found yet
	

	uint64_t cnt=0;
	for (uint64_t i=0; i<(uint64_t)NDdataPoints.size(); i++)
	{
		for (uint64_t j=0; j<(uint64_t)kNN+1; j++)
		{
		ptr_to_neighbortable[cnt]=-1;	
		cnt++;
		}
	}


	
	unsigned int * outputOrderedQueryPntIDs=new unsigned int[NDdataPoints.size()];
	//Order the work based on points in each cell
	double tstartOrderWork=omp_get_wtime();
	computeWorkDifficulty(outputOrderedQueryPntIDs, gridCellLookupArr, &nNonEmptyCells, indexLookupArr, index);


	//store the number of queries so we can comput the fail rate of the batch (used for increasing epsilon)
	std::vector<unsigned int>queriesGPU;
	queriesGPU.insert(queriesGPU.end(), &outputOrderedQueryPntIDs[0], &outputOrderedQueryPntIDs[NDdataPoints.size()]);



	double tendOrderWork=omp_get_wtime();
	double timeOrderWork=tendOrderWork - tstartOrderWork;
	printf("\nTime to order the work: %f",timeOrderWork);

	delete [] outputOrderedQueryPntIDs;

	//Start join computation here
	double tstart=omp_get_wtime();		


	unsigned int * queriesGPUBuffer=new unsigned int[NDdataPoints.size()];
	unsigned int * queriesIncompleteGPUBuffer=new unsigned int[NDdataPoints.size()];

	//Get queries to work on from the producer rank
	unsigned int numQueries=NDdataPoints.size();
	

	




	int numIter=0;	

	double fractionFailuresPrevIter=0;

	
	while (numQueries!=0)
	{

		CTYPE* workCounts = (CTYPE*)malloc(2*sizeof(CTYPE));
		workCounts[0]=0;
		workCounts[1]=0;

		
		pointersToNeighbors.clear();		

	
		///////////////////////////////////////////////
		//Only increase epsilon if reaches threshold failures

		//If not the first iteration, we re-index with the increased epsilon
		//If the number of failures on the previous iteration is >25%

		
		if (numIter!=0 && (fractionFailuresPrevIter>0.25) )
		{

			//free previously allocated memory
			delete [] gridCellLookupArr;
			delete [] index;

			double tstartindex=omp_get_wtime();
			//Increase epsilon by 0.5 epsilon:
			eps_est+=(eps_est_initial*0.5);

			if(eps_est==0.0)
			{
				printf("\n\n******************\nError: Epsilon is 0!\nExiting.\n******************\n");
				return;
			}
			
			generateNDGridDimensions(&NDdataPoints,eps_est, minArr, maxArr, nCells, &totalCells);

			//CPU Index construction- used in GPGPU paper
			// populateNDGridIndexAndLookupArrayParallel(&NDdataPoints, eps_est, &gridCellLookupArr, &index, indexLookupArr, minArr,  nCells, totalCells, &nNonEmptyCells);
			
			//GPU index construction:
			populateNDGridIndexAndLookupArrayGPU(&NDdataPoints, &eps_est, minArr, totalCells, nCells, &gridCellLookupArr, &index, indexLookupArr, &nNonEmptyCells);
			double tendindex=omp_get_wtime();
			printf("\nGPU: Time to reindex: %f", tendindex - tstartindex);
		}	
		
		
		
		printf("\nIteration: %d, epsilon: %f",numIter, eps_est);


		//End only increase if failures on previous iteration
		/////////////////////////////////////

		
		
		//Run GPU
		//with pointers to windowed memory
		//only if there are queries for the GPU
		
		
		unsigned int numQueriesForFailRate=queriesGPU.size();

		printf("\nNumber of GPU queries: %lu", queriesGPU.size());	
		if (queriesGPU.size()>0)
		{
		double totaldisttmp=0.0; //for consistency
		double tstartGPU=omp_get_wtime();	
		distanceTableNDGridBatcheskNN(&NDdataPoints, ptr_to_neighbortable, outNearestNeighborTableDistances, &totaldisttmp, &queriesGPU, &eps_est, kNN, index, gridCellLookupArr, &nNonEmptyCells,  minArr, nCells, indexLookupArr, neighborTable, &pointersToNeighbors, &totalNeighbors, workCounts);	
		double tendGPU=omp_get_wtime();
		printf("\n[GRID] Time to compute distance table for KNN (epsilon=%f): %f,", eps_est, tendGPU - tstartGPU);

		// for consistency
		totalDistance+=totaldisttmp; 

		printf("\nTotal Distance: %f", totaldisttmp);

		//kNN find points with insufficient neighbors from the GPU execution (the ANN execution always finds enough neighbors)
		double tstartprocNeighborTable=omp_get_wtime();
		procNeighborTableForkNNUsingDirectTableGPUOnly(&NDdataPoints, kNN, &queriesGPU, ptr_to_neighbortable);
		double tendprocNeighborTable=omp_get_wtime();
		printf("\nTime proc queries from neighbor table (GPU only): %f", tendprocNeighborTable- tstartprocNeighborTable);
		}

		//The load imbalance may be due to the work-queue and not the CPU/GPU end times
		//Print the time since the start that the GPU finished computing its batch
		printf("\nGPU finished batch at: %f",omp_get_wtime()-tstart);

		
		
		fractionFailuresPrevIter=(queriesGPU.size()*1.0)/(numQueriesForFailRate*1.0);
		unsigned long int numGPUFailures=queriesGPU.size();
		printf("\nNumber of GPU failures: %lu, Fraction failures: %f",numGPUFailures,fractionFailuresPrevIter);

		//copy the failures to array
		for (int x=0; x<numGPUFailures; x++){
			queriesIncompleteGPUBuffer[x]=queriesGPU[x];
		}



		//Send the workmaster the failed queries
		// unsigned int cntIncomplete=numGPUFailures;

		
		//get the next batch of query points:
		queriesGPU.clear();

		//XXX feed the failures back into the query set
		queriesGPU.insert(queriesGPU.end(), &queriesIncompleteGPUBuffer[0], &queriesIncompleteGPUBuffer[numGPUFailures]);		

		
		numIter++;

		//set the number of queries for the while loop
		numQueries=numGPUFailures;


	} //end looping over GPU queries

	

	double tendGPU=omp_get_wtime();
	printf("\nFinished all GPU queries. Time: %f", tendGPU - tstart);fflush(stdout);

	
	//Comment cleaning up pinned in the Python implementation because
	//we call the shared library numerous times
	#ifndef PYTHON
	cleanUpPinned();
	#endif

	double tend=omp_get_wtime();
	printf("\nTime to join only: %f",tend-tstart);



	totalTime=(tend-tstart)+timeReorderByDimVariance+timeEstEps+timeOrderWork;

	



	
	printf("\n[Verification] Total distance (without sqaure root): %f",totalDistance);

	//verification, check entire neighbortable
	procNeighborTableForkNNUsingDirectTable(&NDdataPoints, kNN, &queriesGPU, ptr_to_neighbortable);
	

	double gputime=tendGPU-tstart;
	
	#if PRINTNEIGHBORTABLE==1
	printNeighborTable(NDdataPoints.size(), kNN+1, ptr_to_neighbortable, outNearestNeighborTableDistances);
	#endif


	//for outlier detection with SNAPS
	#if PRINTOUTLIERSCORES==1
	printOutlierScores(NDdataPoints.size(), kNN+1, ptr_to_neighbortable, outNearestNeighborTableDistances);
	#endif
	
	//Output stats to gpu_stats.txt
	//dynamic	
	#if THREADMULTI==-1
	gpu_stats<<totalTime<<", KNN: "<<kNN<<nprocs<<", Eps bucket: "<<bucket<<", Total dist: "<<setprecision(9)<<
	totalDistance<<", "<<
	"GPUNUMDIM/NUMINDEXEDDIM/REORDER/SHORTCIRCUIT/THREADMULTI/MAXTHREADSPERPOINT/DYNAMICTHRESHOLD/STATICTHREADSPERPOINT/BETA/DTYPE(float/double): "
	<<GPUNUMDIM<<", "<<NUMINDEXEDDIM<<", "<<REORDER<< ", "<<SHORTCIRCUIT<<", "<<THREADMULTI<<", "<<MAXTHREADSPERPOINT<<", "
	<<DYNAMICTHRESHOLD<<", N/A, "<<BETA<<", "
	<< STR(DTYPE)<<endl;
	#endif

	#if THREADMULTI==-2
	gpu_stats<<totalTime<<", KNN: "<<kNN<<", Eps bucket: "<<bucket<<", Total dist: "<<setprecision(9)
	<<totalDistance<<", "<<
	"GPUNUMDIM/NUMINDEXEDDIM/REORDER/SHORTCIRCUIT/THREADMULTI/MAXTHREADSPERPOINT/DYNAMICTHRESHOLD/STATICTHREADSPERPOINT/BETA/DTYPE(float/double): "
	<<GPUNUMDIM<<", "<<NUMINDEXEDDIM<<", "<<REORDER<< ", "<<SHORTCIRCUIT<<", "<<THREADMULTI<<", "<<MAXTHREADSPERPOINT<<", N/A, "<<STATICTHREADSPERPOINT<<", "
	<<BETA<<", "
	<<STR(DTYPE)<<endl;
	#endif
	gpu_stats.close();

	//TESTING: Print NeighborTable:
		//new neighbortable in arrays
		
	/*
	for (int i=0; i<NDdataPoints.size(); i++)
	{
		printf("\nPoint id: %d, Neighbors: ",i);
		// printf("\nPoint id: %d (coords: %f, %f), Neighbors: ",i, NDdataPoints[i][0],NDdataPoints[i][1]);
		for (int j=0; j<kNN+1; j++){
			printf(" ValIdx: %d, dist: %f", nearestNeighborTable[(i*(kNN+1))+j],nearestNeighborTableDistances[(i*(kNN+1))+j]);
			// printf(" ValIdx: %u (coords: %f, %f), dist: %f", nearestNeighborTable[(i*(kNN+1))+j],NDdataPoints[nearestNeighborTable[(i*(kNN+1))+j]][0],NDdataPoints[nearestNeighborTable[(i*(kNN+1))+j]][1],nearestNeighborTableDistances[(i*(kNN+1))+j]);
			
		}	
	}
	*/

	printf("\n\n");
	return;

}
#endif


void printNeighborTable(unsigned int databaseSize, int k_Neighbors, int * nearestNeighborTable, DTYPE * nearestNeighborTableDistances)
{

	char fname[]="KNN_out.txt";
	ofstream KNN_out;
	KNN_out.open(fname,ios::out);	

	printf("\n\nOutputting neighbors to: %s\n", fname);
	KNN_out<<"#data point (line is the point id), neighbor point ids\n";

	for (unsigned int i=0; i<databaseSize; i++)
	{
		for (int j=0; j<k_Neighbors; j++)
		{
			KNN_out<<nearestNeighborTable[i*k_Neighbors+j]<<", ";
		}
		KNN_out<<"\n";
	}

	KNN_out.close();

	char fname1[]="KNN_out_distances.txt";
	ofstream KNN_out_distances;
	KNN_out_distances.open(fname1, ios::out);	

	printf("\nOutputting distances to: %s\n", fname1);
	KNN_out_distances<<"#data point (line is the point id), neighbor point distances\n";

	for (unsigned int i=0; i<databaseSize; i++)
	{
		for (int j=0; j<k_Neighbors; j++)
		{
			KNN_out_distances<<nearestNeighborTableDistances[i*k_Neighbors+j]<<", ";
		}
		KNN_out_distances<<"\n";
	}

	KNN_out_distances.close();
}


void printOutlierScores(unsigned int databaseSize, int k_Neighbors, int * nearestNeighborTable, DTYPE * nearestNeighborTableDistances)
{

	///////////////////
	//Outlier criterion 1: print the mean distance to the kNN (subtract 1 from the KNN value because we exclude a point finding itself)


	//For each point, compute the total squared distances to its k neighbors
	DTYPE * totalDistancesPerPoint = (DTYPE *)malloc(sizeof(DTYPE)*databaseSize);
	struct keyValPointDistStruct * keyValuePairPointIDDistance = (struct keyValPointDistStruct *)malloc(sizeof(keyValPointDistStruct)*databaseSize);
	for (unsigned int i=0; i<databaseSize; i++)
	{
		DTYPE totalDistancePoint=0;
		for (int j=0; j<k_Neighbors; j++)
		{
			totalDistancePoint+=nearestNeighborTableDistances[i*k_Neighbors+j];
		}
		
		totalDistancesPerPoint[i]=totalDistancePoint;
		//for sorting key/value pairs
		keyValuePairPointIDDistance[i].pointID=i;
		keyValuePairPointIDDistance[i].distance=totalDistancePoint;
		
		
	}

	//sort the point IDs and values by key/value pair
	std::sort(keyValuePairPointIDDistance, keyValuePairPointIDDistance+databaseSize, compareByPointValue);
	
	//store the scores for each point in an array that will be printed
	int * outlierScoreArr=(int *)malloc(sizeof(int)*databaseSize);

	for(int i=0; i<databaseSize; i++)
	{
		int pointId=keyValuePairPointIDDistance[i].pointID;
		outlierScoreArr[pointId]=i;
	}

	//end outlier criterion 1
	///////////////////

	///////////////////
	//Outlier criterion 2: find the number of times a given point appears in another point's set
	//This can be thought of as exploiting the KNN graph


	//For each point, compute the number of times it appears in another KNN set (don't exclude a point finding itself, since all points find themselves)
	uint64_t * totalTimePointInOtherSet = (uint64_t *)malloc(sizeof(uint64_t)*databaseSize);
	struct keyValPointFrequencyKNNGraphStruct * keyValuePairPointIDFrequencyInSets = (struct keyValPointFrequencyKNNGraphStruct *)malloc(sizeof(keyValPointFrequencyKNNGraphStruct)*databaseSize);
	
	//initialize the struct and the array of counts for each object
	for (unsigned int i=0; i<databaseSize; i++)
	{
		keyValuePairPointIDFrequencyInSets[i].pointID=i;
		keyValuePairPointIDFrequencyInSets[i].numTimesInOtherSet=0;
		totalTimePointInOtherSet[i]=0;
	}



	for (unsigned int i=0; i<databaseSize; i++)
	{
		
		for (int j=0; j<k_Neighbors; j++)
		{
			unsigned int idx=nearestNeighborTable[i*k_Neighbors+j];
			keyValuePairPointIDFrequencyInSets[idx].numTimesInOtherSet++; //this will be sorted
			totalTimePointInOtherSet[idx]++; //this will not be sorted

		}
	}

	//sort the point IDs and values by key/value pair
	std::sort(keyValuePairPointIDFrequencyInSets, keyValuePairPointIDFrequencyInSets+databaseSize, compareByPointFrequencyInOtherSets);
	
	//store the scores (rankings) for each point in an array that will be printed
	unsigned int * outlierScoreFrequencyArr=(unsigned int *)malloc(sizeof(unsigned int)*databaseSize);

	for(int i=0; i<databaseSize; i++)
	{
		int pointId=keyValuePairPointIDFrequencyInSets[i].pointID;
		outlierScoreFrequencyArr[pointId]=i;
	}

	//end outlier criterion 2
	///////////////////

	//print to file for each point: its sum of distances to all points and its outlier ranking
	
	char fname[]="KNN_outlier_scores.txt";
	ofstream KNN_out;
	KNN_out.open(fname, ios::out);	

	printf("\nOutputting outlier scores to: %s\n", fname);
	KNN_out<<"#data point (line is the point id), col0: mean distance between point and its k neighbors,";
	KNN_out<<"col1: outlier ranking for col0, col2: the number of times the point appears in another set (in-degree), col3: the ranking for col2 \n";

	for(int i=0; i<databaseSize; i++)
	{
		KNN_out<<totalDistancesPerPoint[i]/((k_Neighbors*1.0)-1.0)<<", "<<outlierScoreArr[i]<<", "<<totalTimePointInOtherSet[i]<<", "<<outlierScoreFrequencyArr[i]<<endl;
	}

	KNN_out.close();

	//free all memory allocated in this function
	//criterion 1
	free(outlierScoreArr);
	free(totalDistancesPerPoint);
	free(keyValuePairPointIDDistance);
	//criterion 2
	free(outlierScoreFrequencyArr);
	free(keyValuePairPointIDFrequencyInSets);
	free(totalTimePointInOtherSet);
	

}

//find if neighbortable has at least k neighbors for each point
void procNeighborTableForkNN(std::vector<std::vector<DTYPE> > * NDdataPoints, neighborTableLookup * neighborTable, int k_Neighbors, std::vector<unsigned int> *queryPts)
{


	queryPts->clear();
	
	int min_neighbors=k_Neighbors+1;
	unsigned long int max_neighbors=0;

	unsigned long int totalNewMax=0;

	unsigned long int totalMoreThanK=0;

	for (int i=0; i<NDdataPoints->size(); i++){

		unsigned int num_neighbors=(neighborTable[i].indexmax-neighborTable[i].indexmin+1);
		//<= because we want k_neighbors+1 (each point finds itself)
		if (num_neighbors<=k_Neighbors)
		{
			queryPts->push_back(i);
			// printf("\nid: %d, num neighbors: %u", i,num_neighbors);

			//find smallest number of neighbors found
			if(min_neighbors>num_neighbors)
			{
				min_neighbors=num_neighbors;
			}

			
		}

		if (num_neighbors>(k_Neighbors+1))
		{
		totalMoreThanK+=num_neighbors-(k_Neighbors+1);
		}

		//find largest number of neighbors found
		if(max_neighbors<num_neighbors)
		{
			
			totalNewMax+=num_neighbors;
			max_neighbors=num_neighbors;
			// printf("\nMax neighbors: %u, total new max: %u", num_neighbors, totalNewMax);
		}




		// for (int j=neighborTable[i].indexmin; j<=neighborTable[i].indexmax; j++){
		// 	printf("%d, ",neighborTable[i].dataPtr[j]);
		// }

		// printf("\npoint id: %d, distances: ",i);
		// for (int j=neighborTable[i].indexmin; j<=neighborTable[i].indexmax; j++){
		// 	printf("%f, ",neighborTable[i].distancePtr[j]);
		// }


		
	}

	printf("\nMin neighbors found: %u",min_neighbors);cout.flush();
	printf("\nMax neighbors found: %lu",max_neighbors);cout.flush();
	printf("\nTotal neighbors > kNN+1 - kNN+1 (i.e., the waste): %lu", totalMoreThanK);cout.flush();
	printf("\nNum neighbors insufficient points: %lu",queryPts->size());cout.flush();

	// printf("\nInput epsilon: %f",*in_epsilon);

	// DTYPE in_eps_square=(*in_epsilon)*(*in_epsilon);

	// //based on the minimum number of neighbors found, compute the new epsilon as a function of volume
	// DTYPE vol_out_eps=(k_Neighbors*1.0/min_neighbors*1.0)*(in_eps_square);
	// DTYPE out_epsilon=sqrt(vol_out_eps);
	// printf("\nOutput epsilon: %f",out_epsilon);

	// return out_epsilon;


}



//find if neighbortable has at least k neighbors for each point
void procNeighborTableForkNNUsingDirectTable(std::vector<std::vector<DTYPE> > * NDdataPoints, int k_Neighbors, std::vector<unsigned int> *queryPts, int * nearestNeighborTable)
{


	queryPts->clear();
	
	uint64_t KNN=k_Neighbors+1;

	for (uint64_t i=0; i<(uint64_t)NDdataPoints->size()*(KNN); i+=(KNN)){
		//check to see if the neighbors haven't been found for each point, i.e., are set to -1
		if (nearestNeighborTable[i]==-1)
		{
			queryPts->push_back(i/KNN);

		}

		
	}

	// printf("\nMin neighbors found: %u",min_neighbors);cout.flush();
	// printf("\nMax neighbors found: %lu",max_neighbors);cout.flush();
	// printf("\nTotal neighbors > kNN+1 - kNN+1 (i.e., the waste): %lu", totalMoreThanK);cout.flush();
	printf("\nNum neighbors insufficient points: %lu",queryPts->size());cout.flush();

	// printf("\nInput epsilon: %f",*in_epsilon);

	// DTYPE in_eps_square=(*in_epsilon)*(*in_epsilon);

	// //based on the minimum number of neighbors found, compute the new epsilon as a function of volume
	// DTYPE vol_out_eps=(k_Neighbors*1.0/min_neighbors*1.0)*(in_eps_square);
	// DTYPE out_epsilon=sqrt(vol_out_eps);
	// printf("\nOutput epsilon: %f",out_epsilon);

	// return out_epsilon;


}


void procNeighborTableForkNNUsingDirectTableGPUOnly(std::vector<std::vector<DTYPE> > * NDdataPoints, int k_Neighbors, std::vector<unsigned int> *queryPts, int * nearestNeighborTable)
{


	std::vector<unsigned int> tmpQueries;
	uint64_t KNN=(uint64_t)k_Neighbors+1;

	//loop over the query array for the GPU
	//find if the queries have k neighbors
	//if not add to the query array
	for (uint64_t i=0; i<queryPts->size(); i++)
	// for (int i=0; i<100; i++)
	{
		//index in neighbortable result array
		uint64_t query=(uint64_t)(*queryPts)[i];
		uint64_t idx=query*KNN;
		if (nearestNeighborTable[idx]==-1)
		{
			// tmpQueries.push_back(idx/(k_Neighbors+1));
			tmpQueries.push_back((*queryPts)[i]);
		}
	}


	//clear the query points
	queryPts->clear();

	std::copy(tmpQueries.begin(),tmpQueries.end(),std::back_inserter(*queryPts));

	printf("\nNum neighbors insufficient points (GPU queries only): %lu",queryPts->size());cout.flush();

}



//find if neighbortable has at least k neighbors for each point
// void procNeighborTableForkNNNew(std::vector<std::vector<DTYPE> > * NDdataPoints, neighborTableLookup * neighborTable, int k_Neighbors, std::vector<unsigned int> *queryPtsOut)
// {

// 	//want to add to the query set only unique elements
// 	std::set<unsigned int >tmpQuerySet;
	
	
// 	int min_neighbors=k_Neighbors+1;
// 	unsigned long int max_neighbors=0;

// 	unsigned long int totalNewMax=0;

// 	unsigned long int totalMoreThanK=0;

// 	for (int i=0; i<NDdataPoints->size(); i++){

// 		unsigned int num_neighbors=(neighborTable[i].indexmax-neighborTable[i].indexmin+1);
// 		//<= because we want k_neighbors+1 (each point finds itself)
// 		if (num_neighbors<=k_Neighbors)
// 		{
// 			tmpQuerySet.insert(i);	
// 		}

		
// 	}


// 	printf("\nNum neighbors insufficient points: %lu",queryPts->size());cout.flush();




// }

void storeNeighborTableForkNN(std::vector<std::vector<DTYPE> > * NDdataPoints, neighborTableLookup * neighborTable, 
	int k_Neighbors, std::vector<unsigned int> *queryPts, int ** nearestNeighborTable, 
	DTYPE ** nearestNeighborTableDistances)
{



	//3-4 threads can saturate memory bandwidth for memcpy
	#pragma omp parallel for num_threads(4) shared(queryPts, neighborTable,nearestNeighborTable, nearestNeighborTableDistances)
	for (int i=0; i<queryPts->size(); i++){

		unsigned int pointID=(*queryPts)[i];
		int indexmax=k_Neighbors+1;

		unsigned int num_neighbors=(neighborTable[pointID].indexmax-neighborTable[pointID].indexmin+1);
		if (num_neighbors>k_Neighbors)
		{
			
			/*	
			int cnt=0;
			for (int j=neighborTable[pointID].indexmin; j<neighborTable[pointID].indexmin+indexmax; j++)
			{
				nearestNeighborTable[pointID][cnt]=neighborTable[pointID].dataPtr[j];	
				nearestNeighborTableDistances[pointID][cnt]=neighborTable[pointID].distancePtr[j];	
				cnt++;
			}				
			*/

			
			int idxMin=neighborTable[pointID].indexmin;
			int idxMax=neighborTable[pointID].indexmin+indexmax;
			std::copy(&neighborTable[pointID].dataPtr[idxMin], &neighborTable[pointID].dataPtr[idxMax], 
				nearestNeighborTable[pointID]);

			std::copy(&neighborTable[pointID].distancePtr[idxMin], &neighborTable[pointID].distancePtr[idxMax], 
				nearestNeighborTableDistances[pointID]);
			
			// int a[5]={0,0,0,0,0};
			// std::copy(a, a+5, nearestNeighborTable[pointID]);			
		}

		
	}



}



//use the index to compute the neighbors
void generateNeighborTableCPUPrototype(std::vector<std::vector <DTYPE> > *NDdataPoints, unsigned int queryPoint, DTYPE epsilon, grid * index, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, std::vector<uint64_t> * cellsToCheck, table * neighborTableCPUPrototype)
{
	
	for (int i=0; i<cellsToCheck->size(); i++){
		//find the id in the compressed grid index of the cell:
		//CHANGE TO BINARY SEARCH
		uint64_t GridIndex=0;

		struct gridCellLookup tmp;
		tmp.gridLinearID=(*cellsToCheck)[i];
		// struct gridCellLookup resultBinSearch;
		struct gridCellLookup * resultBinSearch=std::lower_bound(gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
		GridIndex=resultBinSearch->idx;


		for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
			DTYPE runningTotalDist=0;
			//printf("\nPoint id for dist calc: %d",k);
			unsigned int dataIdx=indexLookupArr[k];

			// printf("\nqueryPoint: %d, dataidx: %d",queryPoint,dataIdx);

			for (int l=0; l<GPUNUMDIM; l++){
			runningTotalDist+=((*NDdataPoints)[dataIdx][l]-(*NDdataPoints)[queryPoint][l])*((*NDdataPoints)[dataIdx][l]-(*NDdataPoints)[queryPoint][l]);
			}

			if (sqrt(runningTotalDist)<=epsilon){
				neighborTableCPUPrototype[queryPoint].neighbors.push_back(dataIdx);
			}
		}
}

	return;
}

//gridCellNDMaskOffsets -- this is NDIM*2 (min/max) for each grid dimension
void findNonEmptyCellsPrototype(DTYPE * point, DTYPE * epsilon, grid * index, DTYPE * minArr, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * nCells, unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets, std::vector<uint64_t> * cellsToCheck)
{

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];
	unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
	for (int i=0; i<NUMINDEXEDDIM; i++){
		nDCellIDs[i]=(point[i]-minArr[i])/(*epsilon);
		nDMinCellIDs[i]=max(0,nDCellIDs[i]-1); //boundary conditions (don't go beyond cell 0)
		nDMaxCellIDs[i]=min(nCells[i]-1,nDCellIDs[i]+1); //boundary conditions (don't go beyond the maximum number of cells)
		//printf("\n point ranges dim: %d, min,max: %d,%d", i,nDMinCellIDs[i],nDMaxCellIDs[i]);
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

		if(std::binary_search(gridCellNDMask+ gridCellNDMaskOffsets[(i*2)],
			gridCellNDMask+ gridCellNDMaskOffsets[(i*2)+1]+1,nDMinCellIDs[i])){ //extra +1 here is because we include the upper bound
			foundMin=1;
		}
		if(std::binary_search(gridCellNDMask+ gridCellNDMaskOffsets[(i*2)],
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
	
	// printf("\nbegun taking intersection!!");cout.flush();

	int cntIter=0;
	unsigned int indexes[NUMINDEXEDDIM];

	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0]=rangeFilteredCellIdsMin[0]; loopRng[0]<=rangeFilteredCellIdsMax[0]; loopRng[0]++)
	for (loopRng[1]=rangeFilteredCellIdsMin[1]; loopRng[1]<=rangeFilteredCellIdsMax[1]; loopRng[1]++)
	#if NUMINDEXEDDIM>=3
	for (loopRng[2]=rangeFilteredCellIdsMin[2]; loopRng[2]<=rangeFilteredCellIdsMax[2]; loopRng[2]++)
	#endif
	#if NUMINDEXEDDIM>=4
	for (loopRng[3]=rangeFilteredCellIdsMin[3]; loopRng[3]<=rangeFilteredCellIdsMax[3]; loopRng[3]++)
	#endif
	#if NUMINDEXEDDIM>=5
	for (loopRng[4]=rangeFilteredCellIdsMin[4]; loopRng[4]<=rangeFilteredCellIdsMax[4]; loopRng[4]++)
	#endif
	#if NUMINDEXEDDIM>=6
	for (loopRng[5]=rangeFilteredCellIdsMin[5]; loopRng[5]<=rangeFilteredCellIdsMax[5]; loopRng[5]++)
	#endif
	#if NUMINDEXEDDIM>=7
	for (loopRng[6]=rangeFilteredCellIdsMin[6]; loopRng[6]<=rangeFilteredCellIdsMax[6]; loopRng[6]++)
	#endif
	#if NUMINDEXEDDIM>=8
	for (loopRng[7]=rangeFilteredCellIdsMin[7]; loopRng[7]<=rangeFilteredCellIdsMax[7]; loopRng[7]++)
	#endif
	#if NUMINDEXEDDIM>=9
	for (loopRng[8]=rangeFilteredCellIdsMin[8]; loopRng[8]<=rangeFilteredCellIdsMax[8]; loopRng[8]++)
	#endif
	#if NUMINDEXEDDIM>=10
	for (loopRng[9]=rangeFilteredCellIdsMin[9]; loopRng[9]<=rangeFilteredCellIdsMax[9]; loopRng[9]++)
	#endif
	#if NUMINDEXEDDIM>=11
	for (loopRng[10]=rangeFilteredCellIdsMin[10]; loopRng[10]<=rangeFilteredCellIdsMax[10]; loopRng[10]++)
	#endif
	#if NUMINDEXEDDIM>=12
	for (loopRng[11]=rangeFilteredCellIdsMin[11]; loopRng[11]<=rangeFilteredCellIdsMax[11]; loopRng[11]++)
	#endif
	#if NUMINDEXEDDIM>=13
	for (loopRng[12]=rangeFilteredCellIdsMin[12]; loopRng[12]<=rangeFilteredCellIdsMax[12]; loopRng[12]++)
	#endif
	#if NUMINDEXEDDIM>=14
	for (loopRng[13]=rangeFilteredCellIdsMin[13]; loopRng[13]<=rangeFilteredCellIdsMax[13]; loopRng[13]++)
	#endif
	#if NUMINDEXEDDIM>=15
	for (loopRng[14]=rangeFilteredCellIdsMin[14]; loopRng[14]<=rangeFilteredCellIdsMax[14]; loopRng[14]++)
	#endif
	#if NUMINDEXEDDIM>=16
	for (loopRng[15]=rangeFilteredCellIdsMin[15]; loopRng[15]<=rangeFilteredCellIdsMax[15]; loopRng[15]++)
	#endif
	#if NUMINDEXEDDIM>=17
	for (loopRng[16]=rangeFilteredCellIdsMin[16]; loopRng[16]<=rangeFilteredCellIdsMax[16]; loopRng[16]++)
	#endif
	#if NUMINDEXEDDIM>=18
	for (loopRng[17]=rangeFilteredCellIdsMin[17]; loopRng[17]<=rangeFilteredCellIdsMax[17]; loopRng[17]++)
	#endif
	#if NUMINDEXEDDIM>=19
	for (loopRng[18]=rangeFilteredCellIdsMin[18]; loopRng[18]<=rangeFilteredCellIdsMax[18]; loopRng[18]++)
	#endif
	#if NUMINDEXEDDIM>=20
	for (loopRng[19]=rangeFilteredCellIdsMin[19]; loopRng[19]<=rangeFilteredCellIdsMax[19]; loopRng[19]++)
	#endif
	#if NUMINDEXEDDIM>=21
	for (loopRng[20]=rangeFilteredCellIdsMin[20]; loopRng[20]<=rangeFilteredCellIdsMax[20]; loopRng[20]++)
	#endif
	#if NUMINDEXEDDIM>=22
	for (loopRng[21]=rangeFilteredCellIdsMin[21]; loopRng[21]<=rangeFilteredCellIdsMax[21]; loopRng[21]++)
	#endif
	#if NUMINDEXEDDIM>=23
	for (loopRng[22]=rangeFilteredCellIdsMin[22]; loopRng[22]<=rangeFilteredCellIdsMax[22]; loopRng[22]++)
	#endif
	#if NUMINDEXEDDIM>=24
	for (loopRng[23]=rangeFilteredCellIdsMin[23]; loopRng[23]<=rangeFilteredCellIdsMax[23]; loopRng[23]++)
	#endif
	#if NUMINDEXEDDIM>=25
	for (loopRng[24]=rangeFilteredCellIdsMin[24]; loopRng[24]<=rangeFilteredCellIdsMax[24]; loopRng[24]++)
	#endif
	#if NUMINDEXEDDIM>=26
	for (loopRng[25]=rangeFilteredCellIdsMin[25]; loopRng[25]<=rangeFilteredCellIdsMax[25]; loopRng[25]++)
	#endif
	#if NUMINDEXEDDIM>=27
	for (loopRng[26]=rangeFilteredCellIdsMin[26]; loopRng[26]<=rangeFilteredCellIdsMax[26]; loopRng[26]++)
	#endif
	#if NUMINDEXEDDIM>=28
	for (loopRng[27]=rangeFilteredCellIdsMin[27]; loopRng[27]<=rangeFilteredCellIdsMax[27]; loopRng[27]++)
	#endif
	#if NUMINDEXEDDIM>=29
	for (loopRng[28]=rangeFilteredCellIdsMin[28]; loopRng[28]<=rangeFilteredCellIdsMax[28]; loopRng[28]++)
	#endif
	#if NUMINDEXEDDIM>=30
	for (loopRng[29]=rangeFilteredCellIdsMin[29]; loopRng[29]<=rangeFilteredCellIdsMax[29]; loopRng[29]++)
	#endif
	#if NUMINDEXEDDIM>=31
	for (loopRng[30]=rangeFilteredCellIdsMin[30]; loopRng[30]<=rangeFilteredCellIdsMax[30]; loopRng[30]++)
	#endif
	#if NUMINDEXEDDIM>=32
	for (loopRng[31]=rangeFilteredCellIdsMin[31]; loopRng[31]<=rangeFilteredCellIdsMax[31]; loopRng[31]++)
	#endif
	#if NUMINDEXEDDIM>=33
	for (loopRng[32]=rangeFilteredCellIdsMin[32]; loopRng[32]<=rangeFilteredCellIdsMax[32]; loopRng[32]++)
	#endif
	#if NUMINDEXEDDIM>=34
	for (loopRng[33]=rangeFilteredCellIdsMin[33]; loopRng[33]<=rangeFilteredCellIdsMax[33]; loopRng[33]++)
	#endif
	#if NUMINDEXEDDIM>=35
	for (loopRng[34]=rangeFilteredCellIdsMin[34]; loopRng[34]<=rangeFilteredCellIdsMax[34]; loopRng[34]++)
	#endif
	#if NUMINDEXEDDIM>=36
	for (loopRng[35]=rangeFilteredCellIdsMin[35]; loopRng[35]<=rangeFilteredCellIdsMax[35]; loopRng[35]++)
	#endif
	#if NUMINDEXEDDIM>=37
	for (loopRng[36]=rangeFilteredCellIdsMin[36]; loopRng[36]<=rangeFilteredCellIdsMax[36]; loopRng[36]++)
	#endif
	#if NUMINDEXEDDIM>=38
	for (loopRng[37]=rangeFilteredCellIdsMin[37]; loopRng[37]<=rangeFilteredCellIdsMax[37]; loopRng[37]++)
	#endif
	#if NUMINDEXEDDIM>=39
	for (loopRng[38]=rangeFilteredCellIdsMin[38]; loopRng[38]<=rangeFilteredCellIdsMax[38]; loopRng[38]++)
	#endif
	#if NUMINDEXEDDIM>=40
	for (loopRng[39]=rangeFilteredCellIdsMin[39]; loopRng[39]<=rangeFilteredCellIdsMax[39]; loopRng[39]++)
	#endif
	#if NUMINDEXEDDIM>=41
	for (loopRng[40]=rangeFilteredCellIdsMin[40]; loopRng[40]<=rangeFilteredCellIdsMax[40]; loopRng[40]++)
	#endif
	#if NUMINDEXEDDIM>=42
	for (loopRng[41]=rangeFilteredCellIdsMin[41]; loopRng[41]<=rangeFilteredCellIdsMax[41]; loopRng[41]++)
	#endif
	#if NUMINDEXEDDIM>=43
	for (loopRng[42]=rangeFilteredCellIdsMin[42]; loopRng[42]<=rangeFilteredCellIdsMax[42]; loopRng[42]++)
	#endif
	#if NUMINDEXEDDIM>=44
	for (loopRng[43]=rangeFilteredCellIdsMin[43]; loopRng[43]<=rangeFilteredCellIdsMax[43]; loopRng[43]++)
	#endif
	#if NUMINDEXEDDIM>=45
	for (loopRng[44]=rangeFilteredCellIdsMin[44]; loopRng[44]<=rangeFilteredCellIdsMax[44]; loopRng[44]++)
	#endif
	#if NUMINDEXEDDIM>=46
	for (loopRng[45]=rangeFilteredCellIdsMin[45]; loopRng[45]<=rangeFilteredCellIdsMax[45]; loopRng[45]++)
	#endif
	#if NUMINDEXEDDIM>=47
	for (loopRng[46]=rangeFilteredCellIdsMin[46]; loopRng[46]<=rangeFilteredCellIdsMax[46]; loopRng[46]++)
	#endif
	#if NUMINDEXEDDIM>=48
	for (loopRng[47]=rangeFilteredCellIdsMin[47]; loopRng[47]<=rangeFilteredCellIdsMax[47]; loopRng[47]++)
	#endif
	#if NUMINDEXEDDIM>=49
	for (loopRng[48]=rangeFilteredCellIdsMin[48]; loopRng[48]<=rangeFilteredCellIdsMax[48]; loopRng[48]++)
	#endif
	#if NUMINDEXEDDIM>=50
	for (loopRng[49]=rangeFilteredCellIdsMin[49]; loopRng[49]<=rangeFilteredCellIdsMax[49]; loopRng[49]++)
	#endif
	#if NUMINDEXEDDIM>=51
	for (loopRng[50]=rangeFilteredCellIdsMin[50]; loopRng[50]<=rangeFilteredCellIdsMax[50]; loopRng[50]++)
	#endif
	#if NUMINDEXEDDIM>=52
	for (loopRng[51]=rangeFilteredCellIdsMin[51]; loopRng[51]<=rangeFilteredCellIdsMax[51]; loopRng[51]++)
	#endif
	#if NUMINDEXEDDIM>=53
	for (loopRng[52]=rangeFilteredCellIdsMin[52]; loopRng[52]<=rangeFilteredCellIdsMax[52]; loopRng[52]++)
	#endif
	#if NUMINDEXEDDIM>=54
	for (loopRng[53]=rangeFilteredCellIdsMin[53]; loopRng[53]<=rangeFilteredCellIdsMax[53]; loopRng[53]++)
	#endif
	#if NUMINDEXEDDIM>=55
	for (loopRng[54]=rangeFilteredCellIdsMin[54]; loopRng[54]<=rangeFilteredCellIdsMax[54]; loopRng[54]++)
	#endif
	#if NUMINDEXEDDIM>=56
	for (loopRng[55]=rangeFilteredCellIdsMin[55]; loopRng[55]<=rangeFilteredCellIdsMax[55]; loopRng[55]++)
	#endif
	#if NUMINDEXEDDIM>=57
	for (loopRng[56]=rangeFilteredCellIdsMin[56]; loopRng[56]<=rangeFilteredCellIdsMax[56]; loopRng[56]++)
	#endif
	#if NUMINDEXEDDIM>=58
	for (loopRng[57]=rangeFilteredCellIdsMin[57]; loopRng[57]<=rangeFilteredCellIdsMax[57]; loopRng[57]++)
	#endif
	#if NUMINDEXEDDIM>=59
	for (loopRng[58]=rangeFilteredCellIdsMin[58]; loopRng[58]<=rangeFilteredCellIdsMax[58]; loopRng[58]++)
	#endif
	#if NUMINDEXEDDIM>=60
	for (loopRng[59]=rangeFilteredCellIdsMin[59]; loopRng[59]<=rangeFilteredCellIdsMax[59]; loopRng[59]++)
	#endif
	#if NUMINDEXEDDIM>=61
	for (loopRng[60]=rangeFilteredCellIdsMin[60]; loopRng[60]<=rangeFilteredCellIdsMax[60]; loopRng[60]++)
	#endif
	#if NUMINDEXEDDIM>=62
	for (loopRng[61]=rangeFilteredCellIdsMin[61]; loopRng[61]<=rangeFilteredCellIdsMax[61]; loopRng[61]++)
	#endif
	#if NUMINDEXEDDIM>=63
	for (loopRng[62]=rangeFilteredCellIdsMin[62]; loopRng[62]<=rangeFilteredCellIdsMax[62]; loopRng[62]++)
	#endif
	#if NUMINDEXEDDIM>=64
	for (loopRng[63]=rangeFilteredCellIdsMin[63]; loopRng[63]<=rangeFilteredCellIdsMax[63]; loopRng[63]++)
	#endif					
	{ //beginning of loop body
	
	for (int x=0; x<NUMINDEXEDDIM; x++){
	indexes[x]=loopRng[x];	
	}
	
	cntIter++;
	uint64_t calcLinearID=getLinearID_nDimensions(indexes, nCells, NUMINDEXEDDIM);
	//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says 
	//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

	//CHANGE THIS TO A BINARY SEARCH LATER	

	// for (int x=0; x<(*nNonEmptyCells);x++){
	// 	if (calcLinearID==gridCellLookupArr[x].gridLinearID){
	// 		cellsToCheck->push_back(calcLinearID); 
	// 	}
	// }
	
	struct gridCellLookup tmp;
	tmp.gridLinearID=calcLinearID;
	if (std::binary_search(gridCellLookupArr, gridCellLookupArr+ (*nNonEmptyCells), gridCellLookup(tmp))){
		//in the GPU implementation we go directly to computing neighbors so that we don't need to
		//store a buffer of the cells to check 
		cellsToCheck->push_back(calcLinearID); 
	}

	
	//printf("\nLinear id: %d",calcLinearID);
	} //end loop body

	// printf("\nloop iters for point: %d",cntIter);cout.flush();
	// printf("\nNum cells to check: %d",cellsToCheck->size());cout.flush();

}


/*
struct cmpDim{
        __host__ __device__
        bool operator()(const std::vector<std::vector <double>> a, const std::vector<std::vector<double>> b) {
#if GPUNUMDIM == NUMINDEXEDDIM
                return a[0] < b[0];
#else
                return a[GPUNUMDIM+1] < b[GPUNUMDIM+1];
#endif
        }
};
*/

//bool cmpPtDim(int a, int b) {
//#if GPUNUMDIM == NUMINDEXEDDIM
//	return NDdataPoints[a][0] < NDdataPoints[b][0];
//#else
//	return a[GPUNUMDIM+1] < b[GPUNUMDIM+1];
//#endif
//}

/*
struct cmpStruct {
	cmpStruct(std::vector <std::vector <DTYPE>> points) {this -> points = points;}
	bool operator() (int a, int b) {
		return points[a][0] < points[b][0];
	}

	std::vector<std::vector<DTYPE>> points;
};
*/




void populateNDGridIndexAndLookupArray(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr,  DTYPE* minArr, unsigned int * nCells, uint64_t totalCells, unsigned int * nNonEmptyCells)
{

	

	/////////////////////////////////
	//Populate grid lookup array
	//and corresponding indicies in the lookup array
	/////////////////////////////////
	printf("\n\n*****************************\nPopulating Grid Index and lookup array:\n*****************************\n");
	// printf("\nSize of dataset: %lu", NDdataPoints->size());


	///////////////////////////////
	//First, we need to figure out how many non-empty cells there will be
	//For memory allocation
	//Need to do a scan of the dataset and calculate this
	//Also need to keep track of the list of uniquie linear grid cell IDs for inserting into the grid
	///////////////////////////////
	std::set<uint64_t> uniqueGridCellLinearIds;
	std::vector<uint64_t>uniqueGridCellLinearIdsVect; //for random access

	for (int i=0; i<NDdataPoints->size(); i++){
		unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
		for (int j=0; j<NUMINDEXEDDIM; j++){
			tmpNDCellIdx[j]=(((*NDdataPoints)[i][j]-minArr[j])/epsilon);
		}
		uint64_t linearID=getLinearID_nDimensions(tmpNDCellIdx, nCells, NUMINDEXEDDIM);
		uniqueGridCellLinearIds.insert(linearID);

	}

	// printf("uniqueGridCellLinearIds: %d",uniqueGridCellLinearIds.size());

	




	//copy the set to the vector (sets can't do binary searches -- no random access)
	std::copy(uniqueGridCellLinearIds.begin(), uniqueGridCellLinearIds.end(), std::back_inserter(uniqueGridCellLinearIdsVect));
	


	
	

	


	std::vector<uint64_t> * gridElemIDs;
	gridElemIDs = new std::vector<uint64_t>[uniqueGridCellLinearIds.size()];

	//Create ND array mask:
	//This mask determines which cells in each dimension has points in them.
	std::set<unsigned int> NDArrMask[NUMINDEXEDDIM];
	
	vector<uint64_t>::iterator lower;
	
	
	for (int i=0; i<NDdataPoints->size(); i++){
		unsigned int tmpNDCellID[NUMINDEXEDDIM];
		for (int j=0; j<NUMINDEXEDDIM; j++){
			tmpNDCellID[j]=(((*NDdataPoints)[i][j]-minArr[j])/epsilon);

			//add value to the ND array mask
			
			NDArrMask[j].insert(tmpNDCellID[j]);
		}

		//get the linear id of the cell
		uint64_t linearID=getLinearID_nDimensions(tmpNDCellID, nCells, NUMINDEXEDDIM);
		//printf("\nlinear id: %d",linearID);
		if (linearID > totalCells){

			printf("\n\nERROR Linear ID is: %lu, total cells is only: %lu\n\n", linearID, totalCells);
		}

		//find the index in gridElemIds that corresponds to this grid cell linear id
		
		lower=std::lower_bound(uniqueGridCellLinearIdsVect.begin(), uniqueGridCellLinearIdsVect.end(),linearID);
		uint64_t gridIdx=lower - uniqueGridCellLinearIdsVect.begin();
		gridElemIDs[gridIdx].push_back(i);
	}


	



	///////////////////////////////
	//Here we fill a temporary index with points, and then copy the non-empty cells to the actual index
	///////////////////////////////
	
	struct grid * tmpIndex=new grid[uniqueGridCellLinearIdsVect.size()];

	int cnt=0;

	

	//populate temp index and lookup array

	for (int i=0; i<uniqueGridCellLinearIdsVect.size(); i++)
	{
			tmpIndex[i].indexmin=cnt;
			for (int j=0; j<gridElemIDs[i].size(); j++)
			{
				if (j>((NDdataPoints->size()-1)))
				{
					printf("\n\n***ERROR Value of a data point is larger than the dataset! %d\n\n", j);
					return;
				}
				indexLookupArr[cnt]=gridElemIDs[i][j]; 
				cnt++;
			}
			tmpIndex[i].indexmax=cnt-1;
	}

	// printf("\nExiting grid populate method early!");
	// return;

	printf("\nFull cells: %d (%f, fraction full)",(unsigned int)uniqueGridCellLinearIdsVect.size(), uniqueGridCellLinearIdsVect.size()*1.0/double(totalCells));
	printf("\nEmpty cells: %ld (%f, fraction empty)",totalCells-(unsigned int)uniqueGridCellLinearIdsVect.size(), (totalCells-uniqueGridCellLinearIdsVect.size()*1.0)/double(totalCells));
	
	*nNonEmptyCells=uniqueGridCellLinearIdsVect.size();


	printf("\nSize of index that would be sent to GPU (GiB) -- (if full index sent), excluding the data lookup arr: %f", (double)sizeof(struct grid)*(totalCells)/(1024.0*1024.0*1024.0));
	printf("\nSize of compressed index to be sent to GPU (GiB) , excluding the data and grid lookup arr: %f", (double)sizeof(struct grid)*(uniqueGridCellLinearIdsVect.size()*1.0)/(1024.0*1024.0*1024.0));




	//////////////
	
	/////////////////////////////////////////
	//copy the tmp index into the actual index that only has the non-empty cells
	

	//allocate memory for the index that will be sent to the GPU
	*index=new grid[uniqueGridCellLinearIdsVect.size()];
	*gridCellLookupArr= new struct gridCellLookup[uniqueGridCellLinearIdsVect.size()];

	

	// cmpStruct theStruct(*NDdataPoints);

	

	// #pragma omp parallel for num_threads(4) 
	for (int i=0; i<uniqueGridCellLinearIdsVect.size(); i++){
			(*index)[i].indexmin=tmpIndex[i].indexmin;
			(*index)[i].indexmax=tmpIndex[i].indexmax;
			(*gridCellLookupArr)[i].idx=i;
			(*gridCellLookupArr)[i].gridLinearID=uniqueGridCellLinearIdsVect[i];
	}

	

	printf("\nWhen copying from entire index to compressed index: number of non-empty cells: %lu",uniqueGridCellLinearIdsVect.size());
		
	//copy NDArrMask from set to an array

	//find the total size and allocate the array
	/*
	unsigned int cntNDOffsets=0;
	unsigned int cntNonEmptyNDMask=0;
	for (int i=0; i<NUMINDEXEDDIM; i++){
		cntNonEmptyNDMask+=NDArrMask[i].size();
	}	
	*gridCellNDMask = new unsigned int[cntNonEmptyNDMask];
	
	*nNDMaskElems=cntNonEmptyNDMask;

	
	//copy the offsets to the array
	for (int i=0; i<NUMINDEXEDDIM; i++){
		//Min
		gridCellNDMaskOffsets[(i*2)]=cntNDOffsets;
		for (std::set<unsigned int>::iterator it=NDArrMask[i].begin(); it!=NDArrMask[i].end(); ++it){
    		(*gridCellNDMask)[cntNDOffsets]=*it;
    		cntNDOffsets++;
		}
		//max
		gridCellNDMaskOffsets[(i*2)+1]=cntNDOffsets-1;
	}
	*/
	



	


	
	//print for testing -- full index
	/*
	int count=0;
	for (int i=0; i<totalCells; i++)
	{
		
		printf("\nLinear id: %d, index min: %d, index max: %d \nids: ", i, tmpIndex[i].indexmin, tmpIndex[i].indexmax);
		if (tmpIndex[i].indexmin!=-1 && tmpIndex[i].indexmax!=-1)
		{
			for (int j=tmpIndex[i].indexmin; j<=tmpIndex[i].indexmax; j++)
			{
				count++;
				printf("%d, ",indexLookupArr[j]);
			}
		}

	}
	printf("\ntest number of data elems: %d", count);
	printf("\n------------------------------");
	printf("\nCompressed index:");


	//print for testing -- compressed index
	
	int countCompressed=0;
	for (int i=0; i<cntFullCells; i++){
		printf("\nLinear id (from grid lookup arr): %d, index min: %d, index max: %d \nids: ", (*gridCellLookupArr)[i].gridLinearID, (*index)[i].indexmin, (*index)[i].indexmax);
		
			for (int j=(*index)[i].indexmin; j<=(*index)[i].indexmax; j++)
			{
				countCompressed++;
				printf("%d, ",indexLookupArr[j]);
			}
	}
	printf("\ntest number of data elems: %d", countCompressed);
	
	printf("\nND arr mask -- set: ");
	for (int i=0; i<NUMINDEXEDDIM; i++){
		printf("\NUMINDEXEDDIM: %d :",i);
		for (std::set<unsigned int>::iterator it=NDArrMask[i].begin(); it!=NDArrMask[i].end(); ++it){
    		std::cout << *it << ',';
		}
			
	}

	printf("\nND arr mask: -- arr to be sent to GPU");
	for (int i=0; i<NUMINDEXEDDIM; i++){
		printf("\NUMINDEXEDDIM: %d :",i);
		for (int j=gridCellNDMaskOffsets[(i*2)]; j<=gridCellNDMaskOffsets[(i*2)+1]; j++){
    		printf("%d, ",(*gridCellNDMask)[j]);
		}
			
	}
	
	printf("\nND arr mask offsets [min,max,min,max,min,max...]: ");
	for (int i=0; i<NUMINDEXEDDIM; i++){
    		printf("min/max: %d,%d ", gridCellNDMaskOffsets[(i*2)],gridCellNDMaskOffsets[(i*2)+1]);
		}


	printf("\n------------------------------");
	*/

	delete [] tmpIndex;
		


} //end function populate grid index and lookup array



void populateNDGridIndexAndLookupArrayParallel(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr,  DTYPE* minArr, unsigned int * nCells, uint64_t totalCells, unsigned int * nNonEmptyCells)
{

	int NUMTHREADSINDEX=4;

	/////////////////////////////////
	//Populate grid lookup array
	//and corresponding indicies in the lookup array
	/////////////////////////////////
	printf("\n\n*****************************\nPopulating Grid Index and lookup array:\n*****************************\n");
	// printf("\nSize of dataset: %lu", NDdataPoints->size());


	///////////////////////////////
	//First, we need to figure out how many non-empty cells there will be
	//For memory allocation
	//Need to do a scan of the dataset and calculate this
	//Also need to keep track of the list of uniquie linear grid cell IDs for inserting into the grid
	///////////////////////////////
	
	//instead of using a set, do data deduplication manually by sorting
	std::vector<uint64_t> allGridCellLinearIds;

	// std::set<uint64_t> uniqueGridCellLinearIds;
	std::vector<uint64_t>uniqueGridCellLinearIdsVect; //for random access

	

	//original
	/*
	// std::set<uint64_t> uniqueGridCellLinearIds;
	// std::vector<uint64_t>uniqueGridCellLinearIdsVect; //for random access

	for (int i=0; i<NDdataPoints->size(); i++){
		unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
		for (int j=0; j<NUMINDEXEDDIM; j++){
			tmpNDCellIdx[j]=(((*NDdataPoints)[i][j]-minArr[j])/epsilon);
		}
		uint64_t linearID=getLinearID_nDimensions(tmpNDCellIdx, nCells, NUMINDEXEDDIM);
		uniqueGridCellLinearIds.insert(linearID);

	}
	
	fprintf(stderr,"\nsize of set unique: %u",uniqueGridCellLinearIds.size());
	*/
	////////// end original

	for (int i=0; i<NDdataPoints->size(); i++){
		unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
		for (int j=0; j<NUMINDEXEDDIM; j++){
			tmpNDCellIdx[j]=(((*NDdataPoints)[i][j]-minArr[j])/epsilon);
		}
		uint64_t linearID=getLinearID_nDimensions(tmpNDCellIdx, nCells, NUMINDEXEDDIM);
		allGridCellLinearIds.push_back(linearID);

	}

		
	//replace with parallel sort later
	// std::sort(allGridCellLinearIds.begin(),allGridCellLinearIds.end());
	
	//call parallel sort, but is in my custom function because nvcc conflicts with parallel mode extensions
	sortLinearIds(&allGridCellLinearIds);
	
	
	uniqueGridCellLinearIdsVect.push_back(allGridCellLinearIds[0]);
	for (int i=1; i<allGridCellLinearIds.size(); i++)
	{
		if (allGridCellLinearIds[i]!=allGridCellLinearIds[i-1])
		{
			uniqueGridCellLinearIdsVect.push_back(allGridCellLinearIds[i]);
		}
	}
	
	

	




	//copy the set to the vector (sets can't do binary searches -- no random access)
	//original
	// std::copy(uniqueGridCellLinearIds.begin(), uniqueGridCellLinearIds.end(), std::back_inserter(uniqueGridCellLinearIdsVect));
	


	
	

	

	
	//before separating loop
	/*
	std::vector<uint64_t> * gridElemIDs;
	gridElemIDs = new std::vector<uint64_t>[uniqueGridCellLinearIds.size()];

	//Create ND array mask:
	//This mask determines which cells in each dimension has points in them.
	std::set<unsigned int> NDArrMask[NUMINDEXEDDIM];
	
	vector<uint64_t>::iterator lower;
	
	
	for (int i=0; i<NDdataPoints->size(); i++){
		unsigned int tmpNDCellID[NUMINDEXEDDIM];
		for (int j=0; j<NUMINDEXEDDIM; j++){
			tmpNDCellID[j]=(((*NDdataPoints)[i][j]-minArr[j])/epsilon);

			//add value to the ND array mask
			
			
			NDArrMask[j].insert(tmpNDCellID[j]);
			
		}

		//get the linear id of the cell
		uint64_t linearID=getLinearID_nDimensions(tmpNDCellID, nCells, NUMINDEXEDDIM);
		//printf("\nlinear id: %d",linearID);
		if (linearID > totalCells){

			printf("\n\nERROR Linear ID is: %lu, total cells is only: %lu\n\n", linearID, totalCells);
		}

		//find the index in gridElemIds that corresponds to this grid cell linear id
		
		lower=std::lower_bound(uniqueGridCellLinearIdsVect.begin(), uniqueGridCellLinearIdsVect.end(),linearID);
		uint64_t gridIdx=lower - uniqueGridCellLinearIdsVect.begin();
		gridElemIDs[gridIdx].push_back(i);
	}

	
	*/

	
	
	//temp vectors so that arrays canconcurrently write without mutual exclusion
	// std::vector<uint64_t> gridElemIDsTmpThreads[NUMTHREADSINDEX][uniqueGridCellLinearIds.size()];
	std::vector<uint64_t> ** gridElemIDsTmpThreads;	
	gridElemIDsTmpThreads=new std::vector<uint64_t>*[NUMTHREADSINDEX]; 		
	for (int i=0; i<NUMTHREADSINDEX; i++)
	{
		gridElemIDsTmpThreads[i]=new std::vector<uint64_t>[uniqueGridCellLinearIdsVect.size()];
	}

	//threads will store the ids in here
	std::vector<uint64_t> * gridElemIDs;
	gridElemIDs = new std::vector<uint64_t>[uniqueGridCellLinearIdsVect.size()];





	//Create ND array mask:
	//This mask determines which cells in each dimension has points in them.
	

	
	std::set<unsigned int> NDArrMask[NUMINDEXEDDIM];
	
	for (int i=0; i<NDdataPoints->size(); i++){
		unsigned int tmpNDCellID[NUMINDEXEDDIM];
		for (int j=0; j<NUMINDEXEDDIM; j++){
			tmpNDCellID[j]=(((*NDdataPoints)[i][j]-minArr[j])/epsilon);

			//add value to the ND array mask
			
			
			NDArrMask[j].insert(tmpNDCellID[j]);
			
		}
	}	



	


	
	#pragma omp parallel num_threads(NUMTHREADSINDEX)  
	{
		int tid=omp_get_thread_num();
		#pragma omp for 
		for (int i=0; i<NDdataPoints->size(); i++){
			
			
			unsigned int tmpNDCellID[NUMINDEXEDDIM];
			for (int j=0; j<NUMINDEXEDDIM; j++){
				tmpNDCellID[j]=(((*NDdataPoints)[i][j]-minArr[j])/epsilon);			
			}

			//get the linear id of the cell
			uint64_t linearID=getLinearID_nDimensions(tmpNDCellID, nCells, NUMINDEXEDDIM);
			//printf("\nlinear id: %d",linearID);
			if (linearID > totalCells){

				printf("\n\nERROR Linear ID is: %lu, total cells is only: %lu\n\n", linearID, totalCells);
			}

			//find the index in gridElemIds that corresponds to this grid cell linear id
			
			vector<uint64_t>::iterator lower=std::lower_bound(uniqueGridCellLinearIdsVect.begin(), uniqueGridCellLinearIdsVect.end(),linearID);
			uint64_t gridIdx=lower - uniqueGridCellLinearIdsVect.begin();
			
			//original
			// gridElemIDs[gridIdx].push_back(i);

			gridElemIDsTmpThreads[tid][gridIdx].push_back(i);
			
		}
	}


	

	//copy the grid elem ids from the temp vectors 
	for (int i=0; i<NUMTHREADSINDEX; i++)
	{
		for (int j=0; j<uniqueGridCellLinearIdsVect.size(); j++)
		{
			for (int k=0; k<gridElemIDsTmpThreads[i][j].size(); k++)
			{
					gridElemIDs[j].push_back(gridElemIDsTmpThreads[i][j][k]);				
			}	
		}
	}

	
	




	
	///////////////////////////////
	//Here we fill a temporary index with points, and then copy the non-empty cells to the actual index
	///////////////////////////////
	
	struct grid * tmpIndex=new grid[uniqueGridCellLinearIdsVect.size()];

	int cnt=0;

	

	//populate temp index and lookup array

	for (int i=0; i<uniqueGridCellLinearIdsVect.size(); i++)
	{
			tmpIndex[i].indexmin=cnt;
			for (int j=0; j<gridElemIDs[i].size(); j++)
			{
				if (j>((NDdataPoints->size()-1)))
				{
					printf("\n\n***ERROR Value of a data point is larger than the dataset! %d\n\n", j);
					return;
				}
				indexLookupArr[cnt]=gridElemIDs[i][j]; 
				cnt++;
			}
			tmpIndex[i].indexmax=cnt-1;
	}

	// printf("\nExiting grid populate method early!");
	// return;

	printf("\nFull cells: %d (%f, fraction full)",(unsigned int)uniqueGridCellLinearIdsVect.size(), uniqueGridCellLinearIdsVect.size()*1.0/double(totalCells));
	printf("\nEmpty cells: %ld (%f, fraction empty)",totalCells-(unsigned int)uniqueGridCellLinearIdsVect.size(), (totalCells-uniqueGridCellLinearIdsVect.size()*1.0)/double(totalCells));
	
	*nNonEmptyCells=uniqueGridCellLinearIdsVect.size();


	printf("\nSize of index that would be sent to GPU (GiB) -- (if full index sent), excluding the data lookup arr: %f", (double)sizeof(struct grid)*(totalCells)/(1024.0*1024.0*1024.0));
	printf("\nSize of compressed index to be sent to GPU (GiB) , excluding the data and grid lookup arr: %f", (double)sizeof(struct grid)*(uniqueGridCellLinearIdsVect.size()*1.0)/(1024.0*1024.0*1024.0));


	

	//////////////
	
	/////////////////////////////////////////
	//copy the tmp index into the actual index that only has the non-empty cells


	//allocate memory for the index that will be sent to the GPU
	*index=new grid[uniqueGridCellLinearIdsVect.size()];
	*gridCellLookupArr= new struct gridCellLookup[uniqueGridCellLinearIdsVect.size()];

	

	// cmpStruct theStruct(*NDdataPoints);
	

	
	for (int i=0; i<uniqueGridCellLinearIdsVect.size(); i++){
			(*index)[i].indexmin=tmpIndex[i].indexmin;
			(*index)[i].indexmax=tmpIndex[i].indexmax;
			(*gridCellLookupArr)[i].idx=i;
			(*gridCellLookupArr)[i].gridLinearID=uniqueGridCellLinearIdsVect[i];
	}
	
	

	printf("\nWhen copying from entire index to compressed index: number of non-empty cells: %lu",uniqueGridCellLinearIdsVect.size());
		
	//copy NDArrMask from set to an array

	//find the total size and allocate the array
	/*	
	unsigned int cntNDOffsets=0;
	unsigned int cntNonEmptyNDMask=0;
	for (int i=0; i<NUMINDEXEDDIM; i++){
		cntNonEmptyNDMask+=NDArrMask[i].size();
	}	
	*gridCellNDMask = new unsigned int[cntNonEmptyNDMask];
	
	*nNDMaskElems=cntNonEmptyNDMask;

	

	//copy the offsets to the array
	for (int i=0; i<NUMINDEXEDDIM; i++){
		//Min
		gridCellNDMaskOffsets[(i*2)]=cntNDOffsets;
		for (std::set<unsigned int>::iterator it=NDArrMask[i].begin(); it!=NDArrMask[i].end(); ++it){
    		(*gridCellNDMask)[cntNDOffsets]=*it;
    		cntNDOffsets++;
		}
		//max
		gridCellNDMaskOffsets[(i*2)+1]=cntNDOffsets-1;
	}
	*/
	
	
	
	

	


	

	//print for testing -- full index
	/*
	int count=0;
	for (int i=0; i<totalCells; i++)
	{
		
		printf("\nLinear id: %d, index min: %d, index max: %d \nids: ", i, tmpIndex[i].indexmin, tmpIndex[i].indexmax);
		if (tmpIndex[i].indexmin!=-1 && tmpIndex[i].indexmax!=-1)
		{
			for (int j=tmpIndex[i].indexmin; j<=tmpIndex[i].indexmax; j++)
			{
				count++;
				printf("%d, ",indexLookupArr[j]);
			}
		}

	}
	printf("\ntest number of data elems: %d", count);
	printf("\n------------------------------");
	printf("\nCompressed index:");


	//print for testing -- compressed index
	
	int countCompressed=0;
	for (int i=0; i<cntFullCells; i++){
		printf("\nLinear id (from grid lookup arr): %d, index min: %d, index max: %d \nids: ", (*gridCellLookupArr)[i].gridLinearID, (*index)[i].indexmin, (*index)[i].indexmax);
		
			for (int j=(*index)[i].indexmin; j<=(*index)[i].indexmax; j++)
			{
				countCompressed++;
				printf("%d, ",indexLookupArr[j]);
			}
	}
	printf("\ntest number of data elems: %d", countCompressed);
	
	printf("\nND arr mask -- set: ");
	for (int i=0; i<NUMINDEXEDDIM; i++){
		printf("\NUMINDEXEDDIM: %d :",i);
		for (std::set<unsigned int>::iterator it=NDArrMask[i].begin(); it!=NDArrMask[i].end(); ++it){
    		std::cout << *it << ',';
		}
			
	}

	printf("\nND arr mask: -- arr to be sent to GPU");
	for (int i=0; i<NUMINDEXEDDIM; i++){
		printf("\NUMINDEXEDDIM: %d :",i);
		for (int j=gridCellNDMaskOffsets[(i*2)]; j<=gridCellNDMaskOffsets[(i*2)+1]; j++){
    		printf("%d, ",(*gridCellNDMask)[j]);
		}
			
	}
	
	printf("\nND arr mask offsets [min,max,min,max,min,max...]: ");
	for (int i=0; i<NUMINDEXEDDIM; i++){
    		printf("min/max: %d,%d ", gridCellNDMaskOffsets[(i*2)],gridCellNDMaskOffsets[(i*2)+1]);
		}


	printf("\n------------------------------");
	*/

	delete [] tmpIndex;
		


} //end function populate grid index and lookup array

//determines the linearized ID for a point in n-dimensions
//indexes: the indexes in the ND array: e.g., arr[4][5][6]
//dimLen: the length of each array e.g., arr[10][10][10]
//nDimensions: the number of dimensions


uint64_t getLinearID_nDimensions(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions) {
    // int i;
    // uint64_t offset = 0;
    // for( i = 0; i < nDimensions; i++ ) {
    //     offset += (uint64_t)pow(dimLen[i],i) * (uint64_t)indexes[nDimensions - (i + 1)];
    // }
    // return offset;

    uint64_t index = 0;
	uint64_t multiplier = 1;
	for (int i = 0; i<nDimensions; i++){
  	index += (uint64_t)indexes[i] * multiplier;
  	multiplier *= dimLen[i];
	}

	return index;
}



//testing this one
// uint64_t getLinearID_nDimensions(uint64_t * indexes, unsigned int * dimLen, unsigned int nDimensions) {
// 	uint64_t index = 0;
// 	uint64_t multiplier = 1;
// 	for (int i = 0;i<nDimensions;i++)
// 	{
// 	  index += indexes[i] * multiplier;
// 	  multiplier *= (uint64_t)dimLen[i];
// 	}
// 	// printf("\nLinear Index: %lld",index);
// 	return index;
// }




//min arr- the minimum value of the points in each dimensions - epsilon
//we can use this as an offset to calculate where points are located in the grid
//max arr- the maximum value of the points in each dimensions + epsilon 
//returns the time component of sorting the dimensions when SORT=1
void generateNDGridDimensions(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, DTYPE* minArr, DTYPE* maxArr, unsigned int * nCells, uint64_t * totalCells)
{

	printf("\n\n*****************************\nGenerating grid dimensions.\n*****************************\n");
	
	
	/////////////////////////////////
	//calculate the min and max points in each dimension
	/////////////////////////////////


	printf("\nNumber of dimensions data: %d, Number of dimensions indexed: %d", GPUNUMDIM, NUMINDEXEDDIM);
	
	//make the min/max values for each grid dimension the first data element
	for (int j=0; j<NUMINDEXEDDIM; j++){
		minArr[j]=(*NDdataPoints)[0][j];
		maxArr[j]=(*NDdataPoints)[0][j];
	}



	for (int i=1; i<NDdataPoints->size(); i++)
	{
		for (int j=0; j<NUMINDEXEDDIM; j++){
		if ((*NDdataPoints)[i][j]<minArr[j]){
			minArr[j]=(*NDdataPoints)[i][j];
		}
		if ((*NDdataPoints)[i][j]>maxArr[j]){
			maxArr[j]=(*NDdataPoints)[i][j];
		}	
		}
	}	
		

	printf("\n");
	for (int j=0; j<NUMINDEXEDDIM; j++){
		printf("Data Dim: %d, min/max: %f,%f\n",j,minArr[j],maxArr[j]);
	}	

	//add buffer around each dim so no weirdness later with putting data into cells
	for (int j=0; j<NUMINDEXEDDIM; j++){
		minArr[j]-=epsilon;
		maxArr[j]+=epsilon;
	}	

	for (int j=0; j<NUMINDEXEDDIM; j++){
		printf("Appended by epsilon Dim: %d, min/max: %f,%f\n",j,minArr[j],maxArr[j]);
	}	
	
	//calculate the number of cells:
	for (int j=0; j<NUMINDEXEDDIM; j++){
		nCells[j]=ceil((maxArr[j]-minArr[j])/epsilon);
		printf("Number of cells dim: %d: %d\n",j,nCells[j]);
	}

	//calc total cells: num cells in each dim multiplied
	uint64_t tmpTotalCells=nCells[0];
	for (int j=1; j<NUMINDEXEDDIM; j++){
		tmpTotalCells*=nCells[j];
	}

	*totalCells=tmpTotalCells;

}



//CPU brute force
void CPUBruteForceTable(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, table * neighborTable, unsigned int * totalNeighbors)
{
	DTYPE runningDist=0;
	unsigned int runningNeighbors=0;
	for (int i=0; i<NDdataPoints->size(); i++)
	{
		neighborTable[i].pointID=i;
		for (int j=0; j<NDdataPoints->size(); j++)
		{
			runningDist=0;
			for (int k=0; k<GPUNUMDIM; k++){
				runningDist+=((*NDdataPoints)[i][k]-(*NDdataPoints)[j][k])*((*NDdataPoints)[i][k]-(*NDdataPoints)[j][k]);
			}
			
			//if within epsilon:
			if ((sqrt(runningDist))<=epsilon){
				neighborTable[i].neighbors.push_back(j);
				runningNeighbors++;
			}
		}

	}
	//update the total neighbor count
	(*totalNeighbors)=runningNeighbors;

}

//reorders the input data by variance of each dimension
void ReorderByDimension(std::vector<std::vector <DTYPE> > *NDdataPoints)
{
	
	double tstart_sort=omp_get_wtime();
	DTYPE sums[GPUNUMDIM];
	DTYPE average[GPUNUMDIM];
	struct dim_reorder_sort dim_variance[GPUNUMDIM];
	for (int i=0; i< GPUNUMDIM; i++){
		sums[i]=0;
		average[i]=0;
	}

	DTYPE greatest_variance=0;
	int greatest_variance_dim=0;

	
	int sample=100;
	DTYPE inv_sample=1.0/(sample*1.0);
	printf("\nCalculating variance based on on the following fraction of pts: %f",inv_sample);
	double tvariancestart=omp_get_wtime();
		//calculate the variance in each dimension	
		for (int i=0; i<GPUNUMDIM; i++)
		{
			//first calculate the average in the dimension:
			//only use every 10th point
			for (int j=0; j<(*NDdataPoints).size(); j+=sample)
			{
			sums[i]+=(*NDdataPoints)[j][i];
			}


			average[i]=(sums[i])/((*NDdataPoints).size()*inv_sample);
			// printf("\nAverage in dim: %d, %f",i,average[i]);

			//Next calculate the std. deviation
			sums[i]=0; //reuse this for other sums
			for (int j=0; j<(*NDdataPoints).size(); j+=sample)
			{
			sums[i]+=(((*NDdataPoints)[j][i])-average[i])*(((*NDdataPoints)[j][i])-average[i]);
			}
			
			dim_variance[i].variance=sums[i]/((*NDdataPoints).size()*inv_sample);
			dim_variance[i].dim=i;
			
			// printf("\nDim:%d, variance: %f",dim_variance[i].dim,dim_variance[i].variance);

			if(greatest_variance<dim_variance[i].variance)
			{
				greatest_variance=dim_variance[i].variance;
				greatest_variance_dim=i;
			}
		}


	// double tvarianceend=omp_get_wtime();
	// printf("\nTime to compute variance only: %f",tvarianceend - tvariancestart);
	//sort based on variance in dimension:

	// double tstartsortreorder=omp_get_wtime();
	std::sort(dim_variance,dim_variance+GPUNUMDIM,compareByDimVariance); 	

	for (int i=0; i<GPUNUMDIM; i++)
	{
		printf("\nReodering dimension by: dim: %d, variance: %f",dim_variance[i].dim,dim_variance[i].variance);
	}

	printf("\nDimension with greatest variance: %d",greatest_variance_dim);

	//copy the database
	// double * tmp_database= (double *)malloc(sizeof(double)*(*NDdataPoints).size()*(GPUNUMDIM));  
	// std::copy(database, database+((*DBSIZE)*(GPUNUMDIM)),tmp_database);
	std::vector<std::vector <DTYPE> > tmp_database;

	//copy data into temp database
	tmp_database=(*NDdataPoints);

	
	
	#pragma omp parallel for num_threads(5) shared(NDdataPoints, tmp_database)
	for (int j=0; j<GPUNUMDIM; j++){

		int originDim=dim_variance[j].dim;	
		for (int i=0; i<(*NDdataPoints).size(); i++)
		{	
			(*NDdataPoints)[i][j]=tmp_database[i][originDim];
		}
	}

	double tend_sort=omp_get_wtime();
	// double tendsortreorder=omp_get_wtime();
	// printf("\nTime to sort/reorder only: %f",tendsortreorder-tstartsortreorder);
	double timecomponent=tend_sort - tstart_sort;
	printf("\nTime to reorder cols by variance (this gets added to the time because its an optimization): %f",timecomponent);
	
}



double estimateEpsilon(std::vector<std::vector<DTYPE> > * NDdataPoints, unsigned int k_neighbors)
{

	double minArrData[NUMINDEXEDDIM];
	double maxArrData[NUMINDEXEDDIM];

	for (int i=0; i<NUMINDEXEDDIM; i++)
    {
      minArrData[i]=INT_MAX;
      maxArrData[i]=INT_MIN;
    }



		//compute the min/max of the number of indexed dimensions
		for (int i=0; i<NDdataPoints->size(); i++){
			for (int j=0; j<NUMINDEXEDDIM; j++){
		        if ((*NDdataPoints)[i][j]<minArrData[j])
		        {
		          minArrData[j]=(*NDdataPoints)[i][j];
		        }
		        if ((*NDdataPoints)[i][j]>maxArrData[j])
		        {
		          maxArrData[j]=(*NDdataPoints)[i][j];
		        }
			}
		}


		for (int j=0; j<NUMINDEXEDDIM; j++){
			printf("\nMin/max of each dimension indexed. Dim: %d: %f, %f",j,minArrData[j],maxArrData[j]);
		}

	    
        


	double totalVolume=maxArrData[0]-minArrData[0];
	for (int i=1; i<NUMINDEXEDDIM; i++)
	{
		totalVolume*=maxArrData[i]-minArrData[i];				
		printf("\nRunning total volume: %0.14f",totalVolume);
	}

	printf("\nTotal volume: %0.14f",totalVolume);

	printf("\nNum data points: %lu",NDdataPoints->size());
	double density=(NDdataPoints->size()*1.0)/totalVolume;

	printf("\nDensity: %0.14f",density);

	double volumeForkNN=(k_neighbors*1.0)/density;

	printf("\nVolume for knn: %0.14f",volumeForkNN);

	return sqrt(volumeForkNN);



}





//checks to see if the ratio of the total cells searched to the total cells is >=0.9
//If so, we set the flag to brute force search
//since we append by epsilon, this means that we would have no selectivity in the dimensions
bool checkIndexSelectivity(unsigned int * nCells)
{
	// bool flag=false;
	int cnt=0;

	for (int i=0; i<NUMINDEXEDDIM; i++)
	{
		//testing this one:	
		if (nCells[i]>5) //5 means 3 cells + the 2 appended ones. Need at more than 3 cells (excluding appended ones) to have selectivity
		{
			cnt++;
		}
	}



	double cellsInASearch=powf(3.0,1.0*NUMINDEXEDDIM); 
	// double fractionSearched=cellsInASearch/(1.0*totalCells);

	// printf("\n[Checking Index Selectivity] Total cells (excluding appended ones): %lu, Cells in a search: %f, Fraction: %f",totalCells,cellsInASearch,fractionSearched);
	printf("\n[Checking Index Selectivity] Testing criteria: that as long as 1 dimension has more than 3 cells, then it prunes the search. Num dim >3 cells: %d", cnt);

	if (cnt==0)
	{
		return (1);
	}
	else
	{
		return(0);
	}


	/*
	//when using the fraction of cells
	if (fractionSearched>=1.0)
	{
		printf("\nIndex does not yield enough selectivity, returning true to brute force next iteration");
		flag=true;
	}
	

	return (flag);
	*/
}


void splitWork(unsigned int k_neighbors, std::vector<unsigned int> * queriesCPU, std::vector<unsigned int> * queriesGPU, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, struct grid * index)
{





		//We split the work based on the number of points in a cell
		//But the volume of an n-sphere inside of a cube decreases with dimension
		//so we need to increase the number of points that we say we need in order to
		//run on the GPU or CPU
		
		
		//we have the k-dist information, but we use all of the points in a cell, not within the hypersphere defined by epsilon	
		double volume_unit_ncube=powf(2*1,NUMINDEXEDDIM); //hypercube with radius 1 at the center (2* radius)

		// https://en.wikipedia.org/wiki/N-sphere
		// sanity check: https://keisan.casio.com/exec/system/1223381019
		double a=(powf(M_PI,(NUMINDEXEDDIM*0.5))*powf(1,NUMINDEXEDDIM));
		double b=(tgamma(((NUMINDEXEDDIM*0.5))+1.0));
		
		double volume_unit_nsphere=a/b; //n-sphere

		double ratio_ncube_nsphere=volume_unit_ncube/volume_unit_nsphere;

		
		
		unsigned long int point_threshold=ceil((1.0*k_neighbors+1.0)*ratio_ncube_nsphere);

		//overestimation factor to make it so that GPU gets high density regions
		//gamma=0: no overestimation; gamma=1: order of magnitude more than (rounded) point threshold
		//gamma=0 gives more compute to GPU
		double gamma=0;


		unsigned long int min_pts=point_threshold;
		unsigned long int max_pts=point_threshold*10;
		
		unsigned long int point_threshold_gamma=min_pts+(max_pts-min_pts)*gamma;
		


		printf("\nRatio ncube to nsphere: %f, Points required in a cell: %lu, With gamma: %f, points: %lu",ratio_ncube_nsphere,point_threshold,gamma,point_threshold_gamma);




	//use this to reassign GPU query points back to CPU when we have a minimum amount of work that we want to assign the CPU


	

	std::vector<GPUQueryNumPts>	tmpGPUQueriesAndCellPts; 
		


	//assign CPU and GPU points
	for (int i=0; i<*nNonEmptyCells; i++)
	{
		unsigned int grid_cell_idx=gridCellLookupArr[i].idx;		
		
		//if insufficient points in origin cell, assign to CPU
		if (((index[grid_cell_idx].indexmax-index[grid_cell_idx].indexmin)+1)<=point_threshold_gamma)
		{
			for (int j=index[i].indexmin; j<=index[i].indexmax;j++)
			{
				queriesCPU->push_back(indexLookupArr[j]);
			}
		}
		//if sufficient points, assign to GPU (temporarily, may get switched back later)
		else
		{
			for (int j=index[i].indexmin; j<=index[i].indexmax;j++)
			{
				//original
				// queriesGPU->push_back(indexLookupArr[j]);
				
				//temporarily assign GPU points
				GPUQueryNumPts tmp;
				tmp.queriesGPU=indexLookupArr[j];
				tmp.pntsInCell=(index[grid_cell_idx].indexmax-index[grid_cell_idx].indexmin)+1;
				tmpGPUQueriesAndCellPts.push_back(tmp);
			}
		}
	}

	unsigned int totalPnts=queriesCPU->size()+tmpGPUQueriesAndCellPts.size();

	printf("\nInitial split of work: CPU queries: %lu, GPU queries: %lu",queriesCPU->size(),tmpGPUQueriesAndCellPts.size());

	unsigned int minCPUQueries=0;
	printf("\nMinimum CPU queries: %u",minCPUQueries);

	//check to make sure there's a minimum amount of CPU work
	
	//if there's a minimum amount of CPU work, assign the GPU its points
	if ((double)queriesCPU->size()>(minCPUQueries))
	{
		printf("\nSufficient CPU Work");

		for (unsigned int i=0; i<tmpGPUQueriesAndCellPts.size(); i++)
		{
			 queriesGPU->push_back(tmpGPUQueriesAndCellPts[i].queriesGPU);
		}
	}	
	//if there's insufficient CPU work
	//sort the GPU query points from highest to lowest and assign to CPU or GPU as appropriate
	//CPU gets the points with fewer points in the cell.
	else
	{
		printf("\nInsufficient CPU Work. Assigning some of the GPU points to the CPU.");		
		std::sort(tmpGPUQueriesAndCellPts.begin(), tmpGPUQueriesAndCellPts.end(), compareByNumPointsInCell);

		//validation
		// printf("\n****\nPoint init assigned to GPU/Num points in cell:\n");
		// for (int i=0; i<tmpGPUQueriesAndCellPts.size(); i++)
		// {
		// 	printf("%u, %lu\n",tmpGPUQueriesAndCellPts[i].queriesGPU, tmpGPUQueriesAndCellPts[i].pntsInCell);
		// }

		unsigned int numPntsReassign=(double)queriesCPU->size();
		printf("\nReassigning %u GPU points to the CPU", numPntsReassign);

		//reassign to CPU
		// printf("\n****\nQuery point CPU:");
		for (unsigned int i=0; i<numPntsReassign; i++)
		{

			queriesCPU->push_back(tmpGPUQueriesAndCellPts[i].queriesGPU);
			//validation
			// printf("%u\n",queriesCPU->back());
		}
		// printf("\n****\nQuery point GPU:");
		//assign to GPU
		for (unsigned int i=numPntsReassign; i<tmpGPUQueriesAndCellPts.size(); i++)
		{
			queriesGPU->push_back(tmpGPUQueriesAndCellPts[i].queriesGPU);
			//validation
			// printf("%u\n",queriesGPU->back());
		}
	}

	
	#if TESTTHROUGHPUT>0
	
		//if assigning all queries to the GPU
		#if TESTTHROUGHPUT==1
		printf("\nTESTING THROUGHPUT, ASSIGNING ALL QUERIES TO GPU");
		for (int i=0; i<queriesCPU->size(); i++)
		{
			queriesGPU->push_back((*queriesCPU)[i]);
		}
		queriesCPU->clear();
		#endif

		//if assigning all queries to the GPU
		#if TESTTHROUGHPUT==2
		printf("\nTESTING THROUGHPUT, ASSIGNING ALL QUERIES TO CPU");
		for (int i=0; i<queriesGPU->size(); i++)
		{
			queriesCPU->push_back((*queriesGPU)[i]);
		}
		queriesGPU->clear();
		#endif
	#endif


	//used when smapling the dataset for testing the best parameter values
	#if GRIDSEARCHSAMPLE==1
	printf("\n[GRIDSEARCHSAMPLE==1] Sampling the data points for parameter search");

	
	//copy the queries into temp vectors
	std::vector<unsigned int> queriesCPUtmp;
	std::vector<unsigned int> queriesGPUtmp;

	std::copy(queriesCPU->begin(), queriesCPU->end(), std::back_inserter(queriesCPUtmp));
	std::copy(queriesGPU->begin(), queriesGPU->end(), std::back_inserter(queriesGPUtmp));
	
	//clear the query arrays
	queriesCPU->clear();
	queriesGPU->clear();

	//reassign 1%

	for (int i=0; i<queriesCPUtmp.size(); i+=GRIDSAMPLE)
	{
		queriesCPU->push_back(queriesCPUtmp[i]);
	}
	
	for (int i=0; i<queriesGPUtmp.size(); i+=GRIDSAMPLE)
	{
		queriesGPU->push_back(queriesGPUtmp[i]);
	}

	
	#endif	
	
	return;	

}




//Order the work from most to least work
//Based on the points within each cell
//Like the original splitwork function but we do not make a static splitting of points
void computeWorkDifficulty(unsigned int * outputOrderedQueryPntIDs, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, struct grid * index)
{

	std::vector<workArray> totalWork; 
		

	//loop over each non-empty cell and find the points contained within
	//record the number of points in the cell
	for (int i=0; i<*nNonEmptyCells; i++)
	{
		unsigned int grid_cell_idx=gridCellLookupArr[i].idx;		

		unsigned int numPtsInCell=(index[grid_cell_idx].indexmax-index[grid_cell_idx].indexmin)+1;

			for (int j=index[i].indexmin; j<=index[i].indexmax;j++)
			{
				workArray tmp;
				tmp.queryPntID=indexLookupArr[j];
				tmp.pntsInCell=numPtsInCell;
				totalWork.push_back(tmp);
			}



	}

	//sort the array containing the total work:
	std::sort(totalWork.begin(), totalWork.end(), compareWorkArrayByNumPointsInCell);


	// printf("\nTotal Work Arr: ");
	for (unsigned int i=0; i<totalWork.size(); i++)
	{
		outputOrderedQueryPntIDs[i]=totalWork[i].queryPntID;
	}

	// printf("\nTotal Work Arr: ");
	// for (unsigned int i=0; i<totalWork.size(); i++)
	// {
	// 	printf("\nPoint: %u, pntsInCell: %lu", totalWork[i].queryPntID, totalWork[i].pntsInCell);
	// }


	
	return;	

}




int criticalCheckEquality(unsigned int * totalQueriesCompleted, unsigned long int TotalQueries)
{
	int equal=0;
	#pragma omp critical
	{
		if (*totalQueriesCompleted==TotalQueries)
		{
			equal=1;
		}
	}
	return equal;
}


