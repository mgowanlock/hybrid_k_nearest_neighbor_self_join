//////////////////////////////////////
//Parameters that may change:

//data type of the data
#define DTYPE float

//Set both of these to the number of dimensions
#define GPUNUMDIM 100
#define NUMINDEXEDDIM 6

//This stores the points within the distance of each query for multiple streams
//If your GPU has a memory allocation error, it's likely that this value needs to be decreased to allow
//for smaller batches to be processed
#define GPUBUFFERSIZE 75000000 


///////////////////////
//Utility
//used for outputting the neighbors at the end
#define PRINTNEIGHBORTABLE 0
//used for printing outlier scores based on point density at the end of program execution
#define PRINTOUTLIERSCORES 0 //make sure to disable unicomp (STAMP==0) if printing the outlier scores (it was only implemented for this case)
//end utility
///////////////////////

//////////////////////////////////////

//It is unlikely you will need to change anything below:

#define GPUSTREAMS 3 //number of concurrent gpu streams (GPUBUFFERSIZE) is allocated for each stream

#define SHORTCIRCUIT 1
 
 
#define REORDER 1 //This reorders the data by dimensionality


#define THREADMULTI -2 //0- just use 1 thread for distance calculations per query point
						//>=2 use (THREADMULTI  * iterations) threads for distance calculations
						//use THREADMULTI threads for distance calculations in the batch estimator
						//-1 -dynamically select the number of threads per point (use DYNAMICTHRESHOLD, MAXTHREADSPERPOINT)
						//-2 -statically use a certain number of threads per point regardless of iteration (use STATICTHREADSPERPOINT)

#define DYNAMICTHRESHOLD 1000000 //For the minimum number of threads per kernel invocation
								// Used when THREADMULTI Dynamic is used (THREADMULTI==-1)
								//Also used in the brute force kernel

#define MAXTHREADSPERPOINT 1024 //limit the maximum number of threads per point for dynamic threads/point (THREADMULTI==-1)
								//prevents thousands of threads launched for a few points

#define STATICTHREADSPERPOINT 8 //statically set the number of threads used for distance calculations per point (THREADMULTI==-2)



//BETA- overestimation of the search distance
//In the JPDC paper, BETA=0 because the CPU will find the points in spare regions
//Because this implementation is GPU-only, we increase BETA slightly so that we can find the neighbors in fewer iterations
#define BETA 0.05   //Parameter Between 0-1
					//In the sampleNeighborsBruteForce function
					//BETA=0 means that we only use the epsilon derived by the batch estimator 
					//that finds an average of k neighbors within epsilon
					//The epsilon is multipled by 2 to create a cell that contains the n-sphere
					//BETA=1 means that we use the epsilon value dervied by finding 100x k estimated neighbors
					//BETA=0 means that it will be harder to find cells with k neighbors (accounting for the 
					//volume ratio of the cell to the n-sphere)
					//BETA=1 means that it will be easier to find at least k neighbors, but
					//there will be more GPU computation


			

#define BLOCKSIZE 256
#define SEARCHFILTTIME 0

//turn this off in time trials so that we don't slow down the execution
//used to see how many point comparisons and grid cell searches
#define COUNTMETRICS 0
#define CTYPE unsigned long long

