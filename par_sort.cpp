//for parallel sorting on the host
#include <parallel/algorithm>

#include "par_sort.h"

//need to put this in here because the parallel mode extensions conflict with nvcc
void sortLinearIds(std::vector<uint64_t> *allGridCellLinearIds)
{
	__gnu_parallel::sort(allGridCellLinearIds->begin(), allGridCellLinearIds->end());
}