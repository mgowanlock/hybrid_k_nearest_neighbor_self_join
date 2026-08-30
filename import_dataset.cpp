#include <stdio.h>
#include <vector>
#include <fstream>
#include <istream>
#include <iostream>
#include <string>
#include <string.h>
#include <sstream>
#include <cstdlib>
#include<bits/stdc++.h>
// #include "prototypes.h"
#include <algorithm>
// #include "globals.h"

#include "params.h"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

bool sortNDComp(const std::vector<DTYPE>& a, const std::vector<DTYPE>& b)
{
    for (int i=0; i<GPUNUMDIM; i++){
      if (int(a[i])<int(b[i])){
      return 1;
      }
      else if(int(a[i])>int(b[i])){
      return 0;
      }  
    }

    return 0;

    //in 2-D
    /*
    if (int(a[0])<int(b[0])){
      return 1;
    }
    else if(int(a[0])>int(b[0])){
      return 0;
    }
    //if equal compare on second coord
    else if (int(a[1])<int(b[1])){
      return 1;
    }

    return 0;
    */
    
}


//OLD with vectors
/*
void importNDDataset(std::vector<std::vector <DTYPE> > *dataPoints, char * fname)
{

	std::vector<DTYPE>tmpAllData;
	std::ifstream in(fname);
	int cnttmp=0;
	for (std::string f; getline(in, f, ',');){
	
	DTYPE i;
		 std::stringstream ss(f);
	    while (ss >> i)
	    {
	        tmpAllData.push_back(i);
	        //std::cout<<tmpAllData[cnttmp++]<<"\n";
	        if (ss.peek() == ',')
	            ss.ignore();
	    }
  		
  	}	


    





  	unsigned int cnt=0;
  	const unsigned int totalPoints=(unsigned int)tmpAllData.size()/GPUNUMDIM;
  	// printf("\nData import: Total size of all data (1-D) vect (number of points * GPUNUMDIM): %zu",tmpAllData.size());
  	// printf("\nData import: Total data points: %u",totalPoints);
  	
  	for (unsigned int i=0; i<totalPoints; i++){
  		std::vector<DTYPE>tmpPoint;
  		for (int j=0; j<GPUNUMDIM; j++){

  			tmpPoint.push_back(tmpAllData[cnt]);
  			cnt++;
  		}
  		dataPoints->push_back(tmpPoint);
  	}



    


	//Test output data 
  	// for (int i=0; i<totalPoints; i++){
  	// 	printf("\n");
  	// 	for (int j=0; j<NDIM; j++)
  	// 	printf("%f,",(*dataPoints)[i][j]);

  	// }


}
*/

//use uint64_t for offsets
int importDataset(char * fname, uint64_t N, uint64_t NDIM, DTYPE * dataset)
{

    FILE *fp = fopen(fname, "r");

    if (!fp) {
        printf("Unable to open file\n");
        exit(0);
    }

    char buf[4096];
    uint64_t rowCnt = 0;
    uint64_t colCnt = 0;
    while (fgets(buf, 4096, fp) && rowCnt<N) {
        colCnt = 0;

        char *field = strtok(buf, ",");
        DTYPE tmp;
        if(STR(DTYPE)=="float")
          sscanf(field,"%f",&tmp);
        if(STR(DTYPE)=="double")
          sscanf(field,"%lf",&tmp);
        
        
        dataset[rowCnt*NDIM+colCnt]=tmp;

        
        while (field) {
          colCnt++;
          field = strtok(NULL, ",");
          
          if (field!=NULL)
          {
          DTYPE tmp;
          if(STR(DTYPE)=="float")
            sscanf(field,"%f",&tmp);
          if(STR(DTYPE)=="double")
            sscanf(field,"%lf",&tmp);
          dataset[rowCnt*NDIM+colCnt]=tmp;
          }   

        }
        rowCnt++;
    }

    fclose(fp);

    return 0;


}


void sortInNDBins(std::vector<std::vector <DTYPE> > *dataPoints){
  
  std::sort(dataPoints->begin(),dataPoints->end(),sortNDComp);
  
}



