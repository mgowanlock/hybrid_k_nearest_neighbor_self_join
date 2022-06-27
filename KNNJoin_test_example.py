import csv
import numpy as np
import pandas as pd
#load the GDS-Join Python library
import  knnjoingpu as knnjoingpu


def getColumn(filename, column):
    results = csv.reader(open(filename), delimiter=",")
    # next(results, None)  # skip the headers
    return [result[column] for result in results]




if __name__ == "__main__":

    #parameters
    
    fname="gaia_dr2_ra_dec_25M.txt"
    df=pd.read_csv(fname, delimiter=',', header=None)

    #flatten data to a list which is required by the library
    dataset=df.values.flatten().tolist()
    KNN=5
    
    verbose=False #True/False --- this is the C output from the shared library
    dtype="float"
    numdim=2
    

    neighborTable, neighborTableDistances = knnjoingpu.knnjoin(dataset, KNN, numdim, dtype, verbose)    

    #print the KNN for the first and last object
    print(neighborTable[0:(KNN+1)])
    print(neighborTable[-(KNN+1):])

    #print the distances for the above
    print(neighborTableDistances[0:(KNN+1)])
    print(neighborTableDistances[-(KNN+1):])

        

    #this is only used so that we can run multiple examples at once and surpress the C stdout
    # if(verbose==False):
    #     knnjoingpu.redirect_stdout()

