import os
import numpy.ctypeslib as npct
from ctypes import *
import ctypes
import csv
import numpy as np
from contextlib import contextmanager
import sys

def getColumn(filename, column):
    results = csv.reader(open(filename), delimiter=",")
    # next(results, None)  # skip the headers
    return [result[column] for result in results]


# This function converts an input numpy array into a different
# data type and ensure that it is contigious.  
def convert_type(in_array, new_dtype):

    ret_array = in_array
    
    if not isinstance(in_array, np.ndarray):
        ret_array = np.array(in_array, dtype=new_dtype)
    
    elif in_array.dtype != new_dtype:
        ret_array = np.array(ret_array, dtype=new_dtype)

    if ret_array.flags['C_CONTIGUOUS'] == False:
        ret_array = np.ascontiguousarray(ret_array)

    return ret_array


#from stackoverflow 
# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
def redirect_stdout():
    print ("Verbose mode is false. Redirecting C shared library stdout to /dev/null")
    sys.stdout.flush() # <--- important when redirecting to files
    newstdout = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    sys.stdout = os.fdopen(newstdout, 'w')

#from stackoverflow 
# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    print ("Verbose mode is false. Redirecting C shared library stdout to /dev/null")
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

#from here: https://www.py4u.net/discuss/15884
class SuppressStream(object): 

    def __init__(self, stream=sys.stderr):
        self.orig_stream_fileno = stream.fileno()

    def __enter__(self):
        self.orig_stream_dup = os.dup(self.orig_stream_fileno)
        self.devnull = open(os.devnull, 'w')
        os.dup2(self.devnull.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        os.close(self.orig_stream_fileno)
        os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
        os.close(self.orig_stream_dup)
        self.devnull.close()

#https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
class HideOutput(object):
    '''
    A context manager that block stdout for its scope, usage:

    with HideOutput():
        os.system('ls -l')
    '''

    def __init__(self, *args, **kw):
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        self._newstdout = os.dup(1)
        os.dup2(self._devnull, 1)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, 1)





#https://www.codeforests.com/2020/11/05/python-suppress-stdout-and-stderr/    
@contextmanager
def nullify_output(suppress_stdout=True, suppress_stderr=True):
    stdout = sys.stdout
    stderr = sys.stderr
    devnull = open(os.devnull, "w")
    try:
        if suppress_stdout:
            sys.stdout = devnull
        if suppress_stderr:
            sys.stderr = devnull
        yield
    finally:
        if suppress_stdout:
            sys.stdout = stdout
        if suppress_stderr:
            sys.stderr = stderr




#wrapper to enable the verbose option
def knnjoin(DATASET, KNN, NDIM, DTYPE, verbose=False):
    if(verbose==False):
        with HideOutput():
            ret_outNumNeighborsWithinEps_wrapper, ret_neighborTable_wrapper = knnjoinmain(DATASET, KNN, NDIM, DTYPE)            
    else:
        ret_outNumNeighborsWithinEps_wrapper, ret_neighborTable_wrapper = knnjoinmain(DATASET, KNN, NDIM, DTYPE)

    return ret_outNumNeighborsWithinEps_wrapper, ret_neighborTable_wrapper

#main function
def knnjoinmain(DATASET, KNN, NDIM, DTYPE):

    
    
    ###############################
    #Check for valid parameters
    ###############################
    


    # Create variables that define C interface
    array_1d_double = npct.ndpointer(dtype=c_double, ndim=1, flags='CONTIGUOUS')
    array_1d_float = npct.ndpointer(dtype=c_float, ndim=1, flags='CONTIGUOUS')
    array_1d_unsigned = npct.ndpointer(dtype=c_uint, ndim=1, flags='CONTIGUOUS')
    array_1d_int = npct.ndpointer(dtype=c_int, ndim=1, flags='CONTIGUOUS')

    #load the shared library 
    lib_path = os.getcwd()
    
    libknnjoin = npct.load_library('libgpuknnselfjoin.so', lib_path)
    
    

    #total number of rows in file
    numPoints=int(len(DATASET)/NDIM)
    print("[Python] Number of rows in file: %d" %(numPoints))

    #convert the dataset to a numpy array
    DATASET=np.asfarray(DATASET)

    

    #convert to CTYPES
    if (DTYPE=="float"):
        c_DATASET=convert_type(DATASET, c_float)
    elif (DTYPE=="double"):     
        c_DATASET=convert_type(DATASET, c_double)
        

    # Allocate arrays for results 
    #KNN+1 point ids per point in the dataset
    ret_outNeighborTable = np.zeros(numPoints*(KNN+1), dtype=c_int)

    #distances to the points above
    if (DTYPE=="float"):
        ret_outNeighborTableDistances = np.zeros(numPoints*(KNN+1), dtype=c_float)
    elif (DTYPE=="double"):
        ret_outNeighborTableDistances = np.zeros(numPoints*(KNN+1), dtype=c_double)

    #float
    if (DTYPE=="float"):
        #define the argument types
        libknnjoin.KNNJoinPy.argtypes = [array_1d_float, c_uint, c_uint, c_uint, array_1d_int, array_1d_float]
        #call the library
        libknnjoin.KNNJoinPy(c_DATASET, c_uint(numPoints), c_uint(KNN), c_uint(NDIM), ret_outNeighborTable, ret_outNeighborTableDistances)

    #double
    if (DTYPE=="double"):    
        #define the argument types
        libknnjoin.KNNJoinPy.argtypes = [array_1d_double, c_uint, c_uint, c_uint, array_1d_int, array_1d_double]
        #call the library
        libknnjoin.KNNJoinPy(c_DATASET, c_uint(numPoints), c_uint(KNN), c_uint(NDIM), ret_outNeighborTable, ret_outNeighborTableDistances)

    
    sumOfDistances=np.sum(ret_outNeighborTableDistances)
    print("[Python] Sum of distances to all KNN (each distance has been squared using Euclidean Distance): %f" %(sumOfDistances))

    return ret_outNeighborTable, ret_outNeighborTableDistances





