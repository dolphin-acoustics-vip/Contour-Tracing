import numpy as np
from Contour import Contour




def absolute_distances(base: Contour, new: Contour): 
    # for every point in automated one
    #   get distance from manual interpolation 
    interpolated_y = np.interp(new.time, base.time, base.freq)
    distances = np.abs(new.freq - interpolated_y)
    return distances



def mean_absolute_error(truth: Contour, approx: Contour):
    abs_dist = absolute_distances(truth, approx)
    return sum(abs_dist)/len(abs_dist)



# https://onlinelibrary.wiley.com/doi/full/10.1111/mms.70263
# coverage defined as the amount of the manual contour that the automated successfully traces,
# within a certain tolerence, paper referenced uses +/-25% 
def coverage(truth: Contour, approx: Contour, tolerance=0.25):
    interpolated_y = np.interp(truth.time, approx.time, approx.freq)
    return (np.abs(truth.freq - interpolated_y) <= tolerance * truth.freq)


def fragmentation():
    pass
        
        
    
    


def mean():
    pass


def median():
    pass

def mode():
    pass

def data_range():
    pass


if __name__ == "__main__":
    c1 = Contour("comparison/test-data/cb_sel_01_T08_1_T08_1_Sarasota_analyst_alex.csv", "alex")
    c2 = Contour("comparison/test-data/cb_sel_01_T08_1_T08_1_Sarasota_analyst_andrew.csv", "andrew")
    c3 = Contour("comparison/test-data/cb_sel_01_T08_1_T08_1_Sarasota_sam.csv", "sam")
    # c4 = Contour("comparison/test-data/cb_sel_01_T08_1_T08_1_Sarasota_crepe.csv", "crepe")
    plt_contours([c1, c2, c3], s=1)
    plt.show()