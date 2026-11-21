
from Contour import Contour
import matplotlib.pyplot as plt
from itertools import cycle

def plt_contours(contours: list[Contour], ax=None, **kwargs):
    colors = cycle([
        "blue", "orange", "green", "red", "purple",
        "brown", "pink", "gray", "olive", "cyan"
    ])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    legend_labels = []

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Peak Frequency (Hz)")

    for ctr in contours:
        color = next(colors)
        plt_contour(ctr, ax, color=color, **kwargs)

    return fig, ax

def plt_contour(ctr: Contour, ax=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Plot the contour

    ax.scatter((ctr.times - ctr.times[0]), ctr.freqs, **kwargs)
    plt.xlabel('Time (samples)')
    plt.ylabel('Frequency (Hz)')

    return fig, ax



if __name__ == "__main__":
    c1 = Contour("comparison/test-data/cb_sel_01_T08_1_T08_1_Sarasota_analyst_alex.csv", "alex")
    c2 = Contour("comparison/test-data/cb_sel_01_T08_1_T08_1_Sarasota_analyst_andrew.csv", "andrew")
    c3 = Contour("comparison/test-data/cb_sel_01_T08_1_T08_1_Sarasota_sam.csv", "sam")
    # c4 = Contour("comparison/test-data/cb_sel_01_T08_1_T08_1_Sarasota_crepe.csv", "crepe")
    plt_contours([c1, c2, c3], s=1)
    plt.show()