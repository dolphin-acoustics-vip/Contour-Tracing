import numpy as np
from Contour import Contour
import visualisation as v

import matplotlib.pyplot as plt


# Interpolate two segments onto a common time grid.
def interpolate_segments(segment_a, segment_b):

    times_a, freqs_a = map(np.asarray, segment_a)
    times_b, freqs_b = map(np.asarray, segment_b)

    # Find temporal overlap
    start = max(times_a[0], times_b[0])
    end = min(times_a[-1], times_b[-1])

    if end <= start:
        return None

    # Construct common time grid
    common_times = np.unique(np.concatenate((
        [start, end],
        times_a[(times_a > start) & (times_a < end)],
        times_b[(times_b > start) & (times_b < end)]
    )))

    # Interpolate both contours
    freqs_a_interp = np.interp(common_times, times_a, freqs_a)
    freqs_b_interp = np.interp(common_times, times_b, freqs_b)

    return common_times, freqs_a_interp, freqs_b_interp


# Calculate the time-weighted mean absolute frequency difference between two segmented contours.
def mean_absolute_distance(segments_a, segments_b):
    total_distance = 0.0
    total_duration = 0.0

    for segment_a in segments_a:
        for segment_b in segments_b:

            result = interpolate_segments(segment_a, segment_b)

            if result is None:
                continue

            times, freqs_a, freqs_b = result

            differences = np.abs(freqs_a - freqs_b)

            total_distance += np.trapezoid(differences, times)
            total_duration += times[-1] - times[0]

    if total_duration == 0:
        return np.nan

    return total_distance / total_duration

# https://onlinelibrary.wiley.com/doi/full/10.1111/mms.70263
# coverage defined as the amount of the manual contour that the automated successfully traces,
# within a certain tolerence, paper referenced uses +/-25% 
def coverage(truth_segments, predicted_segments, tolerance=0.25):
    total_points = 0
    covered_points = 0

    for truth in truth_segments:
        total_points += len(truth[0])

        # Track points covered by any predicted segment
        covered = np.zeros(len(truth[0]), dtype=bool)

        for pred in predicted_segments:
            covered |= find_correct_matches(
                truth, pred, tolerance
            )

        covered_points += np.count_nonzero(covered)

    if total_points == 0:
        return np.nan

    return covered_points / total_points

def find_correct_matches(truth, pred, tolerance=0.25):
    truth_times, truth_freqs = map(np.asarray, truth)
    pred_times, pred_freqs = map(np.asarray, pred)

    matches = np.zeros(len(truth_times), dtype=bool)

    if len(truth_times) == 0 or len(pred_times) == 0:
        return matches

    # Find truth points within the predicted segment's time range
    mask = (
        (truth_times >= pred_times[0]) &
        (truth_times <= pred_times[-1])
    )

    # Interpolate predicted frequencies onto truth time points
    interpolated = np.interp(
        truth_times[mask],
        pred_times,
        pred_freqs
    )

    # Check relative frequency difference
    matches[mask] = (
        np.abs(interpolated - truth_freqs[mask])
        <= tolerance * np.abs(truth_freqs[mask])
    )

    return matches
    









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
    c1 = Contour("comparison/test-data/cb_sel_65_T08_1_T08_1_Sarasota_analyst_alex.csv", "alex")
    # c2 = Contour("comparison/test-data/cb_sel_65_T08_1_T08_1_Sarasota_analyst_jackie.csv", "jackie")
    # c3 = Contour("comparison/test-data/cb_sel_65_T08_1_T08_1_Sarasota_sam.csv", "sam")
    c4 = Contour("comparison/test-data/cb_sel_65_T08_1_T08_1_Sarasota_crepe.csv", "crepe")
    
    print(coverage(c1.segments, c4.segments, 0.2))

    v.plt_contours([c1, c4], s=1)
    plt.show()
