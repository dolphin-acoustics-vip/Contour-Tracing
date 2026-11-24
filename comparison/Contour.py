import pandas as pd
from pathlib import Path
import numpy as np


class Contour:
    def __init__(self, path, model):
        if path == None:
            raise TypeError
        self.path = Path(path)

        self.model = model
        self.freqs = None
        self.times = None
        self.confidence = None
        self.load()

        #list[list[(time, freq)]]
        self.segments = None
        self.generate_segments()

    # 0.01s
    def generate_segments(self, min_gap=0.025):
        self.segments = [[[], []]]

        for i in range(len(self.times)):
            if i > 0 and abs(self.times[i] - self.times[i - 1]) > min_gap:
                self.segments.append([[], []])

            self.segments[-1][0].append(self.times[i])
            self.segments[-1][1].append(self.freqs[i])




    def load(self):
        ds = pd.read_csv(self.path, sep=',', header=0)
        ds.columns = ds.columns.str.strip()
        
        if self.model == "sam":
            self.load_sam(ds)
        elif self.model == "crepe":
            self.load_crepe(ds)
        else:
            self.load_manual(ds)

    def load_manual(self, ds):
        self.times = np.array(ds["time_s"])
        self.freqs = np.array(ds["freq_hz"])
        

    def load_crepe(self, ds, confidence_threshold = 0.60):
        self.times = np.asarray(ds["time"])
        self.freqs = np.asarray(ds["f0"])
        self.confidence = np.asarray(ds["confidence"])

        # Keep predictions with sufficient confidence
        mask = self.confidence >= confidence_threshold

        self.times = self.times[mask]
        self.freqs = self.freqs[mask]
        self.confidence = self.confidence[mask]

    def load_sam(self, ds):
        self.times = np.array(ds["time_s"])
        self.freqs = np.array(ds["freq_hz"])
