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

        self.segments = None


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
        

    def load_crepe(self, ds):
        self.times = np.array(ds["time"])
        self.freqs = np.array(ds["f0"])
        self.confidence = np.array(ds["confidence"])

    def load_sam(self, ds):
        self.times = np.array(ds["time_s"])
        self.freqs = np.array(ds["freq_hz"])
