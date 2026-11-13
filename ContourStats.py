import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csvinput', type=str, help="Directory or single CSV contour file")
parser.add_argument('output', type=str, help="Output ContourStats CSV file")
args = parser.parse_args()


def find_files(path, ext=".csv"):
    if os.path.isdir(path):
        return sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(ext)])
    if os.path.isfile(path):
        return [path]
    return []

class ContourProcessor:

    def __init__(self, files):
        self.files = files
        self.rows = []

    def load(self, f):
        df = pd.read_csv(f)
    
        df.columns = (
            df.columns
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.lower()
        )
    
        # optional but useful sanity check
        assert "time" in df.columns, f"Bad columns: {df.columns}"
    
        return df

    def process(self, df):
        t = df["time"].values
        f = df["f0"].values

        n = len(df)
        duration = t[-1] - t[0]

        freqmax = np.max(f)
        freqmin = np.min(f)
        freqbeg = f[0]
        freqend = f[-1]

        freqmean = np.mean(f)
        freqstd = np.std(f)
        freqmedian = np.median(f)

        freqrng = freqmax - freqmin
        freqcenter = (freqmax + freqmin) / 2
        freqrelbw = freqrng / freqcenter if freqcenter != 0 else 0

        freqmaxminratio = freqmax / freqmin if freqmin != 0 else 0
        freqbegendratio = freqend / freqbeg if freqbeg != 0 else 0

        q1 = np.quantile(f, 0.25)
        q2 = np.quantile(f, 0.50)
        q3 = np.quantile(f, 0.75)

        freqspread = q3 - q1

        # COFM (20 samples)
        idx = np.linspace(0, n - 1, 20).astype(int)
        f_s = f[idx]
        cofm = np.sum(np.abs(np.diff(f_s))) / 10000

        # slopes
        slopes = np.diff(f) / np.diff(t)
        slope_mean = np.mean(slopes)
        slope_abs_mean = np.mean(np.abs(slopes))
        pos = slopes[slopes > 0]
        neg = slopes[slopes < 0]

        slope_pos_mean = np.mean(pos) if len(pos) else 0
        slope_neg_mean = np.mean(neg) if len(neg) else 0
        slope_ratio = slope_pos_mean / slope_neg_mean if slope_neg_mean != 0 else 0

        # steps
        step_thresh = 0.11  # 11%
        step_up = 0
        step_down = 0
        for i in range(1, n):
            if f[i] > f[i - 1] * (1 + step_thresh):
                step_up += 1
            elif f[i] < f[i - 1] * (1 - step_thresh):
                step_down += 1

        num_steps = step_up + step_down
        step_dur = num_steps / duration if duration != 0 else 0

        # sweeps (simple discrete classification)
        sweep = np.sign(np.diff(f))
        sweep_up = np.sum(sweep > 0)
        sweep_down = np.sum(sweep < 0)
        sweep_flat = np.sum(sweep == 0)

        sweep_total = max(len(sweep), 1)

        sweep_up_pct = 100 * sweep_up / sweep_total
        sweep_down_pct = 100 * sweep_down / sweep_total
        sweep_flat_pct = 100 * sweep_flat / sweep_total

        # beginning/end sweep
        beg_sweep = np.sign(np.mean(slopes[:3])) if len(slopes) >= 3 else 0
        end_sweep = np.sign(np.mean(slopes[-3:])) if len(slopes) >= 3 else 0

        return {
            "FREQMAX": freqmax,
            "FREQMIN": freqmin,
            "DURATION": duration,
            "FREQBEG": freqbeg,
            "FREQEND": freqend,
            "FREQRANGE": freqrng,
            "DCMEAN": df.get("duty_cycle", pd.Series([0]*n)).mean(),
            "DCSTDDEV": df.get("duty_cycle", pd.Series([0]*n)).std(),
            "FREQMEAN": freqmean,
            "FREQSTDDEV": freqstd,
            "FREQMEDIAN": freqmedian,
            "FREQCENTER": freqcenter,
            "FREQRELBW": freqrelbw,
            "FREQMAXMINRATIO": freqmaxminratio,
            "FREQBEGENDRATIO": freqbegendratio,
            "FREQQUARTER1": q1,
            "FREQQUARTER2": q2,
            "FREQQUARTER3": q3,
            "FREQSPREAD": freqspread,
            "FREQCOFM": cofm,

            "FREQSTEPUP": step_up,
            "FREQSTEPDOWN": step_down,
            "FREQNUMSTEPS": num_steps,

            "FREQSLOPEMEAN": slope_mean,
            "FREQABSSLOPEMEAN": slope_abs_mean,
            "FREQPOSSLOPEMEAN": slope_pos_mean,
            "FREQNEGSLOPEMEAN": slope_neg_mean,
            "FREQSLOPERATIO": slope_ratio,

            "FREQBEGSWEEP": int(beg_sweep),
            "FREQENDSWEEP": int(end_sweep),

            "FREQSWEEPUPPERCENT": sweep_up_pct,
            "FREQSWEEPDWNPERCENT": sweep_down_pct,
            "FREQSWEEPFLATPERCENT": sweep_flat_pct,

            "STEPDUR": step_dur,
        }

    def run(self):
        for f in self.files:
            df = self.load(f)
            stats = self.process(df)
            stats["FILE"] = os.path.basename(f)
            self.rows.append(stats)

        out = pd.DataFrame(self.rows)
        out.to_csv(args.output, index=False)
        print("Wrote:", args.output)


files = find_files(args.csvinput, ".csv")
print(len(files), "files found")

proc = ContourProcessor(files)
proc.run()