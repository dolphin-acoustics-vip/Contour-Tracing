# Apologies, I've used a mix of naming conventions here which i realise isn't 
# convenient (e.g. csv_dict, resampledtime). Hopefully it still all makes sense. 
# The whole script is essentially a fancy interactive plot. I'm VERY open to restructures!
# Additions (Tamra): contrast buttons, dot colour button, spectogram background colour dropdown button, reset button

from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import numpy as np
import argparse, os
import soundfile as sf
from scipy import signal
import pandas as pd
from pathlib import Path
import warnings
# Suppress filter warnings
warnings.filterwarnings("ignore")

# Parse arguments if given
parser = argparse.ArgumentParser()
parser.add_argument('wavinput', type=str, help="Directory with sound files to process, or a single file to process")
parser.add_argument('csvinput', type=str, help="Directory with csv files to process, or a single file to process. Ensure filenames match .wav files exactly (with _f0 suffix)")
parser.add_argument('--threshold', type=float, help="Confidence threshold used when printing F0 predictions on spectrograms ", default=0.1)
parser.add_argument('--res', type=float, help="Time resolution (inter-point interval) in seconds", default=0.002)
parser.add_argument('--NFFT', type=int, help="FFTsize", default=512)
parser.add_argument('--start', type=int, help="Numeric index of first file you wish to analyse in folder", default=512)
#parser.add_argument('--cursorsize', type=float, help="Size of cursor (minimum = 0, larger cursors produce smoother contours, smaller are more precise)", default=1)

args = parser.parse_args()

# Initialisation, basic function to retrieve wavs and csvs
def find_files(path, ext):
    if os.path.isdir(path):
        return [os.path.join(path,f) for f in os.listdir(path) if f.lower().endswith(ext)]
    if os.path.isfile(path):
        return [path]
    return []

# Checking files to see whether we have the same no. of each 
wfiles = find_files(args.wavinput, ".wav")
dfiles = find_files(args.csvinput, ".csv")
print(len(wfiles), "WAV files found")
print(len(dfiles), "CSV files found")
audio_dict = {Path(f).stem: f for f in wfiles}

# matching .wavs to their respective .csvs
csv_dict = {}
for f in dfiles:
    stem = Path(f).stem
    if stem.endswith("_f0"):
        stem = stem[:-3]
    csv_dict[stem] = f

common_keys = sorted(audio_dict.keys() & csv_dict.keys())
print(len(common_keys), "matched pairs")

# looking for stats file, creating one if it doesn't already exist
contourstats = "ContourStats.csv"
if not os.path.exists(contourstats):
    pd.DataFrame(columns = ["file", "peak_frequency_Hz", "peak_time_s", "peak_time_%"]).to_csv(contourstats, index = False)
    
class Annotator:
#initialising gui
    def __init__(self, keys):
        self.keys = keys
        self.i = args.start
        self.saved_contours = []
        self.preview_contour = None
        self.selected_ghost = None

        # Background spectogram colours
        self.colormaps = [
            "viridis",
            "magma",
            "plasma",
            "inferno",
            "gray",
            "cividis",
            "cool",
            "hot"
        ]

        # Set default spectogram colour
        self.cmap = "viridis"

        # Set default contrast to 1
        self.contrast = 1.0

        # Close the spectogram colour toggle when first opened
        self.colour_menu_open = False

        # Set array of dot colours
        self.dot_colours = ["red", "yellow", "lime", "cyan", "magenta", "white", "black"]
        
        # Set default dot colour to red
        self.dot_colour = "red"

        self.load_pair()
        
# loading csv and wav into window, calculating spectrogram, further initialisation/setup for 'ghost' contours etc.
    def load_pair(self):
        if self.i >= len(self.keys):
            self.i = 0
        elif self.i < 0:
            self.i = len(self.keys) - 1
    
        key = self.keys[self.i]
        print("Loading", key)
        wavfile = audio_dict[key]
        csvfile = csv_dict[key]
        self.sig, self.fs = sf.read(wavfile)
        df = pd.read_csv(csvfile)
        df = df[df.confidence > args.threshold]

        self.f, self.t, self.Sxx = signal.spectrogram(self.sig, fs=self.fs, nperseg=int(args.NFFT), 
                                                      noverlap=int(args.NFFT//2), nfft=int(args.NFFT)) # spectrogram code
        # Change here
        self.Sxx = 20 * np.log10(
            np.maximum(self.Sxx, 1e-12)
        )

        # Store default contrast limits
        self.sxx_vmax = np.percentile(self.Sxx, 99)
        self.sxx_vmin_default = np.percentile(self.Sxx, 5)

        self.mat = np.zeros(self.Sxx.shape, bool)

        for _,r in df.iterrows():
            x = np.argmin(abs(self.t - r.time))
            y = np.argmin(abs(self.f - r.f0))
            self.mat[:,x] = False
            self.mat[y,x] = True
            
        self.saved_contours = []
        base = self.keys[self.i] + "_corrected"
        for f in os.listdir():
            if f.startswith(base + "_") and f.endswith(".csv") and "_f0" not in f:
                dfc = pd.read_csv(f)
                step = max(1, len(dfc["time"].values) // 50)
                self.saved_contours.append({
                    "t": dfc["time"].values,
                    "f": dfc["f0"].values,
                    "t_sel": dfc["time"].values[::step],
                    "f_sel": dfc["f0"].values[::step],
                    "file": f
                }) # saving 'ghosts' -- already traced contours

        self.preview_contour = None
        self.selected_ghost = None
        self.plot()
        
    def get_contrast_limits(self):

        default_range = (
            self.sxx_vmax -
            self.sxx_vmin_default
        )

        visible_range = (
            default_range /
            self.contrast
        )

        vmin = self.sxx_vmax - visible_range
        vmax = self.sxx_vmax

        return vmin, vmax
    
    def plot(self):
        ax.clear()
        
        # Get contrast
        vmin, vmax = self.get_contrast_limits()
        # Show 
        ax.imshow(
            self.Sxx,
            origin="lower",
            aspect="auto",
            extent=[0, self.t[-1], 0, self.f[-1]],
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax
        ) # plotting spectrogram
        ax.set_ylim(0,self.fs/2)
        
        for i, g in enumerate(self.saved_contours): # plotting ghosts, highlihgting in red if selected
            if i == self.selected_ghost:
                ax.plot(g["t"], g["f"], color="red", linewidth=3, antialiased=False)
            else:
                ax.plot(g["t"], g["f"], color="black", linewidth=1, alpha=0.4, antialiased=False)
        
        if self.preview_contour is not None: # preview contour (from 'plot' button)
            tp, fp = self.preview_contour
            ax.plot(tp, fp, color="red", linewidth=3, antialiased=False)
        
        y,x = np.where(self.mat)
        ax.scatter(self.t[x], self.f[y], 10, facecolors = "none", edgecolors = self.dot_colour) # plotting predicted points
        ax.set_title(self.keys[self.i] + " | File: " + str(self.i))
        fig.canvas.draw_idle()

    def onclick(self, event): # mouse behaviour, can probably be improved, i struggled with this...
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        if event.dblclick:
            click_t = event.xdata
            click_f = event.ydata
            closest_idx = None
            closest_dist = np.inf
            t_range = self.t[-1] - self.t[0]
            f_range = self.f[-1] - self.f[0]
        
            for i, g in enumerate(self.saved_contours):
                if click_t < g["t"][0] or click_t > g["t"][-1]:
                    continue
                dt = (g["t_sel"] - click_t) / t_range
                df = (g["f_sel"] - click_f) / f_range
                d = np.min(dt**2 + df**2)
        
                if d < closest_dist:
                    closest_dist = d
                    closest_idx = i
        
            if closest_dist < 0.005:
                self.selected_ghost = closest_idx
                print("Selected:", self.saved_contours[closest_idx]["file"])
            else:
                self.selected_ghost = None
                print("No contour selected")
        
            self.plot()
            return

        x = np.argmin(abs(self.t - event.xdata))
        y = np.argmin(abs(self.f - event.ydata))

        if event.button == 3:
            self.mat[:, x] = False
        elif event.button == 1:
            self.mat[:, x] = False
            self.mat[y, x] = True

        self.plot()

    def clear(self,event=None): #clear the point matrix
        self.mat[:] = False
        self.plot()

    def up(self, event=None): # multiply points by 2 to go up a harmonic
        y, x = np.where(self.mat)
        if len(y) == 0:
            return
        freqs = self.f[y]
        new_freqs = freqs * 2
        new_freqs = np.clip(new_freqs, self.f[0], self.f[-1])
        new_y = np.array([np.argmin(abs(self.f - f)) for f in new_freqs])
        self.mat[y, x] = False
        self.mat[new_y, x] = True
        self.plot()

    def down(self, event=None): # ...and divide by 2 to go down, rounding errors can gradually smooth out line if overused
        y, x = np.where(self.mat)
        if len(y) == 0:
            return
        freqs = self.f[y]
        new_freqs = freqs / 2
        new_y = np.array([np.argmin(abs(self.f - f)) for f in new_freqs])
        self.mat[y, x] = False
        self.mat[new_y, x] = True
        self.plot()

    def previous(self,event=None): # navigation button, go to previous clip
        self.i -= 1
        self.load_pair()
        
    def undo(self, event=None): # clear the preview from 'plot' button
        if self.preview_contour is not None:
            self.preview_contour = None
            print("Preview cleared")
            self.plot()
        else:
            print("Nothing to undo")
        
    def preview(self, event=None): # PLOT button, interpolates line between points/predictions
        y, x = np.where(self.mat)
        time = self.t[x]
        f0 = self.f[y]
    
        if len(time) < 2:
            print("Not enough points to interpolate.")
            return
    
        order = np.argsort(time)
        time = time[order]
        f0 = f0[order]
        interval = args.res
        resampledtime = np.arange(time[0], time[-1], interval)
        resampledf0 = np.interp(resampledtime, time, f0)
        self.preview_contour = (resampledtime, resampledf0)
    
        print("Preview generated:", len(resampledtime), "points")
    
        self.plot()
    
    def commit(self, event=None): # saves contour to directory and some basic amplitude-based measured to contourstats, if we integrate other measurements they would go in here.
        if self.preview_contour is None:
            print("No preview found, generating now...")
            self.preview()
    
        if self.preview_contour is None:
            return
    
        resampledtime, resampledf0 = self.preview_contour
        confidence = np.ones(len(resampledtime))
        energy = []
        
        for t_val, f_val in zip(resampledtime, resampledf0):
            timeindex = np.argmin(abs(self.t - t_val))
            freqindex = np.argmin(abs(self.f - f_val))
            energy.append(self.Sxx[freqindex, timeindex])
    
        energy    = np.array(energy)
        peakindex = np.argmax(energy)
        peakfreq  = resampledf0[peakindex]
        peaktime  = resampledtime[peakindex]
        peaktimep = (resampledtime[peakindex]/max(resampledtime))*100
        base = self.keys[self.i] + "_corrected"
        n = 1
        while True:
            outname = f"{base}_{n:02d}.csv"
            if not os.path.exists(outname):
                break
            n += 1
    
        df = pd.DataFrame({
            "time": resampledtime,
            "f0": resampledf0,
            "confidence": confidence
        })
        df.to_csv(outname, index=False)
        print("Saved", outname)
    
        stats = pd.read_csv(contourstats)
        stats = pd.concat([
            stats,
            pd.DataFrame([{
                "file": outname,
                "wav": self.keys[self.i],
                "peak_frequency_Hz": peakfreq,
                "peak_time_s": peaktime,
                "peak_time_%": peaktimep 
            }])
        ])
        stats.to_csv(contourstats, index=False)
        print("Updated ContourStats")
        
        step = max(1, len(resampledtime) // 50)
        self.saved_contours.append({
            "t": resampledtime,
            "f": resampledf0,
            "t_sel": resampledtime[::step],
            "f_sel": resampledf0[::step],
            "file": outname
        })
        self.preview_contour = None
        self.mat[:] = False
        self.plot()

    def next(self,event=None): # navigation again, this time forwards
        self.i += 1
        self.load_pair()

    def edit_ghost(self, event=None): # edit previously commited contour (basically overwrite file and replace row in contourstats)
        if self.selected_ghost is None:
            print("No contour selected")
            return

        g = self.saved_contours[self.selected_ghost]
        self.mat[:] = False

        for t_val, f_val in zip(g["t"], g["f"]):
            x = np.argmin(abs(self.t - t_val))
            y = np.argmin(abs(self.f - f_val))
            self.mat[y, x] = True

        os.remove(g["file"])
        stats = pd.read_csv(contourstats)
        stats = stats[stats.file != g["file"]]
        stats.to_csv(contourstats, index=False)
        self.saved_contours.pop(self.selected_ghost)
        self.selected_ghost = None
        print("Contour loaded for editing")
        self.plot()

    def delete_ghost(self, event=None): # very similar, just removing file and row instead of replaacing
        if self.selected_ghost is None:
            print("No contour selected")
            return

        g = self.saved_contours[self.selected_ghost]

        if os.path.exists(g["file"]):
            os.remove(g["file"])
            print("Deleted", g["file"])

        stats = pd.read_csv(contourstats)
        stats = stats[stats.file != g["file"]]
        stats.to_csv(contourstats, index=False)
        self.saved_contours.pop(self.selected_ghost)
        self.selected_ghost = None
        self.plot()

    # COLOUR

    # Change spectogram background colour
    def change_colour(self, colour):
        self.cmap = colour
        self.plot()

    # Cycle through dot colour
    def cycle_dot_colour(self, event=None):
        current = self.dot_colours.index(self.dot_colour)
        self.dot_colour = self.dot_colours[(current + 1) % len(self.dot_colours)]
        self.plot()

    # CONTRAST

    def contrast_up(self, event=None):
        self.contrast *= 1.25
        self.plot()

    def contrast_down(self, event=None):
        self.contrast /= 1.25
        self.contrast = max(0.25, self.contrast)
        self.plot()

    # Reset to defaults
    def reset_appearance(self, event=None):
        self.cmap = "viridis"
        self.contrast = 1.0
        self.dot_colour = "red"
        self.plot()

###############################################
    def save_next(self,event=None): # combining some of the above functions into one button, not in use. 
        if self.preview_contour is None:
            print("No preview found, generating now...")
            self.preview()
        
        if self.preview_contour is None:
            return
        
        resampledtime, resampledf0 = self.preview_contour
        confidence = np.ones(len(resampledtime))
        energy = []
        
        for t_val, f_val in zip(resampledtime, resampledf0):
            timeindex = np.argmin(abs(self.t - t_val))
            freqindex = np.argmin(abs(self.f - f_val))
            energy.append(self.Sxx[freqindex, timeindex])
        
        energy    = np.array(energy)
        peakindex = np.argmax(energy)
        peakfreq  = resampledf0[peakindex]
        peaktime  = resampledtime[peakindex]
        peaktimep = (resampledtime[peakindex]/max(resampledtime))*100
        base = self.keys[self.i] + "_corrected"
        n = 1
        while True:
            outname = f"{base}_{n:02d}.csv"
            if not os.path.exists(outname):
                break
            n += 1
        
        df = pd.DataFrame({
            "time": resampledtime,
            "f0": resampledf0,
            "confidence": confidence
        })
        df.to_csv(outname, index=False)
        print("Saved", outname)
        
        stats = pd.read_csv(contourstats)
        stats = pd.concat([
            stats,
            pd.DataFrame([{
                "file": outname,
                "wav": self.keys[self.i],
                "peak_frequency_Hz": peakfreq,
                "peak_time_s": peaktime,
                "peak_time_%": peaktimep 
            }])
        ])
        stats.to_csv(contourstats, index=False)
        print("Updated ContourStats")
        
        step = max(1, len(resampledtime) // 50)
        self.saved_contours.append({
            "t": resampledtime,
            "f": resampledf0,
            "t_sel": resampledtime[::step],
            "f_sel": resampledf0[::step],
            "file": outname
        })
        self.preview_contour = None
        self.mat[:] = False
        
        print("Proceeding to next file...")
        self.i += 1
        self.load_pair()
##############################################################################

# button layout, mapping functions to them, and text     
fig, ax = plt.subplots(figsize=(10,6))
fig.subplots_adjust(
    bottom=0.15,
    right=0.84
)

ann = Annotator(common_keys)

ax_clear = fig.add_axes([0.01, 0.02, 0.075, 0.05])
ax_up    = fig.add_axes([0.10, 0.02, 0.075, 0.05])
ax_down  = fig.add_axes([0.19, 0.02, 0.075, 0.05])
ax_prev  = fig.add_axes([0.28, 0.02, 0.075, 0.05])
ax_undo  = fig.add_axes([0.37, 0.02, 0.075, 0.05])
ax_edit  = fig.add_axes([0.46, 0.02, 0.075, 0.05])
ax_del   = fig.add_axes([0.55, 0.02, 0.075, 0.05])
ax_plot  = fig.add_axes([0.64, 0.02, 0.075, 0.05])
ax_comm  = fig.add_axes([0.73, 0.02, 0.075, 0.05])
ax_next  = fig.add_axes([0.82, 0.02, 0.075, 0.05])
ax_save  = fig.add_axes([0.91, 0.02, 0.075, 0.05])

btn_clear = Button(ax_clear, "Clear")
btn_clear.on_clicked(ann.clear)
btn_up = Button(ax_up, "Up")
btn_up.on_clicked(ann.up)
btn_down = Button(ax_down, "Down")
btn_down.on_clicked(ann.down)
btn_prev = Button(ax_prev, "Prev")
btn_prev.on_clicked(ann.previous)
btn_undo = Button(ax_undo, "Undo")
btn_undo.on_clicked(ann.undo)
btn_plot = Button(ax_plot, "Plot")
btn_plot.on_clicked(ann.preview)
btn_comm = Button(ax_comm, "Commit")
btn_comm.on_clicked(ann.commit)
btn_next = Button(ax_next, "Next")
btn_next.on_clicked(ann.next)
btn_save = Button(ax_save, "Dev")
btn_save.on_clicked(ann.save_next)
btn_edit = Button(ax_edit, "Edit")
btn_edit.on_clicked(ann.edit_ghost)
btn_del = Button(ax_del, "Delete")
btn_del.on_clicked(ann.delete_ghost)

# ---------------------------------------------------------
# Appearance controls
# ---------------------------------------------------------

# Colour dropdown button
ax_colour = fig.add_axes(
    [0.86, 0.6, 0.11, 0.05]
)

btn_colour = Button(
    ax_colour,
    "Colour ▼"
)


# Colour options
ax_viridis = fig.add_axes(
    [0.86, 0.55, 0.11, 0.04]
)

ax_magma = fig.add_axes(
    [0.86, 0.5, 0.11, 0.04]
)

ax_plasma = fig.add_axes(
    [0.86, 0.45, 0.11, 0.04]
)

ax_inferno = fig.add_axes(
    [0.86, 0.4, 0.11, 0.04]
)

ax_hot = fig.add_axes(
    [0.86, 0.35, 0.11, 0.04]
)

ax_cool = fig.add_axes(
    [0.86, 0.3, 0.11, 0.04]
)

ax_cividis = fig.add_axes(
    [0.86, 0.25, 0.11, 0.04]
)

ax_gray = fig.add_axes(
    [0.86, 0.2, 0.11, 0.04]
)

magma = plt.get_cmap("magma")
viridis = plt.get_cmap("viridis")
plasma = plt.get_cmap("plasma")
inferno = plt.get_cmap("inferno")
hot = plt.get_cmap("hot")
cool = plt.get_cmap("cool")
cividis = plt.get_cmap("cividis")
gray = plt.get_cmap("gray")

btn_viridis = Button(ax_viridis, "Viridis", color=viridis(0.6), hovercolor=viridis(0.7))
btn_magma = Button(ax_magma, "Magma", color=magma(0.6), hovercolor=magma(0.7))
btn_plasma = Button(ax_plasma, "Plasma", color=plasma(0.6), hovercolor=plasma(0.7))
btn_inferno = Button(ax_inferno, "Inferno", color=inferno(0.6), hovercolor=inferno(0.7))
btn_hot = Button(ax_hot, "Hot", color=hot(0.6), hovercolor=hot(0.7))
btn_cool = Button(ax_cool, "Cool", color=cool(0.6), hovercolor=cool(0.7))
btn_cividis = Button(ax_cividis, "Cividis", color=cividis(0.6), hovercolor=cividis(0.7))
btn_gray = Button(ax_gray, "Gray", color=gray(0.6), hovercolor=gray(0.7))

# Start with colour options hidden
colour_axes = [
    ax_viridis,
    ax_magma,
    ax_plasma,
    ax_inferno,
    ax_hot,
    ax_cool,
    ax_cividis,
    ax_gray
]

for colour_ax in colour_axes:
    colour_ax.set_visible(False)

# Toggle whether the colour options are dropped down or not
def toggle_colour(event):

    visible = not ax_viridis.get_visible()

    for colour_ax in colour_axes:
        colour_ax.set_visible(visible)

    if visible:
        btn_colour.label.set_text("Colour ▲")
    else:
        btn_colour.label.set_text("Colour ▼")

    fig.canvas.draw_idle()


btn_colour.on_clicked(toggle_colour)


btn_magma.on_clicked(
    lambda event: select_colour("magma")
)

btn_viridis.on_clicked(
    lambda event: select_colour("viridis")
)

btn_plasma.on_clicked(
    lambda event: select_colour("plasma")
)

btn_inferno.on_clicked(
    lambda event: select_colour("inferno")
)

btn_hot.on_clicked(
    lambda event: select_colour("hot")
)

btn_cool.on_clicked(
    lambda event: select_colour("cool")
)

btn_cividis.on_clicked(
    lambda event: select_colour("cividis")
)

btn_gray.on_clicked(
    lambda event: select_colour("gray")
)

def select_colour(colour):
    # Change the colour to the selected colour
    ann.change_colour(colour)
    fig.canvas.draw_idle()


# Contrast +
ax_contrast_up = fig.add_axes(
    [0.86, 0.77, 0.11, 0.05]
)

btn_contrast_up = Button(
    ax_contrast_up,
    "Contrast +"
)

btn_contrast_up.on_clicked(
    ann.contrast_up
)


# Contrast -
ax_contrast_down = fig.add_axes(
    [0.86, 0.72, 0.11, 0.05]
)

btn_contrast_down = Button(
    ax_contrast_down,
    "Contrast -"
)

btn_contrast_down.on_clicked(
    ann.contrast_down
)


# Reset
ax_reset = fig.add_axes(
    [0.86, 0.83, 0.11, 0.05]
)

btn_reset = Button(
    ax_reset,
    "Reset"
)

btn_reset.on_clicked(
    ann.reset_appearance
)

# Dot colour
ax_dot_colour = fig.add_axes(
    [0.86, 0.66, 0.11, 0.05]
)

btn_dot_colour = Button(
    ax_dot_colour,
    "Dot Colour"
)

btn_dot_colour.on_clicked(
    ann.cycle_dot_colour
)

def on_drag(event):
    if event.inaxes == ax:
        ann.onclick(event)
    if event.dblclick:
        return


#### half baked idea to map keys to buttons for faster tracing, some clashes and 
#### overlaps with matplotlib base key commands so abandoned for now
def on_key(event):
    if event.key == "enter":
        ann.commit()
    elif event.key == "delete":
        ann.delete_ghost()
    elif event.key == "escape":
        ann.selected_ghost = None
        print("Selection cleared")
        ann.plot()
    elif event.key == ",":
        ann.previous()
    elif event.key == ".":
        ann.next()
#    elif event.key == "ArrowUp":
#        ann.up()
#    elif event.key == "ArrowDown":
#        ann.down()
    elif event.key == "'":
        ann.preview()
    elif event.key == ";":
        ann.undo()
    elif event.key == "#":
        ann.edit_ghost()
    elif event.key == "/":
        ann.save_next()

fig.canvas.mpl_connect("button_press_event", on_drag)
fig.canvas.mpl_connect("motion_notify_event", on_drag)
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()