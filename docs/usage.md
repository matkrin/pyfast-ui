# Usage

## Inputs

### General Elements

#### Operate on

Shows the currently selected movie window. On this movie the operations take
place.

#### Import movie

The <kbd>Import movie</kbd> button lets yout import a FastSPM .h5 file.

#### Load config

The <kbd>Load config</kbd> loads a config file in the TOML format.

#### Save config

The <kbd>Save Config</kbd> saves the currently set parameters as a config file
in TOML format.

#### Channel

You can choose another channel of the currently selected movie via the dropdown.
Create a new movie window with the <kbd>New</kbd> button.

#### Colormap

Lets you choose another colormap which gets applied for all open movie windows
via the dropdown.

#### Histogram

Clicking the <kbd>Histogram</kbd> button shows another window with a histogram
of the image intensities. In this window the cutoff of the colormap can be
chosen by:

1. The range slider
2. Absolute minimum and maximum values
3. Minimum and Maximum percentile values

!!! note
    The histogram window will not get updated when the image values change.
    It is best to open it again after a correction function is applied.

### Phase Correction

- Apply auto x-phase: Determine x-phase via correlation.
- Additional x-phase: Value added to the x-phase.
- Manual y-phase: Override y-phase value from metadata.
- Index frame to correlate: Index of the frame used for correlation.
- Sigma Gauss
- <kbd>Apply</kbd> : Perform the phase correction for the currently selected movie
    and update it.
- <kbd>New</kbd> : Perform the phase correction for the currently selected movie
    and create a new one.

### FFT Filters

- Filter x
- Filter y
- Filter x overtones
- Filter high pass
- Filter pump
- Filter noise
- Display spectrum
- Filter broadness
- Num x overtones
- High pass params
- FFt display range
- Pump freqs
- <kbd>Apply</kbd> : Perform FFT filtering for the currently selected movie
    and update it.
- <kbd>New</kbd> : Perform FFT filtering for the currently selected movie
    and create a new one.

### Creep Correction

- None
- sin
- bezier
- root
- Bezier parameters:
  - Weight boundry
  - Creep num cols
- Non-bezier parameters:
  - Initial guess
  - Guess Ind
- <kbd>Apply</kbd> : Perform the creep correction for the currently selected movie
    and update it.
- <kbd>New</kbd> : Perform the creep correction for the currently selected movie
    and create a new one.

### Modify

- <kbd>Toggle crop selection</kbd>
- <kbd>Crop</kbd>
- Cut frames
- <kbd>Cut</kbd>

### Drift

- common: Cut out the largest common area for the drift corrected frames.
- full: Apply padding around drift corrected frames.
- Mode:
    - correlation: Drift correction via cross correlation of two frames.
    - stackreg: Drift correction via [pystackref][https://pystackreg.readthedocs.io/en/latest/]
    - known: Drift correction via a previously saved '.drift.txt' file.
- Correlation paramters (apply only if 'Mode' is set to correlation):
    - Stepsize: The moving window size of frames to correlate to each other.
      Applies on if 'Mode' is set to correlation.
- StackReg parameters (apply only if 'Mode' is set to stackreg):
    - previous: Perform image registration with respect to the previous frame.
    - first: Perform image registration with respect to the first frame.
    - mean: Perform image registration with respect to the average over all frames.
- Filter:
    - Boxcar width: Width of the boxcar filter that is used for smoothing the drift path.
    - Median filter: Tick for smoothing the drift path with a median filter of size 3.
- <kbd>Apply</kbd> : Perform the drift correction for the currently selected movie
    and update it.
- <kbd>New</kbd> : Perform the drift correction for the currently selected movie
    and create a new one.

### Image Correction

- Corretion type
- Align type
- <kbd>Apply</kbd> : Perform the image correction for the currently selected movie
    and update it.
- <kbd>New</kbd> : Perform the image correction for the currently selected movie
    and create a new one.

### Image Filter

- Filter type
- Pixel width
- <kbd>Apply</kbd> : Perform the image filter for the currently selected movie
    and update it.
- <kbd>New</kbd> : Perform the image filter for the currently selected movie
    and create a new one.

### Export

- MP4: Tick for exporting the movie as an .mp4 file.
- TIFF: Tick for exporting the movie as a multipage .tiff file.
- Frames: Tick for exporting one or multiple frames of the movie.
- Scaling: Scaling multiplication factor that gets applied in x and y dimensions.
- FPS factor: Factor that gets multplied to frames per second from movie acquistion.
- Frame export images: Sets the range of frames to export
- Frame export format: Sets the file format for saving frames.
- Auto label: Tick for labeling frames in the .mp4 file or .png file.
- <kbd>Export</kbd>: Perform the movie and frame export.
