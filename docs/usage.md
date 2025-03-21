# Usage

## Inputs

### General Elements

__Operate on:__ Shows the currently selected movie window. On this movie the
operations take place.

__Import movie:__ The <kbd>Import movie</kbd> button lets yout import a FastSPM
.h5 file.

__Load config:__ The <kbd>Load config</kbd> loads a config file in the TOML
format.

__Save config:__ The <kbd>Save Config</kbd> saves the currently set parameters
as a config file in TOML format.
 __Channel:__ You can choose another channel of the currently selected movie via
the dropdown. Create a new movie window with the <kbd>New</kbd> button.

__Colormap:__ Lets you choose another colormap which gets applied for all open
movie windows via the dropdown.

__Histogram:__ Clicking the <kbd>Histogram</kbd> button shows another window
with a histogram of the image intensities. In this window the cutoff of the
colormap can be chosen by:

1. The range slider
2. Absolute minimum and maximum values
3. Minimum and Maximum percentile values

!!! note
    The histogram window will not get updated when the image values change.
    It is best to open it again after a correction function is applied.

### Phase Correction

- __Apply auto x-phase:__ Determine x-phase via correlation.
- __Additional x-phase:__ Value added to the x-phase.
- __Manual y-phase:__ Override y-phase value from metadata.
- __Index frame to correlate:__ Index of the frame used for correlation.
- __Sigma Gauss:__
- <kbd>Apply</kbd> : Perform the phase correction for the currently selected
   movie and update it.
- <kbd>New</kbd> : Perform the phase correction for the currently selected movie
    and create a new one.

### FFT Filters

- __Filter x:__
- __Filter y:__
- __Filter x overtones:__
- __Filter high pass:__
- __Filter pump:__
- __Filter noise:__
- __Display spectrum:__
- __Filter broadness:__
- __Num x overtones:__
- __High pass params:__
- __FFT display range:__
- __Pump freqs:__
- <kbd>Apply</kbd> :_ Perform FFT filtering for the currently selected movie
    and update it.
- <kbd>New</kbd> : Perform FFT filtering for the currently selected movie
    and create a new one.

### Creep Correction

- __None:__ Do not apply a creep correction but just perform an interpolation.
- __sin:__ Creep correction by fitting a sin function.
- __bezier:__ Creep correction by fitting a bezier curve.
- __root:__ Creep correction by fitting a root function.
- __Bezier parameters:__
    - Weight boundry
    - Creep num cols
- __Non-bezier parameters:__
    - Initial guess
    - Guess Ind
- <kbd>Apply</kbd> : Perform the creep correction for the currently selected movie
    and update it.
- <kbd>New</kbd> : Perform the creep correction for the currently selected movie
    and create a new one.

### Modify

- <kbd>Toggle crop selection</kbd> : Toggle the ability to select the crop area by
    dragging the mouse in a movie file. 
- <kbd>Crop</kbd> : Crop the frames of the movie according to the crop selection
    area.
- __Cut frames:__ Input for start and end frame for movie cutting.
- <kbd>Cut</kbd> : Apply cutting of the currently selected movie for the frames
    specified by 'Cut frames'.

### Drift

- __common:__ Cut out the largest common area for the drift corrected frames.
- __full:__ Apply padding around drift corrected frames.
- __Mode:__
    - correlation: Drift correction via cross correlation of two frames.
    - stackreg: Drift correction via [pystackref][https://pystackreg.readthedocs.io/en/latest/]
    - known: Drift correction via a previously saved '.drift.txt' file.
- __Correlation paramters:__ (apply only if 'Mode' is set to correlation)
    - Stepsize: The moving window size of frames to correlate to each other.
      Applies on if 'Mode' is set to correlation.
- __StackReg parameters:__ (apply only if 'Mode' is set to stackreg):
    - previous: Perform image registration with respect to the previous frame.
    - first: Perform image registration with respect to the first frame.
    - mean: Perform image registration with respect to the average over all frames.
- __Filter:__
    - Boxcar width: Width of the boxcar filter that is used for smoothing the drift path.
    - Median filter: Tick for smoothing the drift path with a median filter of size 3.
- <kbd>Apply</kbd> : Perform the drift correction for the currently selected movie
    and update it.
- <kbd>New</kbd> : Perform the drift correction for the currently selected movie
    and create a new one.

### Image Correction

- __Corretion type:__
- __Align type:__
- <kbd>Apply</kbd> : Perform the image correction for the currently selected movie
    and update it.
- <kbd>New</kbd> : Perform the image correction for the currently selected movie
    and create a new one.

### Image Filter

- __Filter type:__
- __Pixel width:__
- <kbd>Apply</kbd> : Perform the image filter for the currently selected movie
    and update it.
- <kbd>New</kbd> : Perform the image filter for the currently selected movie
    and create a new one.

### Export

- __MP4:__ Tick for exporting the movie as an .mp4 file.
- __TIFF:__ Tick for exporting the movie as a multipage .tiff file.
- __Frames:__ Tick for exporting one or multiple frames of the movie.
- __Scaling:__ Scaling multiplication factor that gets applied in x and y dimensions.
- __FPS factor:__ Factor that gets multplied to frames per second from movie acquistion.
- __Frame export images:__ Sets the range of frames to export
- __Frame export format:__ Sets the file format for saving frames.
- __Auto label:__ Tick for labeling frames in the .mp4 file or .png file.
- <kbd>Export</kbd>: Perform the movie and frame export.
