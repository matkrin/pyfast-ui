# Usage

## Inputs

### General Elements

#### Operate on

Shows the currently selected movie window. On this movie the operations take
place.

#### Import movie

The <kbd>Import movie</kbd> button lets yout import a FastSPM .h5 file

#### Load config

The <kbd>Load config</kbd> loads a config file in the TOML format.

#### Save config

The <kbd>Save Config</kbd> saves the currently set parameters as a config file
in TOML format.

#### Channel

You can choose another channel of the currently selected movie via the dropdown.
Create a new movie window with the <kbd>New</kbd> button

#### Colormap

Lets you choose another colormap which gets applied for all open movie windows
via the dropdown.

#### Histogram

Clicking the <kbd>Histogram</kbd> button shows another window with a histogram
of the image intensities. In this window the cutoff of the colormap can be
choose by:

1. The range slider
2. Absolute minimum and maximum values
3. Minimum and Maximum percentile values

!!! note The histogram window will not get updated when the image values change.
It is best to open it again after a correction function is applied.

### Phase Correction

- Apply auto x-phase
- Additional x-phase
- Manual y-phase
- Index frame to correlate
- Sigma Gauss
- <kbd>Apply</kbd>
- <kbd>New</kbd>

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
- <kbd>Apply</kbd>
- <kbd>New</kbd>

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
- <kbd>Apply</kbd>
- <kbd>New</kbd>

### Modify

- <kbd>Toggle crop selection</kbd>
- <kbd>Crop</kbd>
- Cut frames
- <kbd>Cut</kbd>

### Drift

- common
- full
- Mode:
  - correlation
  - stackreg
  - known
- Correlation paramters:
  - Stepsize
- StackReg parameters:
  - previous
  - first
  - mean
- Filter:
  - Boxcar width:
  - Median filter:
- <kbd>Apply</kbd>
- <kbd>New</kbd>

### Image Correction

- Corretion type
- Align type
- <kbd>Apply</kbd>
- <kbd>New</kbd>

### Image Filter

- Filter type
- Pixel width
- <kbd>Apply</kbd>
- <kbd>New</kbd>

### Export

- MP4
- TIFF
- Frames
- Scaling
- FPS factor
- Frame export images
- Frame export channel
- Frame export format
- Auto label
- <kbd>Export</kbd>
