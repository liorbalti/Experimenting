# Experimenting

Extracting and initial processing of image data from videos of dual-fluorescence imaging of ant feeding experiments

The folder "raw_data_processing" includs all relevant scripts.

## call_BugTag_Lior.py
A script to run the BugTag software on all experimental videos.

Yields tracking data for each ant:
- x and y coordinates
- angle of the tag
- error estimation (0 - good detection, 5 - high chance of error)

## AngleCorrection.py
An interactive script to run once per experiment, for manual calibration of tag orientation (defining the the ant's heading direction relative to the tag's direction).

## main.py
The main image processing script.

Identifies fluorescent blobs from fluorescence imaging camera, and assigns each blob to the appropriate tag.

Outputs a dataset that includes, for each ant, for every time step:
- x and y coordinates
- heading direction
- detection error estimation
- total fluorescence in her crop from each color

This script relies on classes from the following files:
- **Reader.py:** for reading the video/image files
- **TagsInfo.py:** for handling the information on the ants' tags
- **Experiment.py:**  for handling experiment-specific parameters
- **Frame.py:**  for manipulations on each frame of the video
- **BlobAnalysis.py:** for fluorescence detection
