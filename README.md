# Object-Stream
Python program for capturing videos along with real-time object detection.

![Status](https://img.shields.io/badge/Status-Experimental-green.svg)
![Generic badge](https://img.shields.io/badge/Python-3.7.3-Blue.svg)

### Dependencies

<ul>
<li>PyTorch</li>
<li>OpenCV</li>
</ul>

A modified version of this program is currently being used for state estimation in a driverless car currently in development by DJSRacing.

### Usage

To get started, run - ```pip install -r requirements.txt``` to install the required dependencies.

This is assuming Ubuntu 18.04 LTS is installed on your computer with the required libraries for x264, DIVX, etc; if you are using any other OS, make sure you that are using the appropriate codec for video encoding.

<b>Arguments</b><br>

The following arguments are required before running - 

Argument         | Type                           | Description
---------------- | ------------------------------ | ---------------------------------------------------------------------------------
camera_index     | integer                        | Index number of the camera that you want to use (web-cam is usually 0).
save_path        | text                           | Save path relative to the current working directory.
duration         | interger                       | Recording duration in seconds.
fps              | integer                        | Frames per second of the recorded video

For example,

```python3 objstr.py 0 "" 10 30``` <br>

This saves the video the in current working directory of length 10 seconds at 30 FPS.

### Notes 

1. The current release is experimental and a lot of bug fixes and features are to be released in the future.