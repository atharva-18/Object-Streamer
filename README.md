# Object-Stream
Python program for real-time object detection with monocular depth estimation using conditional GANs (Isola et al., 2017).

![Status](https://img.shields.io/badge/Status-Experimental-green.svg)
![Generic badge](https://img.shields.io/badge/Python-3.7.3-Blue.svg)

### Dependencies

<ul>
<li>PyTorch</li>
<li>OpenCV</li>
<li>Nvidia GPU</li>
</ul>

Pix2Pix model is trained so as to learn the translation between an RGB image and its stereo depth estimate.
A modified version of this program is currently being used as a part of state estimation in a driverless car currently in development by DJSRacing.

### Usage

To get started, run - ```pip install -r requirements.txt``` to install the required dependencies.

This is assuming that Ubuntu 18.04 LTS is installed on your computer with the required libraries for x264, DIVX, etc; if you are using any other OS, make sure you that are using the appropriate codec for video encoding.

<b>Arguments</b><br>

The following arguments are required before running - 

Argument         | Type                           | Description
---------------- | ------------------------------ | ---------------------------------------------------------------------------------
camera_index     | integer                        | Index number of the camera that you want to use (web-cam is usually 0).
save_path        | text                           | Save path relative to the current working directory.
duration         | integer                        | Recording duration in seconds.
fps              | integer                        | Frames per second of the recorded video

For example,

```python3 objstr.py 0 "" 10 30``` <br>

This saves the video in the current working directory of length 10 seconds at 30 FPS.

### Notes 

1. The current release is experimental and a lot of bug fixes and features are to be released in the future.
