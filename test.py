#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from collections import deque

# Needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


IMG_SIZE = [640, 360]
original_frame = cv2.imread("image.jpg")
img = cv2.resize(original_frame,(640,360))
plt.imshow(img)
plt.show()