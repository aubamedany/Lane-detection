import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from collections import deque

# Needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from preprocess import Preprocess

IMG_SIZE = [640, 360]
def main():
    process = Preprocess()
    # print(type(process.camera.run_calibrate()))
    # mtx, dist=process.camera.run_calibrate()
    process.camera.run_calibrate()
    test_images = []
    test_filenames = glob.glob("{}/*".format("images/test_images"))
    for f in test_filenames:
        img = cv2.imread(f)
        img = cv2.resize(img,IMG_SIZE )
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        test_images.append(np.array(img)) 


    # b, g, r = cv2.split(test_images[8])
    # img = cv2.merge([r, g, b])
    # warp,img = process.process_img(img)
    # print(process.get_steer_angle(warp))
    # cv2.imshow("test",img)
    # # cv2.imshow("test",process.process_img(img))
    # cv2.waitKey(0)
    video = cv2.VideoCapture("camera_front_view.mp4")
    video.set(cv2.CAP_PROP_FPS, 10)

    while (video.isOpened()):
        try:
            flag, img = video.read()
            img = cv2.resize(img, IMG_SIZE)
            if not flag: break
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            warp,img = process.process_img(img)
            cv2.imshow("test",img)
            if cv2.waitKey(1) == ord('q'):break
        except Exception as e:
            print(e)


main()

