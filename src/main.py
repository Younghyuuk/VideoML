import cv2 as cv # type: ignore
import sys
# import panda as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
# from tqdm import tqdm # type: ignore
import subprocess # type: ignore
import time
from glob import glob
import IPython.display as IPD 

# ffmpeg can be used to convert videos to images or convert different files to videos like .mov to .mp4 and audio files

def showImage(path):
    image = cv.imread(path)
    if image is None:
        print(f"Error: could not open {path}")
        sys.exit(1)

    cv.imshow("Image", image)
    cv.waitKey(0)  # wait for a key press to close the window
    cv.destroyAllWindows()


def play_video(path, target_fps=60):
    capture = cv.VideoCapture(path)
    fps = capture.get(cv.CAP_PROP_FPS)  # get the frames per second (fps) of the video
   
    if not capture.isOpened():
        print(f"Error: could not open {path}")
        sys.exit(1)

    # Compute delay between frames (in ms) for a 60 FPS target
    delay = int(1000 / target_fps)    # -> 16 ms
    prev_time = time.time()
    font = cv.FONT_HERSHEY_SIMPLEX
   


    while True:
        isTrue, frame = capture.read()

   
        # isTrue is retention of the frame, frame is the actual frame
        if not isTrue:
            break

        frame_resized = rescaleFrame(frame, 0.8)
        
        # Calculate the instantaneous FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
        prev_time = curr_time
        cv.putText(
            frame_resized,
            f"FPS: {fps:.1f}",
            (10, 30),
            font,
            1,               # font scale
            (0, 255, 0),     # green color
            2                # thickness
        )

        # cv.imshow("Playback", frame)
        cv.imshow("Resized Playback", frame_resized)
        if cv.waitKey(delay) & 0xFF == ord('q'):  # press q to quit early
            break

    capture.release()
    cv.destroyAllWindows()



#  works better for existing videos
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Only works on live videos
def changeRes(capture, width, height):
    capture.set(3, width)  # set width
    capture.set(4, height)  # set height
    return capture

if __name__ == "__main__":
    video_path = "videos/ShortValo.mp4"
    play_video(video_path)
    # IPD.display(IPD.Video(video_path, embed=True, width=800))

