import cv2 as cv # type: ignore
import sys

def showImage(path):
    image = cv.imread(path)
    if image is None:
        print(f"Error: could not open {path}")
        sys.exit(1)

    cv.imshow("Image", image)
    cv.waitKey(0)  # wait for a key press to close the window
    cv.destroyAllWindows()


def play_video(path):
    capture = cv.VideoCapture(path)
    if not capture.isOpened():
        print(f"Error: could not open {path}")
        sys.exit(1)

    while True:
        isTrue, frame = capture.read()

        frame_resized = rescaleFrame(frame, 0.8)

        if not isTrue:
            break
        # cv.imshow("Playback", frame)
        cv.imshow("Resized Playback", frame_resized)
        if cv.waitKey(30) & 0xFF == ord('q'):  # press q to quit early
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

