import cv2
import sys

def play_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: could not open {path}")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Playback", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # press q to quit early
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "videos/ShortValo.mp4"
    play_video(video_path)
