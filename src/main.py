import cv2 as cv
import torch
import sys
import time

# 1) Load your YOLOv5 model once at import time
#   use a .pt file later if u have a trained model u can use
# ─── 1) Load & prep YOLOv5 ─────────────────────────────────────────────────────
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5n',         # or 'custom' + path='best.pt'
    pretrained=True
)
model.conf = 0.4       # confidence threshold
model.eval()           # turn off training behaviors
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)


def showImage(path):
    image = cv.imread(path)
    if image is None:
        print(f"Error: could not open {path}")
        sys.exit(1)

    cv.imshow("Image", image)
    cv.waitKey(0)  # wait for a key press to close the window
    cv.destroyAllWindows()

def rescaleFrame(frame, scale=0.75):
    width  = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

def changeRes(capture, width, height):
    capture.set(3, width)   # set width
    capture.set(4, height)  # set height
    return capture

# plays the video itself with no annotations
def play_video(path, target_fps=60, scale=0.8):
    cap = cv.VideoCapture(path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {path!r}")
    window = "Raw Playback"
    cv.namedWindow(window, cv.WINDOW_AUTOSIZE)
    frame_time = 1.0 / target_fps
    prev_time = time.time()
    font = cv.FONT_HERSHEY_SIMPLEX

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret: break

        disp = rescaleFrame(frame, scale)
        
        # Calculate the instantaneous FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
        prev_time = curr_time
        cv.putText(
            disp,
            f"FPS: {fps:.1f}",
            (10, 30),
            font,
            1,               # font scale
            (0, 255, 0),     # green color
            2                # thickness
        )
        cv.imshow(window, disp)

        elapsed = time.time() - start
        delay = max(1, int((frame_time - elapsed)*1000))
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyWindow(window)


# object detection
def play_video_annotated(path, scale=0.8, target_fps=60, skip=15, det_size=(640, 360)):
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path!r}")

    total      = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    window     = "Playback"
    cv.namedWindow(window, cv.WINDOW_AUTOSIZE)
    cv.createTrackbar(
        "Frame", window, 0, total,
        lambda pos: cap.set(cv.CAP_PROP_POS_FRAMES, pos),
    )

    paused     = False
    frame_idx  = 0
    last_dets  = []
    prev_time  = time.time()
    frame_time = 1.0 / target_fps

    # cache lookups
    names       = model.names
    resize_det  = cv.resize
    resize_disp = rescaleFrame
    put_text    = cv.putText
    rectangle   = cv.rectangle
    font        = cv.FONT_HERSHEY_SIMPLEX

    while True:
        loop_start = time.time()

        # read next frame
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # detect every `skip` frames
            if frame_idx % skip == 0:
                small = resize_det(frame, det_size)
                with torch.no_grad():
                    results = model(small)
                last_dets = results.xyxy[0].cpu().numpy()

        # draw last detection boxes
        for *box, conf, cls in last_dets:
            if names[int(cls)] != "person":
                continue
            x1, y1, x2, y2 = map(int, box)
            rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # rescale display
        disp = resize_disp(frame, scale)

        #  compute & overlay FPS + frame index
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cur = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        put_text(disp, f"FPS: {fps:.1f}",       (10, 30), font, 1, (0,255,0), 2)
        put_text(disp, f"Frame: {cur}/{total}", (10, 65), font, 1, (0,255,0), 2)

        # update slider & show
        cv.setTrackbarPos("Frame", window, cur)
        cv.imshow(window, disp)

        # wait the remainder of frame_time
        elapsed = time.time() - loop_start
        delay   = max(1, int((frame_time - elapsed) * 1000))
        key     = cv.waitKey(delay) & 0xFF

        # input handling
        if   key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv.destroyAllWindows()





if __name__ == "__main__":
    # Example usage:
    video_path = "videos/ShortValo.mp4"
    # 1) Play & detect
    play_video(video_path)

    # play_video_annotated(video_path)
    
    # 2) (Optional) show a still
    # showImage("path/to/some/frame.png")
