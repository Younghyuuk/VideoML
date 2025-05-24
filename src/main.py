import cv2 as cv
import torch
import sys
import time

# 1) Load your YOLOv5 model once at import time
#    Replace 'path/to/best.pt' with your actual .pt file (or use 'yolov5s' for the COCO model)
# at top of your script
# ─── 1) Load & prep YOLOv5 ─────────────────────────────────────────────────────
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',         # or 'custom' + path='best.pt'
    pretrained=True
)
model.conf = 0.5       # confidence threshold
model.eval()           # turn off training behaviors
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


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

def play_video(path, target_fps=60, scale=0.8, skip=3, det_size=(640,360)):
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: could not open {path}")
        sys.exit(1)

    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    window = "Playback"
    cv.namedWindow(window, cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("Frame", window, 0, total, lambda pos: cap.set(cv.CAP_PROP_POS_FRAMES, pos))

    delay     = int(1000/target_fps)
    prev_time = time.time()
    font      = cv.FONT_HERSHEY_SIMPLEX

    paused    = False
    frame_idx = 0
    last_dets = []

    while True:
        # ── read or hold ──────────────────────────
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
        else:
            # when paused, we still keep `frame` from last read
            pass

        # ── decide whether to run detection ───────
      # … inside your while True loop …

        # ── run detection every skip frames or when paused ─────────────────
        do_detect = paused or (frame_idx % skip == 0)
        if do_detect:
            small = cv.resize(frame, det_size)  # e.g. (640,360)
            with torch.no_grad():
                results = model(small)           # raw BGR → model handles letterbox, RGB, CHW, /255 …
            last_dets = results.xyxy[0].cpu().numpy()

        # ── draw detections ────────────────────────────────────────────────────
        for *box, conf, cls in last_dets:
            x1,y1,x2,y2 = map(int, box)
            label      = model.names[int(cls)]
            color      = (0,0,255) if label=='person' else (255,0,0)
            cv.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv.putText(frame, f"{label} {conf:.2f}", (x1,y1-6),
                    font, 0.6, color, 2)


        # ── resize + FPS overlay ─────────────────
        disp = rescaleFrame(frame, scale)
        now  = time.time()
        fps  = 1.0/(now - prev_time) if now!=prev_time else 0.0
        prev_time = now
        cv.putText(disp, f"FPS: {fps:.1f}", (10,30), font, 1, (0,255,0), 2)

        # ── sync slider & show ───────────────────
        cur_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        cv.setTrackbarPos("Frame", window, cur_frame)
        cv.imshow(window, disp)

        # ── key handling ─────────────────────────
        key = cv.waitKey(delay) & 0xFF
        if key == ord('q'):            break
        elif key == ord(' '):          paused = not paused
        elif key == 81:                # left arrow
            paused = True
            cap.set(cv.CAP_PROP_POS_FRAMES, max(0, cur_frame-1))
            frame_idx = cur_frame-1
        elif key == 83:                # right arrow
            paused = True
            cap.set(cv.CAP_PROP_POS_FRAMES, min(total, cur_frame+1))
            frame_idx = cur_frame+1

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    video_path = "videos/ShortValo.mp4"
    # 1) Play & detect
    play_video(video_path)
    # 2) (Optional) show a still
    # showImage("path/to/some/frame.png")
