import cv2 as cv
import numpy as np
import tensorflow as tf

FRAMES = 16
WIDTH = 128
HEIGHT = 128

def extract_frames(video_path):
    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // FRAMES)
    frames = []
    for i in range(FRAMES):
        cap.set(cv.CAP_PROP_POS_FRAMES, i * frame_step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, (WIDTH, HEIGHT))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Ensure RGB
        frames.append(frame)
    cap.release()
    # Pad if fewer than 16 frames
    while len(frames) < FRAMES:
        frames.append(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    return np.array(frames) / 255.0  # Normalize to [0,1]

def predict_video(model, video_path):
    frames = extract_frames(video_path)
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    prediction = model.predict(frames)[0][0]
    label = 'shoplifters' if prediction > 0.5 else 'non shoplifters'
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence