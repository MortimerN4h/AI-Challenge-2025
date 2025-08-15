'''
REQUEST 
    TAO MOI TRUONG RIENG DE TEST
    TER: 
        python3 -m venv yolo_env
        source yolo_env/bin/activate
    NEU NHU MUON OUT KHOI MOI TRUONG AO THI: 
    TER: deactivate

    CAI DAT THU VIEN: 
    TER: pip install opencv-python numpy
    (OPTIONAL):NEU DUNG NVIDIA-CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
'''

import cv2
import os
from skimage.metrics import structural_similarity as ssim

def compare_frames(frame1, frame2, threshold=0.95):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score >= threshold

def extract_unique_frames(video_path, output_folder, group_size=30, threshold=0.95):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path) # video_path thay = 0 de test = webcam
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_frames = []
    frame_idx = 0
    saved_idx = 0

    while True:
        group_frames = []
        for _ in range(group_size):
            ret, frame = cap.read()
            if not ret:
                break
            group_frames.append((frame_idx, frame))
            frame_idx += 1

        if not group_frames:
            break

        sample_indices = [0, len(group_frames)//2, len(group_frames)-1]
        for idx in sample_indices:
            frame_id, frame = group_frames[idx]
            if not saved_frames or not compare_frames(saved_frames[-1], frame, threshold):
                saved_frames.append(frame)
                save_path = os.path.join(output_folder, f"frame_{frame_id}.jpg")
                cv2.imwrite(save_path, frame)
                saved_idx += 1

    cap.release()
    print(f"[DONE] Saved {saved_idx} unique frames to {output_folder}")

video_path = "video.mp4"
output_folder = "frames_output"
extract_unique_frames(video_path, output_folder, group_size=30, threshold=0.95)
