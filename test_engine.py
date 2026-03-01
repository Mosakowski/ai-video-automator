import cv2
import numpy as np
import os
from moviepy import AudioArrayClip
from video_engine import process_video_pipeline

def create_test_assets():
    os.makedirs("test_assets", exist_ok=True)
    
    # 1. Create two dummy 1920x1080 images (horizontal) to see how it crops
    img1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img1[:] = (0, 0, 255) # Red
    cv2.circle(img1, (960, 540), 100, (255, 255, 255), -1) # white circle center
    cv2.imwrite("test_assets/test1.jpg", img1)
    
    img2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img2[:] = (255, 0, 0) # Blue
    cv2.rectangle(img2, (200, 200), (600, 600), (0, 255, 0), -1) # Green rect
    cv2.imwrite("test_assets/test2.jpg", img2)
    
    # 2. Create a dummy 5-second audio clip
    import math
    duration = 5.0
    fps = 44100
    t = np.linspace(0, duration, int(fps * duration))
    # sine wave
    audio_array = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Ensure it's 2D for Moviepy
    audio_array = np.vstack((audio_array, audio_array)).T
    audio_clip = AudioArrayClip(audio_array, fps=fps)
    audio_clip.write_audiofile("test_assets/test_audio.mp3")

if __name__ == "__main__":
    print("Creating test assets...")
    create_test_assets()
    
    print("Running video pipeline...")
    def status_cb(msg):
        print("STATUS:", msg)
        
    process_video_pipeline(
        ["test_assets/test1.jpg", "test_assets/test2.jpg"],
        "test_assets/test_audio.mp3",
        "output.mp4",
        logo_path="test_assets/test_logo.png",
        logo_position="Upper-Middle-Center",
        logo_opacity=0.5,
        status_callback=status_cb
    )
    print("Test finished successfully!")
