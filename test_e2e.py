import os
from video_engine import process_video_pipeline
import traceback

os.makedirs("temp", exist_ok=True)

# create dummy image
from PIL import Image
Image.new("RGB", (1080, 1920), color="blue").save("temp/dummy.jpg")

try:
    process_video_pipeline(
        image_paths=["temp/dummy.jpg"],
        audio_path=None,
        output_path="temp/out.mp4",
        logo_path=None,
        header_text="TEST HEADER\nLINE 2",
        header_position="Bottom-Right",
        header_style="1. Neon Edge",
        header_animation="1. Slide-in (Side)"
    )
    print("Pipeline executed successfully!")
except Exception as e:
    print("PIPELINE EXCEPTION:")
    traceback.print_exc()
