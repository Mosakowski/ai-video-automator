from moviepy import ColorClip, ImageClip, CompositeVideoClip
import numpy as np

# Create a background
bg = ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=2)

# Create a foreground clip with alpha channel (RGBA simulating the header: 1030x148)
fg_arr = np.zeros((148, 1030, 4), dtype=np.uint8)
fg_arr[:,:,3] = 255 # Full opacity mask
fg = ImageClip(fg_arr, is_mask=False, transparent=True).with_duration(2)

# Position it partially offscreen
def pos(t):
    return (-1030 + 50, 500) # x = -980 (partially off left edge)

fg = fg.with_position(pos)

comp = CompositeVideoClip([bg, fg])

try:
    comp.get_frame(0.1)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
