from moviepy.editor import ColorClip, ImageClip, CompositeVideoClip
import numpy as np

bg = ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=2)

fg_arr = np.zeros((148, 1030, 4), dtype=np.uint8)
fg_arr[:,:,3] = 255 # Full opacity mask
fg = ImageClip(fg_arr).set_duration(2)

def pos(t):
    return (-1030, 500) # x = -1030 (exactly bordering left edge)

fg = fg.set_position(pos)

comp = CompositeVideoClip([bg, fg])

try:
    comp.make_frame(0.1)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
