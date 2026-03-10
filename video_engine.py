import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, VideoClip
from moviepy.video.fx.CrossFadeIn import CrossFadeIn

# Initialize YOLO model globally so it's only loaded once
model = YOLO('yolov8n.pt')

def get_yolo_center(image_path):
    """
    Analyzes an image with YOLOv8 to find the center of detected objects.
    Returns (center_x, center_y) in relative coordinates (0.0 to 1.0) or absolute pixels.
    Let's return absolute pixels for easier calculation.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0, 0
    
    h, w = img.shape[:2]
    
    # Run YOLO detection
    results = model(img, verbose=False)
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        # Fallback to geometric center
        return w // 2, h // 2
    
    # Calculate the center of the largest box, or average of all boxes
    # Let's use the average center of all detected boxes weighted by confidence, or just simply the center of the largest bounding box.
    largest_area = 0
    best_center = (w // 2, h // 2)
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        if area > largest_area:
            largest_area = area
            best_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
    return best_center

def create_ken_burns_clip(image_path, target_w, target_h, duration, center_x, center_y, scale_start=1.05, scale_end=1.15):
    """
    Creates a MoviePy VideoClip with a Ken Burns effect.
    Uses 3x Oversampling and simple array cropping to completely eliminate jittering.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    orig_h, orig_w = img_bgr.shape[:2]
    orig_aspect = orig_w / orig_h
    target_aspect = target_w / target_h

    # --- STEP 1: CREATE THE GIGANTIC 3x OVERSAMPLED CANVAS ---
    OVERSAMPLE_FACTOR = 3.0
    huge_w = int(target_w * OVERSAMPLE_FACTOR)
    huge_h = int(target_h * OVERSAMPLE_FACTOR)

    # Fit original image into huge dimensions
    if orig_aspect > target_aspect:
        fit_w = huge_w
        fit_h = int(fit_w / orig_aspect)
    else:
        fit_h = huge_h
        fit_w = int(fit_h * orig_aspect)

    img_fit = cv2.resize(img_bgr, (fit_w, fit_h), interpolation=cv2.INTER_LANCZOS4)

    # Fill background for the huge canvas
    if orig_aspect > target_aspect:
        bg_h = huge_h
        bg_w = int(bg_h * orig_aspect)
    else:
        bg_w = huge_w
        bg_h = int(bg_w / orig_aspect)

    # Use INTER_LINEAR for background as it will be heavily blured anyway (much faster than LANCZOS4)
    bg_img = cv2.resize(img_bgr, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
    
    # Crop the overgrown background to exact huge target dimensions
    bg_x1 = (bg_w - huge_w) // 2
    bg_y1 = (bg_h - huge_h) // 2
    bg_cropped = bg_img[bg_y1:bg_y1+huge_h, bg_x1:bg_x1+huge_w]

    # Fast Blur Hack: downscale 4x -> light blur -> upscale (avoids blurring full resolution)
    small_h, small_w = huge_h // 4, huge_w // 4
    bg_small = cv2.resize(bg_cropped, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    bg_small_blurred = cv2.GaussianBlur(bg_small, (25, 25), 0)
    bg_blurred = cv2.resize(bg_small_blurred, (huge_w, huge_h), interpolation=cv2.INTER_LINEAR)

    # Paste the properly scaled main image into the center
    paste_x = (huge_w - fit_w) // 2
    paste_y = (huge_h - fit_h) // 2
    bg_blurred[paste_y:paste_y+fit_h, paste_x:paste_x+fit_w] = img_fit

    # Convert BGR to RGB for MoviePy compatibility
    huge_base_img_rgb = cv2.cvtColor(bg_blurred, cv2.COLOR_BGR2RGB)

    # --- STEP 2: MAP YOLO CENTER TO THE HUGE CANVAS ---
    scale_factor_fit = fit_w / orig_w
    huge_cx = (center_x * scale_factor_fit) + paste_x
    huge_cy = (center_y * scale_factor_fit) + paste_y

    # --- STEP 3: CALCULATE SLIDING WINDOW OVER THE HUGE CANVAS ---
    # The maximum size of the sliding window determines how zoomed in we are.
    # At scale 1.0, the window is the entire huge_w x huge_h. 
    # At 1.15 scale, the window is smaller, meaning we "zoom in"
    window_w_start = huge_w / scale_start
    window_h_start = huge_h / scale_start
    
    window_w_end = huge_w / scale_end
    window_h_end = huge_h / scale_end

    # Ensure the window centers don't push the window outside the huge canvas bounds
    cx_start = min(max(huge_cx, window_w_start / 2.0), huge_w - window_w_start / 2.0)
    cy_start = min(max(huge_cy, window_h_start / 2.0), huge_h - window_h_start / 2.0)
    
    cx_end = min(max(huge_cx, window_w_end / 2.0), huge_w - window_w_end / 2.0)
    cy_end = min(max(huge_cy, window_h_end / 2.0), huge_h - window_h_end / 2.0)

    # --- STEP 4: OVERSAMPLED SLIDING WINDOW ANIMATION ---
    def make_frame(t):
        progress = t / duration
        
        # Smooth interpolation of the window size and position
        current_window_w = window_w_start + (window_w_end - window_w_start) * progress
        current_window_h = window_h_start + (window_h_end - window_h_start) * progress
        current_cx = cx_start + (cx_end - cx_start) * progress
        current_cy = cy_start + (cy_end - cy_start) * progress
        
        # Calculate exactly where the window begins
        x1_f = current_cx - current_window_w / 2.0
        y1_f = current_cy - current_window_h / 2.0
        
        # Cast to int (eliminates visible jitter)
        x1 = int(x1_f)
        y1 = int(y1_f)
        w_int = int(current_window_w)
        h_int = int(current_window_h)
        
        x2 = x1 + w_int
        y2 = y1 + h_int
        
        x1 = max(0, min(x1, huge_w - 1))
        y1 = max(0, min(y1, huge_h - 1))
        x2 = max(0, min(x2, huge_w))
        y2 = max(0, min(y2, huge_h))
        
        cropped = huge_base_img_rgb[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return resized

    from moviepy import VideoClip
    clip = VideoClip(make_frame, duration=duration)
    return clip

# --- EASING FUNCTIONS FOR ANIMATION ---
def ease_out_cubic(t):
    return 1 - pow(1 - t, 3)

def ease_out_back(t, overshoot=1.70158):
    return 1 + (overshoot + 1) * pow(t - 1, 3) + overshoot * pow(t - 1, 2)

def make_animated_position(t, hx, hy, target_w, target_h, box_w, box_h, duration, anim_type):
    """
    Returns the (x, y) position of the header at time t, based on the animation type.
    """
    anim_duration = 1.0 # 1 second intro/outro
    
    def clamp(val, min_val, max_val):
        return max(min_val, min(val, max_val))
    
    # INTRO
    if t < anim_duration:
        progress = t / anim_duration
        
        if anim_type == "1. Slide-in (Side)":
            ease = ease_out_cubic(progress)
            # Decide if left or right side based on final hx
            if hx < target_w / 2:
                start_x = -box_w - 50
            else:
                start_x = target_w + 50
                
            current_x = start_x + (hx - start_x) * ease
            return (int(clamp(current_x, -box_w + 1, target_w - 1)), int(hy))
            
        elif anim_type == "2. Pop-up (Bottom)":
            ease = ease_out_back(progress, overshoot=1.2)
            start_y = target_h + 50
            current_y = start_y + (hy - start_y) * ease
            return (int(hx), int(clamp(current_y, -box_h + 1, target_h - 1)))
            
    # OUTRO
    elif t > duration - anim_duration:
        time_left = duration - t
        progress = 1.0 - (time_left / anim_duration) # reverse progress 0 to 1
        ease = ease_out_cubic(progress) # Using simpler cubic out reversed feels like cubic in
        
        if anim_type == "1. Slide-in (Side)":
            if hx < target_w / 2:
                end_x = -box_w - 50
            else:
                end_x = target_w + 50
                
            current_x = hx + (end_x - hx) * ease
            return (int(clamp(current_x, -box_w + 1, target_w - 1)), int(hy))
            
        elif anim_type == "2. Pop-up (Bottom)":
            end_y = target_h + 50
            current_y = hy + (end_y - hy) * ease
            return (int(hx), int(clamp(current_y, -box_h + 1, target_h - 1)))
            
    # STATIC
    return (int(hx), int(hy))

def generate_dynamic_header_img(text, scale, color_hex, bg_color_hex, opacity, style, header_position):
    from PIL import Image, ImageFont, ImageDraw 
    import cairosvg
    import io

    if not text.strip():
        return Image.new('RGBA', (1,1), (0,0,0,0))
    
    lines = text.strip().split('\n')
    
    # Scale base metrics
    pad_x = int(40 * scale)
    pad_y = int(20 * scale)
    line_spacing = int(-10 * scale)
    font_size = int(75 * scale)
    
    # Text measurement using PIL 
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        
    dummy_draw = ImageDraw.Draw(Image.new('RGBA', (1,1)))
    line_dims = []
    total_w = 0
    total_h = 0
    
    for line in lines:
        if not line.strip():
            tw, th = 0, font_size
        else:
            try:
                left, top, right, bottom = dummy_draw.textbbox((0,0), line, font=font)
                tw = right - left
                th = bottom - top
            except AttributeError:
                tw = len(line) * int(35 * scale)
                th = font_size
            
        tw = max(1, int(tw))
        th = max(1, int(th))
        box_w = tw + pad_x * 2 
        box_h = th + pad_y * 2 
        line_dims.append({'text': line, 'tw': tw, 'th': th, 'bw': box_w, 'bh': box_h})
        
        if box_w > total_w: total_w = box_w
        total_h += box_h + line_spacing
        
    total_h -= line_spacing
    
    glow_radius = int(50 * scale)
    canvas_w = int(max(1, total_w + glow_radius * 2))
    canvas_h = int(max(1, total_h + glow_radius * 2))
    
    def hex_to_rgb_str(hex_val, alpha=1.0):
        hex_val = hex_val.lstrip('#')
        if len(hex_val) == 6:
            r, g, b = tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))
        else:
            r, g, b = 0, 0, 0
        return f"rgba({r}, {g}, {b}, {alpha})"
        
    def hex_to_darker_rgb_str(hex_val, factor=0.2, alpha=1.0):
        hex_val = hex_val.lstrip('#')
        if len(hex_val) == 6:
            r, g, b = tuple(int(int(hex_val[i:i+2], 16) * factor) for i in (0, 2, 4))
        else:
            r, g, b = 0, 0, 0
        return f"rgba({r}, {g}, {b}, {alpha})"

    # Start SVG document
    svg = f'''<svg width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}" xmlns="http://www.w3.org/2000/svg">
    <defs>
    '''
    
    # Define Drop Shadows / Glows
    glow_std_dev = int(6 * scale) if "1." in style else int(12 * scale)
    svg += f'''
        <filter id="neonGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="{glow_std_dev}" result="coloredBlur"/>
            <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="coloredBlur"/> 
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>
        <filter id="pillGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="{int(15 * scale)}" result="coloredBlur"/>
            <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>
        <filter id="textShadow">
            <feDropShadow dx="{max(1, int(3*scale))}" dy="{max(1, int(3*scale))}" stdDeviation="1" flood-color="rgba(0,0,0,{opacity})"/>
        </filter>
    '''

    current_y = glow_radius
    shapes_svg = ""
    texts_svg = ""
    glow_svg = ""
    
    for i, dim in enumerate(line_dims):
        if "Left" in header_position:
            x_offset = glow_radius
        elif "Right" in header_position:
            x_offset = glow_radius + (total_w - dim['bw'])
        else: # Center
            x_offset = glow_radius + (total_w - dim['bw']) // 2
            
        box_w, box_h = dim['bw'], dim['bh']
        
        if box_w <= 0 or box_h <= 0:
            continue
            
        # Radius Rules
        radius = int(20 * scale)
        if "3." in style: # Floating Pill
            radius = int(box_h / 2)
        elif "4." in style or "5." in style: # Split Grid / Double Stroke
            radius = int(8 * scale)
            
        # Define Gradient for this box
        grad_id = f"grad{i}"
        
        if "2." in style:
            stop1 = f"rgba(30, 30, 30, {0.7 * opacity})"
            stop2 = f"rgba(10, 10, 10, {0.4 * opacity})"
        elif "3." in style:
            stop1 = f"rgba(26, 26, 26, {opacity})"
            stop2 = f"rgba(26, 26, 26, {opacity})"
        elif "4." in style:
            stop1 = f"rgba(45, 45, 45, {opacity})"
            stop2 = f"rgba(20, 20, 20, {0.78 * opacity})"
        else:
            is_top_bar = (i == 0) and (len(line_dims) > 1)
            if is_top_bar:
                stop1 = hex_to_rgb_str(bg_color_hex, opacity)
                stop2 = hex_to_rgb_str(bg_color_hex, opacity)
            else:
                stop1 = hex_to_rgb_str(bg_color_hex, opacity)
                stop2 = hex_to_darker_rgb_str(bg_color_hex, 0.2, opacity)

        svg += f'''
        <linearGradient id="{grad_id}" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="{stop1}" />
            <stop offset="100%" stop-color="{stop2}" />
        </linearGradient>
        '''
        
        # Build Shapes
        border_color_full = hex_to_rgb_str(color_hex, 1.0)
        border_color_op = hex_to_rgb_str(color_hex, opacity)
        
        # Box Background Fill
        shapes_svg += f'<rect x="{x_offset}" y="{current_y}" width="{box_w}" height="{box_h}" rx="{radius}" ry="{radius}" fill="url(#{grad_id})" />\n'
        
        if "1." in style: # Neon Edge
            glow_svg += f'<rect x="{x_offset}" y="{current_y}" width="{box_w}" height="{box_h}" rx="{radius}" ry="{radius}" fill="none" stroke="{border_color_full}" stroke-width="{int(12 * scale)}" filter="url(#neonGlow)" />\n'
            shapes_svg += f'<rect x="{x_offset}" y="{current_y}" width="{box_w}" height="{box_h}" rx="{radius}" ry="{radius}" fill="none" stroke="{border_color_op}" stroke-width="{max(1, int(4 * scale))}" />\n'
            
        elif "2." in style: # Glassmorphic Ribbon
            shapes_svg += f'<rect x="{x_offset}" y="{current_y}" width="{box_w}" height="{box_h}" rx="{radius}" ry="{radius}" fill="none" stroke="rgba(255,255,255,{0.4*opacity})" stroke-width="{max(1, int(2*scale))}" />\n'
            shapes_svg += f'<rect x="{x_offset+2}" y="{current_y+2}" width="{box_w}" height="{box_h}" rx="{radius}" ry="{radius}" fill="none" stroke="{border_color_op}" stroke-width="{max(1, int(4*scale))}" />\n'
            
        elif "3." in style: # Floating Pill
            # Add custom gradient specifically for the border
            pill_stroke_grad_id = f"pillStrokeGrad{i}"
            # We transition from the selected hex color to a vibrant complementary offset, or simply a bright to slightly darker version. 
            # Let's make a beautiful transition using the base border color and a lighter version of it.
            stroke_stop1 = border_color_op
            
            # Extract hex to generate a bright vibrant secondary color (e.g. shift hue/lightness). 
            # We'll shift it towards a nice warmer/brighter tone. For simplicity, we just use a lighter alpha or different shade.
            # We can use the hex_to_rgb_str with a lowered alpha for a fade, or just a hardcoded pink/orange vibe.
            # Let's use a sunset-style gradient: Base Color -> White or Pinkish
            # We'll use a semi-transparent version of the main color to a whiteish glow.
            stroke_stop2 = f"rgba(255, 255, 255, {opacity})"
            
            svg += f'''
            <linearGradient id="{pill_stroke_grad_id}" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="{stroke_stop1}" />
                <stop offset="50%" stop-color="{stroke_stop2}" />
                <stop offset="100%" stop-color="{stroke_stop1}" />
            </linearGradient>
            '''
            
            glow_svg += f'<rect x="{x_offset}" y="{current_y}" width="{box_w}" height="{box_h}" rx="{radius}" ry="{radius}" fill="none" stroke="{border_color_full}" stroke-width="{int(18 * scale)}" filter="url(#pillGlow)" />\n'
            shapes_svg += f'<rect x="{x_offset}" y="{current_y}" width="{box_w}" height="{box_h}" rx="{radius}" ry="{radius}" fill="none" stroke="url(#{pill_stroke_grad_id})" stroke-width="{max(1, int(4 * scale))}" />\n'
            
        elif "4." in style: # Split Grid
            bar_w = int(14 * scale)
            shapes_svg += f'<rect x="{x_offset}" y="{current_y}" width="{bar_w}" height="{box_h}" rx="{max(1, int(radius/2))}" ry="{max(1, int(radius/2))}" fill="{border_color_op}" />\n'
            shapes_svg += f'<rect x="{x_offset}" y="{current_y}" width="{box_w}" height="{box_h}" rx="{radius}" ry="{radius}" fill="none" stroke="{border_color_op}" stroke-width="{max(1, int(2 * scale))}" />\n'
            
        elif "5." in style: # Double Stroke
            shapes_svg += f'<rect x="{x_offset}" y="{current_y}" width="{box_w}" height="{box_h}" rx="{radius}" ry="{radius}" fill="none" stroke="rgba(255,255,255,{0.8*opacity})" stroke-width="{max(1, int(1.5*scale))}" />\n'
            outer_gap = int(6 * scale)
            r_out = radius + max(1, int(outer_gap/2))
            shapes_svg += f'<rect x="{x_offset-outer_gap}" y="{current_y-outer_gap}" width="{box_w+(outer_gap*2)}" height="{box_h+(outer_gap*2)}" rx="{r_out}" ry="{r_out}" fill="none" stroke="{border_color_op}" stroke-width="{max(1, int(4*scale))}" />\n'
            
        # Text
        tx = x_offset + pad_x
        # Using exact center of box layout with dominant-baseline for SVG
        ty = current_y + (box_h / 2)
        if "4." in style: 
            tx += int(20 * scale)
            
        text_color = f"rgba(255, 255, 255, {opacity})"
        texts_svg += f'<text x="{tx}" y="{ty}" font-family="Arial, sans-serif" font-weight="bold" font-size="{font_size}px" fill="{text_color}" dominant-baseline="central" filter="url(#textShadow)">{dim["text"]}</text>\n'
        
        current_y += box_h + line_spacing

    svg += "</defs>\n"
    
    # Layering
    if "1." in style or "3." in style:
        svg += f"<g id='glowLayer'>{glow_svg}</g>\n"
        
    svg += f"<g id='shapesLayer'>{shapes_svg}</g>\n"
    svg += f"<g id='textsLayer'>{texts_svg}</g>\n"
    svg += "</svg>"

    try:
        png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        return Image.open(io.BytesIO(png_data))
    except Exception as e:
        print(f"CairoSVG Render Error: {e}")
        # Fallback to empty image on error
        return Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))

def process_video_pipeline(image_paths, audio_path, output_path, logo_path=None, logo_position="Bottom-Right", logo_opacity=0.8, header_text="", header_position="Freestyle", header_opacity=0.9, header_scale=1.0, header_color="#FF6E00", header_bg_color="#000000", header_style="1. Neon Edge", header_animation="None", video_bg_volume=0.15, progress_callback=None, status_callback=None):
    """
    Main function to process the images and audio into a final video.
    Supports a 4x3 grid / 5x5 grid logo, and automatic dynamic text headers generated with PIL.
    """
    import proglog
    
    class StreamlitProgressLogger(proglog.ProgressBarLogger):
        def __init__(self, ui_cb):
            super().__init__()
            self.ui_cb = ui_cb
            self.last_percentage = -1

        def bars_callback(self, bar, attr, value, old_value=None):
            if attr == 'index' and self.ui_cb:
                total = self.state.get('bars', {}).get(bar, {}).get('total', 1)
                # MoviePy handles rendering at "50% to 100%" of our overall app bar.
                # So we calculate rendering percentage (0-100) and scale it.
                percentage = int((value / total) * 100)
                
                # Only update UI when percentage genuinely changes to prevent flooding Streamlit
                if percentage != self.last_percentage:
                    self.last_percentage = percentage
                    # We map this 0-100 render progress to the 50-100 range of the main progress bar
                    overall_percentage = 50 + int(percentage * 0.5)
                    self.ui_cb(overall_percentage)
                    
    logger = StreamlitProgressLogger(progress_callback) if progress_callback else None
    target_w, target_h = 1080, 1920
    fps = 30
    crossfade_duration = 0.4
    clips = []
    audio_clip = None
    final_video = None
    
    try:
        num_files = len(image_paths)
        if num_files == 0:
            raise ValueError("No media files provided.")
            
        # Distinguish between images and videos
        image_list = []
        video_list = []
        video_extensions = ['.mp4', '.mov']
        
        for p in image_paths:
            ext = os.path.splitext(p)[1].lower()
            if ext in video_extensions:
                video_list.append(p)
            else:
                image_list.append(p)
                
        # Calculate durations
        total_video_duration = 0
        video_clips_raw = []
        from moviepy import VideoFileClip
        for vp in video_list:
            v_clip = VideoFileClip(vp)
            total_video_duration += v_clip.duration
            video_clips_raw.append(v_clip)
            
        if audio_path and os.path.exists(audio_path):
            if status_callback: status_callback("Loading Audio...")
            audio_clip = AudioFileClip(audio_path)
            total_audio_duration = audio_clip.duration
            
            remaining_audio_time = total_audio_duration - total_video_duration
            
            # If videos are longer than audio, we will have to trim them later.
            # For now, remaining audio time for images is capped at 0.
            if remaining_audio_time < 0:
                remaining_audio_time = 0
                
            num_images = len(image_list)
            if num_images > 0:
                slide_duration = (remaining_audio_time + (num_images - 1) * crossfade_duration) / num_images
                if slide_duration < 0.5: # Failsafe minimum duration
                    slide_duration = 0.5
            else:
                slide_duration = 0
        else:
            audio_clip = None
            total_audio_duration = 0
            slide_duration = 2.0  # Default duration for images without voiceover
            
        # Process Timeline
        # We need to maintain the original order from image_paths
        
        from moviepy import VideoFileClip
        
        for i, path in enumerate(image_paths):
            ext = os.path.splitext(path)[1].lower()
            if status_callback: status_callback(f"Processing Media {i+1}/{num_files}...")
            
            if ext in video_extensions:
                # Process strictly as Video
                # No YOLO, keep original aspect ratio, fit to 1080x1920, center, add blurred frame 1 bg
                
                v_clip = VideoFileClip(path)
                
                # Extract first frame for blur background
                first_frame = v_clip.get_frame(0)
                
                # Create blurred background
                bg_w, bg_h = target_w, target_h
                frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
                
                # Crop and resize frame to fill 9:16 canvas for bg
                orig_h, orig_w = frame_bgr.shape[:2]
                orig_aspect = orig_w / orig_h
                target_aspect = target_w / target_h
                
                if orig_aspect > target_aspect:
                    fit_h = target_h
                    fit_w = int(fit_h * orig_aspect)
                else:
                    fit_w = target_w
                    fit_h = int(fit_w / orig_aspect)
                    
                bg_img = cv2.resize(frame_bgr, (fit_w, fit_h), interpolation=cv2.INTER_LANCZOS4)
                
                bg_x1 = (fit_w - target_w) // 2
                bg_y1 = (fit_h - target_h) // 2
                bg_cropped = bg_img[bg_y1:bg_y1+target_h, bg_x1:bg_x1+target_w]
                
                # Fast Blur Hack: downscale 4x -> light blur -> upscale
                small_h, small_w = target_h // 4, target_w // 4
                bg_small = cv2.resize(bg_cropped, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                bg_small_blurred = cv2.GaussianBlur(bg_small, (25, 25), 0)
                bg_blurred = cv2.resize(bg_small_blurred, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                bg_blurred_rgb = cv2.cvtColor(bg_blurred, cv2.COLOR_BGR2RGB)
                
                bg_clip = ImageClip(bg_blurred_rgb).with_duration(v_clip.duration)
                
                # Resize the actual video clip to fit within 1080x1920 keeping aspect ratio
                try:
                    from moviepy.video.fx.Resize import Resize
                    # Scale down to fit width or height
                    if orig_aspect > target_aspect:
                        # Width is the constraint
                        v_clip_resized = v_clip.with_effects([Resize(width=target_w)])
                    else:
                        # Height is the constraint
                        v_clip_resized = v_clip.with_effects([Resize(height=target_h)])
                except Exception:
                    # Fallback if Resize FX is missing
                    v_clip_resized = v_clip
                    
                v_clip_centered = v_clip_resized.with_position("center")
                
                # Apply volume multiplier configuration
                if v_clip_centered.audio is not None:
                    v_clip_centered = v_clip_centered.with_volume_scaled(video_bg_volume)
                
                # Composite the video over the blurred background
                composite_v_clip = CompositeVideoClip([bg_clip, v_clip_centered]).with_duration(v_clip.duration)
                
                clip = composite_v_clip
                
            else:
                # Process strictly as Image via YOLO Ken Burns
                cx, cy = get_yolo_center(path)
                clip = create_ken_burns_clip(
                    path, target_w, target_h, slide_duration, cx, cy, 
                    scale_start=1.05, scale_end=1.15
                )
        
            if i > 0:
                clip = clip.with_effects([CrossFadeIn(crossfade_duration)])
        
            clips.append(clip)
            if progress_callback: progress_callback(int((i+1) / num_files * 50))
    
        if status_callback: status_callback("Concatenating clips...")
        final_video = concatenate_videoclips(clips, method="compose", padding=-crossfade_duration)
        
        if audio_clip is not None:
            # Trim final compiled video if it exceeds voiceover length
            if final_video.duration > total_audio_duration:
                final_video = final_video.subclipped(0, total_audio_duration)
                
            # Audio Mixing
            # Ensure the final compiled video keeps its tracks (bg video tracks) and composite with voiceover
            if final_video.audio is not None:
                from moviepy import CompositeAudioClip
                final_audio = CompositeAudioClip([final_video.audio, audio_clip]).with_duration(final_video.duration)
                final_video = final_video.with_audio(final_audio)
            else:
                final_video = final_video.with_audio(audio_clip.with_duration(final_video.duration))

        # We will collect all final overlay layers here to flatten the MoviePy compositing tree
        final_layers = [final_video]

        # --- WATERMARK LOGO INTEGRATION ---
        if logo_path and os.path.exists(logo_path):
            if status_callback: status_callback("Adding Watermark Logo...")
        
            # Multiply the alpha channel securely using OpenCV before letting MoviePy composite it
            logo_img_cv = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo_img_cv is not None and len(logo_img_cv.shape) == 3 and logo_img_cv.shape[2] == 4:
                # Decrease Alpha channel by opacity multiplier
                logo_img_cv[:, :, 3] = (logo_img_cv[:, :, 3] * logo_opacity).astype(np.uint8)
                # Convert BGRA to RGBA for MoviePy
                logo_img_cv = cv2.cvtColor(logo_img_cv, cv2.COLOR_BGRA2RGBA)
                logo_clip = ImageClip(logo_img_cv)
            else:
                logo_clip = ImageClip(logo_path)
        
            # Limit max width of logo to 25% of the screen horizontally
            max_logo_w = int(target_w * 0.25)
            lw, lh = logo_clip.size
        
            if lw > max_logo_w:
                # We use moviepy's resized method if available, or resize down gracefully
                try:
                    from moviepy.video.fx.Resize import Resize
                    logo_clip = logo_clip.with_effects([Resize(width=max_logo_w)])
                except Exception:
                    pass # If FX fails, we proceed with original size
                
            lw, lh = logo_clip.size
            padding = 40
        
            # Compute exact position based on XY coordinates from sliders
            if logo_position.startswith("XY:"):
                coords = logo_position.replace("XY:", "").split(",")
                x, y = int(coords[0]), int(coords[1])
            else:
                # Fallback
                x = target_w - lw - padding
                y = target_h - lh - padding
            
            # Position logo and make it last the entire video duration
            logo_clip = logo_clip.with_position((x, y)).with_duration(final_video.duration)
        
            # Queue it for single-pass compositing
            final_layers.append(logo_clip)
            clips.append(logo_clip) # Track to cleanup

        # --- DYNAMIC TEXT HEADER INTEGRATION ---
        if header_text.strip():
            if status_callback: status_callback("Generating Dynamic Header...")
            
            # 1. Generate Header Plate leveraging our new styling function
            header_img = generate_dynamic_header_img(
                header_text, header_scale, header_color, header_bg_color,
                header_opacity, header_style, header_position
            )
            
            canvas_w, canvas_h = header_img.size
            
            header_np = np.array(header_img)
            header_clip = ImageClip(header_np)
        
            # 2. Position Header
            # Decode position
            
            # box size used for grid logic AND animation
            box_w, box_h = canvas_w, canvas_h
            
            if header_position.startswith("XY:"):
                # Freestyle Canvas Position
                coords = header_position.replace("XY:", "").split(",")
                hx, hy = int(coords[0]), int(coords[1])
            else:
                # 3x3 Grid Mode
                hx, hy = 0, 0
                grid_margin = 60
            
                if "Left" in header_position:
                    if "Center-" in header_position: hx = int(target_w * 0.25) - (box_w // 2)
                    else: hx = grid_margin
                elif "Right" in header_position:
                    if "Center-" in header_position: hx = int(target_w * 0.75) - (box_w // 2)
                    else: hx = target_w - box_w - grid_margin
                else: # Center horizontally
                    hx = (target_w - box_w) // 2
                
                if "Top" in header_position: hy = grid_margin
                elif "Upper-Middle" in header_position: hy = int(target_h * 0.25) - (box_h // 2)
                elif "Center" in header_position and "Left" not in header_position and "Right" not in header_position: 
                    hy = int(target_h * 0.5) - (box_h // 2) # True center vertical
                elif "Lower-Middle" in header_position: hy = int(target_h * 0.75) - (box_h // 2)
                else: hy = target_h - box_h - grid_margin
            
            # Evaluate header animation
            if header_animation != "None":
                def pos_func(t):
                    return make_animated_position(
                        t, hx, hy, target_w, target_h, box_w, box_h, final_video.duration, header_animation
                    )
                header_clip = header_clip.with_position(pos_func).with_duration(final_video.duration)
            else:
                header_clip = header_clip.with_position((hx, hy)).with_duration(final_video.duration)
            
            final_layers.append(header_clip)
            clips.append(header_clip)
            
        # Composite all collected layers in a single pass
        if len(final_layers) > 1:
            final_video = CompositeVideoClip(final_layers)
    
        if status_callback: status_callback("Rendering Final Video (this may take a while)...")
        # Custom logger passes progress to Streamlit without freezing IO with terminal prints
        final_video.write_videofile(
            output_path, 
            fps=fps, 
            codec="libx264", 
            audio_codec="aac",
            preset="superfast",  # Balanced speed vs IO — better than ultrafast for disk-bound systems
            threads=os.cpu_count(),
            logger=logger
        )
        
    finally:
        # Cleanup clips unconditionally
        for clip in clips:
            try: clip.close()
            except: pass
        if audio_clip:
            try: audio_clip.close()
            except: pass
        if final_video:
            try: final_video.close()
            except: pass
