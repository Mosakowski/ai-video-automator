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
    OVERSAMPLE_FACTOR = 3
    huge_w = target_w * OVERSAMPLE_FACTOR
    huge_h = target_h * OVERSAMPLE_FACTOR

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

    bg_img = cv2.resize(img_bgr, (bg_w, bg_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Crop the overgrown background to exact huge target dimensions
    bg_x1 = (bg_w - huge_w) // 2
    bg_y1 = (bg_h - huge_h) // 2
    bg_cropped = bg_img[bg_y1:bg_y1+huge_h, bg_x1:bg_x1+huge_w]

    # Apply heavy Gaussian blur to the huge background
    bg_blurred = cv2.GaussianBlur(bg_cropped, (301, 301), 90)

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
        
        # Cast to int (since it's a 3x huge canvas, a 1 unit integer rounding error 
        # is 1/3rd of a pixel error in the final downscaled output, eliminating visible jitter)
        x1 = int(round(x1_f))
        y1 = int(round(y1_f))
        w_int = int(round(current_window_w))
        h_int = int(round(current_window_h))
        
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

def process_video_pipeline(image_paths, audio_path, output_path, logo_path=None, logo_position="Bottom-Right", logo_opacity=0.8, header_text="", header_position="Freestyle", header_opacity=0.9, header_scale=1.0, video_bg_volume=0.15, progress_callback=None, status_callback=None):
    """
    Main function to process the images and audio into a final video.
    Supports a 4x3 grid / 5x5 grid logo, and automatic dynamic text headers generated with PIL.
    """
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
                
                bg_blurred = cv2.GaussianBlur(bg_cropped, (151, 151), 50)
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
        
            # Composite it over the final video stream
            final_video = CompositeVideoClip([final_video, logo_clip])
            clips.append(logo_clip) # Track to cleanup

        # --- DYNAMIC TEXT HEADER INTEGRATION ---
        if header_text.strip():
            if status_callback: status_callback("Generating Dynamic Header...")
            from PIL import Image, ImageDraw, ImageFont, ImageFilter
        
            # 1. Generate Header Plate (TikTok Pill Style)
            lines = header_text.strip().split('\n')
        
            # For a 1080x1920 video, use larger font and paddings
            pad_x = int(40 * header_scale)
            pad_y = int(20 * header_scale)
            line_spacing = int(-10 * header_scale) # Negative spacing to bunch them up like in the reference
        
            try:
                # On Windows, arialbd.ttf is Arial Bold
                font = ImageFont.truetype("arialbd.ttf", int(75 * header_scale))
            except IOError:
                font = ImageFont.load_default()

            # Calculate dimensions
            dummy_draw = ImageDraw.Draw(Image.new('RGBA', (1,1)))
            line_dims = []
            total_w = 0
            total_h = 0
        
            for line in lines:
                try:
                    left, top, right, bottom = dummy_draw.textbbox((0,0), line, font=font)
                    tw = right - left
                    th = bottom - top
                except AttributeError:
                    tw = len(line) * 35 # rough approx
                    th = 75
                
                box_w = tw + pad_x * 2
                box_h = th + pad_y * 2
                line_dims.append({'text': line, 'tw': tw, 'th': th, 'bw': box_w, 'bh': box_h})
            
                if box_w > total_w: total_w = box_w
                total_h += box_h + line_spacing
            
            total_h -= line_spacing # remove last spacing
        
            # Add padding for glow effect
            glow_radius = int(25 * header_scale)
            canvas_w = total_w + glow_radius * 2
            canvas_h = total_h + glow_radius * 2
        
            header_img = Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))
            glow_layer = Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))
            shapes_layer = Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))
            text_layer = Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))
        
            glow_draw = ImageDraw.Draw(glow_layer)
            shapes_draw = ImageDraw.Draw(shapes_layer)
            text_draw = ImageDraw.Draw(text_layer)
        
            current_y = glow_radius
        
            for dim in line_dims:
                # Align boxes based on position string
                if "Left" in header_position:
                    x_offset = glow_radius
                elif "Right" in header_position:
                    x_offset = glow_radius + (total_w - dim['bw'])
                else: # Center
                    x_offset = glow_radius + (total_w - dim['bw']) // 2
            
                box_rect = [x_offset, current_y, x_offset + dim['bw'], current_y + dim['bh']]
            
                # Glow (Thicker outline drawn on blur layer)
                glow_draw.rounded_rectangle(box_rect, radius=int(20 * header_scale), outline=(255, 110, 0, 255), width=int(20 * header_scale))
            
                # Main Shape (Dark Charcoal Gradient)
                box_w = dim['bw']
                box_h = dim['bh']
                grad_img = Image.new('RGBA', (box_w, box_h))
                grad_draw = ImageDraw.Draw(grad_img)
                color_top = (55, 55, 55, int(255 * header_opacity))
                color_bottom = (15, 15, 15, int(255 * header_opacity))
                for y in range(box_h):
                    ratio = y / float(box_h)
                    r = int(color_top[0] * (1 - ratio) + color_bottom[0] * ratio)
                    g = int(color_top[1] * (1 - ratio) + color_bottom[1] * ratio)
                    b = int(color_top[2] * (1 - ratio) + color_bottom[2] * ratio)
                    a = int(color_top[3] * (1 - ratio) + color_bottom[3] * ratio)
                    grad_draw.line([(0, y), (box_w, y)], fill=(r, g, b, a))
                
                mask = Image.new('L', (box_w, box_h), 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rounded_rectangle([0, 0, box_w, box_h], radius=int(20 * header_scale), fill=255)
                
                grad_wrapper = Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))
                grad_wrapper.paste(grad_img, (x_offset, current_y), mask)
                shapes_layer.alpha_composite(grad_wrapper)
                
                shapes_draw = ImageDraw.Draw(shapes_layer)
                shapes_draw.rounded_rectangle(box_rect, radius=int(20 * header_scale), outline=(255, 140, 0, int(255 * header_opacity)), width=max(1, int(4 * header_scale)))
            
                # Text
                tx = x_offset + pad_x
                ty = current_y + pad_y - int(12 * header_scale) # vertical tweak for Arial bounds offset
            
                # Subtle drop shadow
                shadow_off = max(1, int(3 * header_scale))
                text_draw.text((tx+shadow_off, ty+shadow_off), dim['text'], fill=(0, 0, 0, int(200 * header_opacity)), font=font)
                # Main Text
                text_draw.text((tx, ty), dim['text'], fill=(255, 255, 255, int(255 * header_opacity)), font=font)
            
                current_y += dim['bh'] + line_spacing
            
            # Apply Blur to glow layer
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(glow_radius // 2))
        
            # Composite layers with double glow
            header_img.alpha_composite(glow_layer)
            header_img.alpha_composite(glow_layer) # Extra glow pass
            header_img.alpha_composite(shapes_layer)
            header_img.alpha_composite(text_layer)
            
            header_np = np.array(header_img)
            header_clip = ImageClip(header_np)
        
            # 2. Position Header
            # Decode position
            if header_position.startswith("XY:"):
                # Freestyle Canvas Position
                coords = header_position.replace("XY:", "").split(",")
                hx, hy = int(coords[0]), int(coords[1])
            else:
                # 3x3 Grid Mode
                hx, hy = 0, 0
                grid_margin = 60
            
                # box size used for grid logic
                box_w, box_h = canvas_w, canvas_h
            
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
            
            header_clip = header_clip.with_position((hx, hy)).with_duration(final_video.duration)
            final_video = CompositeVideoClip([final_video, header_clip])
            clips.append(header_clip)
    
        if status_callback: status_callback("Rendering Final Video (this may take a while)...")
        # For progress indication during rendering: MoviePy uses progress_bar=True internally, but to pipe it to stream we might need a custom logger.
        final_video.write_videofile(
            output_path, 
            fps=fps, 
            codec="libx264", 
            audio_codec="aac",
            preset="ultrafast",  # Use ultrafast for quicker generation in the app
            threads=4
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
