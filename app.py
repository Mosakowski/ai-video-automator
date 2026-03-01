import streamlit as st
import os
import shutil
from pathlib import Path
from streamlit_sortables import sort_items

# --- Configuration & Setup ---
st.set_page_config(page_title="AI Video Automator", page_icon="🎥", layout="wide")

# Inject Custom CSS for Dark Studio Aesthetic & Sticky Right Column
st.markdown("""
    <style>
    /* Dark Studio Global Aesthetics */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Hide Streamlit default menus */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Make the second column sticky so it stays visible while scrolling the left */
    [data-testid="column"]:nth-of-type(2) {
        position: sticky;
        top: 3rem;
        z-index: 10;
        align-self: flex-start; /* Required for sticky inside flexbox */
    }
    
    /* Styled Preview Box */
    div[data-testid="stImage"] > img {
        border-radius: 10px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        border: 1px solid #333;
    }
    
    /* Sleek Expanders */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px;
        color: #f0f0f0 !important;
    }
    
    </style>
""", unsafe_allow_html=True)

TEMP_DIR = Path("temp")
OUTPUT_FILE = "output.mp4"

def cleanup_temp_dir():
    """Removes the temp directory and its contents gracefully."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

def generate_video(image_files, audio_file, uploaded_logo, logo_position, logo_opacity, header_text, header_position, header_opacity, header_scale, video_bg_volume, progress_bar, status_text):
    from video_engine import process_video_pipeline
    
    # 1. Save uploaded files to temp
    status_text.text("Saving uploaded files...")
    progress_bar.progress(5)
    
    img_paths = []
    for idx, img in enumerate(image_files):
        img_path = TEMP_DIR / f"img_{idx}_{img.name}"
        with open(img_path, "wb") as f:
            f.write(img.getbuffer())
        img_paths.append(str(img_path))
        
    audio_path_str = None
    if audio_file:
        audio_path = TEMP_DIR / f"audio_{audio_file.name}"
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        audio_path_str = str(audio_path)
        
    logo_path_str = None
    if uploaded_logo:
        logo_path = TEMP_DIR / f"logo_{uploaded_logo.name}"
        with open(logo_path, "wb") as f:
            f.write(uploaded_logo.getbuffer())
        logo_path_str = str(logo_path)
        
    def progress_callback(percentage):
        # Maps 0-100 of process to 10-90 of total bar
        progress_bar.progress(10 + int(percentage * 0.8))
        
    def status_callback(msg):
        status_text.text(msg)
        
    # 2. Run Pipeline
    process_video_pipeline(
        img_paths, 
        audio_path_str, 
        OUTPUT_FILE,
        logo_path=logo_path_str,
        logo_position=logo_position,
        logo_opacity=logo_opacity,
        header_text=header_text,
        header_position=header_position,
        header_opacity=header_opacity,
        header_scale=header_scale,
        video_bg_volume=video_bg_volume,
        progress_callback=progress_callback,
        status_callback=status_callback
    )
    
    progress_bar.progress(100)
    status_text.text("Done!")

# Mockup function
def render_unified_mockup(logo_file, logo_pos, logo_alpha, head_text, head_pos, head_alpha, head_scale):
    from PIL import Image, ImageDraw, ImageFont
    from PIL.Image import Resampling
    
    # Base canvas (540x960)
    W, H = 540, 960
    mockup = Image.new("RGBA", (W, H), (40, 40, 40, 255))
    
    # 1. Add Watermark Logo
    max_logo_w = int(W * 0.25)
    
    if logo_file:
        try:
            logo_file.seek(0)
            logo_img = Image.open(logo_file).convert("RGBA")
            logo_file.seek(0)
        except Exception:
            logo_img = Image.new("RGBA", (135, 60), (255, 0, 0, 150))
    else:
        # Placeholder
        logo_img = Image.new("RGBA", (135, 60), (255, 0, 0, 150))
        draw = ImageDraw.Draw(logo_img)
        draw.text((25, 20), "LOGO", fill=(255,255,255,255))
        
    lw, lh = logo_img.size
    if lw > max_logo_w:
        ratio = max_logo_w / lw
        new_h = int(lh * ratio)
        logo_img = logo_img.resize((max_logo_w, new_h), Resampling.LANCZOS)
        lw, lh = logo_img.size
        
    if logo_alpha < 1.0:
        alpha = logo_img.split()[3]
        alpha = alpha.point(lambda p: int(p * logo_alpha))
        logo_img.putalpha(alpha)
        
    # Standard logo padding
    padding = int(40 * (W / 1080))
    
    x_logo, y_logo = 0, 0
    if logo_pos.startswith("XY:"):
        coords = logo_pos.replace("XY:", "").split(",")
        x_logo = int(int(coords[0]) * (W / 1080))
        y_logo = int(int(coords[1]) * (H / 1920))
    else:
        if "Left" in logo_pos: x_logo = padding
        elif "Right" in logo_pos: x_logo = W - lw - padding
        else: x_logo = (W - lw) // 2
            
        if "Top" in logo_pos: y_logo = padding
        elif "Upper-Middle" in logo_pos: y_logo = int(H * 0.33) - (lh // 2)
        elif "Lower-Middle" in logo_pos: y_logo = int(H * 0.66) - (lh // 2)
        else: y_logo = H - lh - padding
        
    mockup.alpha_composite(logo_img, (x_logo, y_logo))
    
    # 2. Add Dynamic Header Box (TikTok Pill Style - Scaled 50% for 540x960 Mockup)
    if head_text.strip():
        lines = head_text.strip().split('\n')
        
        pad_x = int(20 * head_scale)
        pad_y = int(10 * head_scale)
        line_spacing = int(-5 * head_scale)
        
        try:
            font = ImageFont.truetype("arialbd.ttf", int(37 * head_scale))
        except IOError:
            font = ImageFont.load_default()
            
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
                tw = len(line) * 17
                th = 37
                
            box_w = tw + pad_x * 2
            box_h = th + pad_y * 2
            line_dims.append({'text': line, 'tw': tw, 'th': th, 'bw': box_w, 'bh': box_h})
            
            if box_w > total_w: total_w = box_w
            total_h += box_h + line_spacing
            
        total_h -= line_spacing
        
        glow_radius = int(12 * head_scale)
        canvas_w = total_w + glow_radius * 2
        canvas_h = total_h + glow_radius * 2
        
        header_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        glow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        shapes_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        text_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        
        glow_draw = ImageDraw.Draw(glow_layer)
        shapes_draw = ImageDraw.Draw(shapes_layer)
        text_draw = ImageDraw.Draw(text_layer)
        
        current_y = glow_radius
        
        for dim in line_dims:
            if "Left" in head_pos:
                x_offset = glow_radius
            elif "Right" in head_pos:
                x_offset = glow_radius + (total_w - dim['bw'])
            else:
                x_offset = glow_radius + (total_w - dim['bw']) // 2
                
            box_rect = [x_offset, current_y, x_offset + dim['bw'], current_y + dim['bh']]
            
            glow_draw.rounded_rectangle(box_rect, radius=int(10 * head_scale), outline=(255, 110, 0, 255), width=int(10 * head_scale))
            
            box_w = dim['bw']
            box_h = dim['bh']
            grad_img = Image.new('RGBA', (box_w, box_h))
            grad_draw = ImageDraw.Draw(grad_img)
            color_top = (55, 55, 55, int(255 * head_alpha))
            color_bottom = (15, 15, 15, int(255 * head_alpha))
            for y in range(box_h):
                ratio = y / float(box_h)
                r = int(color_top[0] * (1 - ratio) + color_bottom[0] * ratio)
                g = int(color_top[1] * (1 - ratio) + color_bottom[1] * ratio)
                b = int(color_top[2] * (1 - ratio) + color_bottom[2] * ratio)
                a = int(color_top[3] * (1 - ratio) + color_bottom[3] * ratio)
                grad_draw.line([(0, y), (box_w, y)], fill=(r, g, b, a))
                
            mask = Image.new('L', (box_w, box_h), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rounded_rectangle([0, 0, box_w, box_h], radius=int(10 * head_scale), fill=255)
            
            grad_wrapper = Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))
            grad_wrapper.paste(grad_img, (x_offset, current_y), mask)
            shapes_layer.alpha_composite(grad_wrapper)
            
            shapes_draw = ImageDraw.Draw(shapes_layer)
            shapes_draw.rounded_rectangle(box_rect, radius=int(10 * head_scale), outline=(255, 140, 0, int(255 * head_alpha)), width=max(1, int(2 * head_scale)))
            
            tx = x_offset + pad_x
            ty = current_y + pad_y - 6
            
            shadow_off = max(1, int(2 * head_scale))
            text_draw.text((tx+shadow_off, ty+shadow_off), dim['text'], fill=(0, 0, 0, int(200 * head_alpha)), font=font)
            text_draw.text((tx, ty), dim['text'], fill=(255, 255, 255, int(255 * head_alpha)), font=font)
            
            current_y += dim['bh'] + line_spacing
            
        from PIL import ImageFilter
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(glow_radius // 2))
        
        header_img.alpha_composite(glow_layer)
        header_img.alpha_composite(glow_layer) # Double composite for brighter core glow
        header_img.alpha_composite(shapes_layer)
        header_img.alpha_composite(text_layer)
        
        # Position Header
        box_w, box_h = canvas_w, canvas_h
        hx, hy = 0, 0
        if head_pos.startswith("XY:"):
            # Freestyle Coordinates from Sliders (Scaled back to 540x960)
            coords = head_pos.replace("XY:", "").split(",")
            hx = int(int(coords[0]) * (W / 1080))
            hy = int(int(coords[1]) * (H / 1920))
        else:
            # 5x5 Grid math (Scaled from engine)
            grid_margin = 30 # 60 * 0.5
            
            if "Left" in head_pos:
                if "Center-" in head_pos: hx = int(W * 0.25) - (box_w // 2)
                else: hx = grid_margin
            elif "Right" in head_pos:
                if "Center-" in head_pos: hx = int(W * 0.75) - (box_w // 2)
                else: hx = W - box_w - grid_margin
            else:
                hx = (W - box_w) // 2
                
            if "Top" in head_pos: hy = grid_margin
            elif "Upper-Middle" in head_pos: hy = int(H * 0.25) - (box_h // 2)
            elif "Center" in head_pos and "Left" not in head_pos and "Right" not in head_pos: hy = int(H * 0.5) - (box_h // 2)
            elif "Lower-Middle" in head_pos: hy = int(H * 0.75) - (box_h // 2)
            else: hy = H - box_h - grid_margin
            
        mockup.alpha_composite(header_img, (hx, hy))

    return mockup.convert("RGB")

# --- UI Layout Architecture ---
st.title("AI Video Automator")

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.markdown("### 🗂️ Upload Assets")
    uploaded_images = st.file_uploader(
        "Upload Media (Images & Videos)", 
        type=["jpg", "jpeg", "png", "mp4", "mov"], 
        accept_multiple_files=True
    )

    uploaded_audio = st.file_uploader(
        "Upload Voiceover (Optional, MP3)", 
        type=["mp3"], 
        accept_multiple_files=False
    )
    
    # --- Media Reordering ---
    st.markdown("### 🔄 Media Timeline Order")
    ordered_images = []
    if uploaded_images:
        # Create a mapping of filename -> UploadedFile object
        file_map = {img.name: img for img in uploaded_images}
        
        # Determine current order or fall back to original upload order
        original_names = list(file_map.keys())
        
        # Display the draggable sortable list
        st.write("Drag and drop to rearrange the order in which media will appear in the video:")
        sorted_names = sort_items(original_names)
        
        # If sort_items returns something, build the sorted list. Otherwise fallback to original.
        if sorted_names:
            ordered_images = [file_map[name] for name in sorted_names]
        else:
            ordered_images = uploaded_images
    else:
        st.info("Upload media above to set their order.")
    
    st.markdown("### ⚙️ Configuration")

    with st.expander("🎞️ Video Options", expanded=True):
        video_bg_volume = st.slider("Background Video Volume", min_value=0.0, max_value=1.0, value=0.15, step=0.05)

    with st.expander("💧 Watermark Logo", expanded=False):
        uploaded_logo = st.file_uploader(
            "Upload Logo (PNG)", 
            type=["png"], 
            accept_multiple_files=False
        )
        st.write("Position Coordinates (X, Y):")
        col_lg_x, col_lg_y = st.columns(2)
        with col_lg_x:
            logo_x = st.slider("X (px)", 0, 1080, 800, key="lx")
        with col_lg_y:
            logo_y = st.slider("Y (px)", 0, 1920, 1600, key="ly")
            
        logo_position = f"XY:{logo_x},{logo_y}"
        logo_opacity = st.slider("Logo Opacity", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

    with st.expander("💬 Dynamic Header", expanded=True):
        header_text = st.text_area("Header Text (Enter = Newline)", value="IRÁNSKI ATAK RAKIETOWY\\nNA DUBAJ")
        
        col_hd_op, col_hd_sc = st.columns(2)
        with col_hd_op:
            header_opacity = st.slider("Opacity", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
        with col_hd_sc:
            header_scale = st.slider("Scale (Size)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

        st.write("Position Coordinates (X, Y):")
        col_hd_x, col_hd_y = st.columns(2)
        with col_hd_x:
            header_x = st.slider("X (px) ", 0, 1080, 108, key="hx")
        with col_hd_y:
            header_y = st.slider("Y (px) ", 0, 1920, 200, key="hy")
        final_header_position = f"XY:{header_x},{header_y}"

    st.markdown("### 🎬 Action")
    if st.button("Generate Video", type="primary"):
        if not ordered_images:
            st.error("Please upload at least one image or video to continue.")
        else:
            cleanup_temp_dir()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                generate_video(
                    ordered_images, uploaded_audio, uploaded_logo, logo_position,  
                    logo_opacity, header_text, final_header_position, header_opacity, 
                    header_scale, video_bg_volume, progress_bar, status_text
                )
                
                st.success("VIDEO GENERATED SUCCESSFULLY")
                
                if Path(OUTPUT_FILE).exists():
                    with open(OUTPUT_FILE, "rb") as file:
                        btn = st.download_button(
                            label="Download Render",
                            data=file,
                            file_name="rendered_video.mp4",
                            mime="video/mp4"
                        )
            except Exception as e:
                st.error(f"Generation error: {e}")
            finally:
                cleanup_temp_dir()

# This column is targeted by the CSS above to be sticky on the right
with col_right:
    st.markdown("<h4 style='text-align: center; color: #888;'>LIVE PREVIEW</h4>", unsafe_allow_html=True)
    
    preview_image = render_unified_mockup(
        uploaded_logo, logo_position, logo_opacity, 
        header_text, final_header_position, header_opacity, header_scale
    )
    st.image(preview_image, use_column_width=True)

