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
    /* ===== DARK STUDIO GLOBAL ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== STICKY RIGHT COLUMN ===== */
    [data-testid="column"]:nth-of-type(2) {
        position: sticky;
        top: 1rem;
        z-index: 10;
        align-self: flex-start;
        height: fit-content;
    }
    
    /* ===== PREVIEW IMAGE ===== */
    div[data-testid="stImage"] > img {
        border-radius: 12px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);
        border: 1px solid #444;
    }
    
    /* ===== SECTION HEADERS ===== */
    .section-header {
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #999;
        padding-bottom: 8px;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #333;
    }
    
    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.5px;
        color: #e0e0e0 !important;
    }
    
    /* ===== GENERATE BUTTON ===== */
    div.stButton > button[kind="primary"] {
        width: 100%;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(255, 110, 0, 0.3);
    }
    
    /* ===== PREVIEW CONTAINER ===== */
    .preview-container {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .preview-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #666;
        margin-bottom: 12px;
    }
    .preview-res {
        font-size: 0.65rem;
        color: #555;
        margin-top: 8px;
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

def generate_video(image_files, audio_file, uploaded_logo, logo_position, logo_opacity, header_text, header_position, header_opacity, header_scale, header_color, header_bg_color, header_style, header_animation, video_bg_volume, progress_bar, status_text):
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
        if isinstance(uploaded_logo, str):
            logo_path_str = uploaded_logo
        else:
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
        header_color=header_color,
        header_bg_color=header_bg_color,
        header_style=header_style,
        header_animation=header_animation,
        video_bg_volume=video_bg_volume,
        progress_callback=progress_callback,
        status_callback=status_callback
    )
    
    progress_bar.progress(100)
    status_text.text("Done!")

# Mockup function
def render_unified_mockup(logo_file, logo_pos, logo_alpha, head_text, head_pos, head_alpha, head_scale, head_color, head_bg_color, head_style):
    from PIL import Image, ImageDraw, ImageFont
    from PIL.Image import Resampling
    
    # Base canvas (540x960)
    W, H = 540, 960
    mockup = Image.new("RGBA", (W, H), (40, 40, 40, 255))
    
    # 1. Add Watermark Logo
    max_logo_w = int(W * 0.25)
    
    if logo_file:
        try:
            if isinstance(logo_file, str):
                logo_img = Image.open(logo_file).convert("RGBA")
            else:
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
    
    # 2. Add Dynamic Header Box
    if head_text.strip():
        from video_engine import generate_dynamic_header_img
        
        # Scale for mockup is exactly 50% of the target 1080p scale.
        mockup_scale = head_scale * 0.5
        header_img = generate_dynamic_header_img(
            head_text, mockup_scale, head_color, head_bg_color, head_alpha, head_style, head_pos
        )
        
        # Position Header
        canvas_w, canvas_h = header_img.size
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
st.markdown("<h2 style='margin-bottom: 0.2rem;'>AI Video Automator</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #666; font-size: 0.85rem; margin-top: 0;'>Automated 9:16 video generation pipeline</p>", unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.markdown("<div class='section-header'>Upload Assets</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='section-header'>Media Timeline Order</div>", unsafe_allow_html=True)
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
    
    st.markdown("<div class='section-header'>Configuration</div>", unsafe_allow_html=True)

    with st.expander("🎞️ Video Options", expanded=True):
        video_bg_volume = st.slider("Background Video Volume", min_value=0.0, max_value=1.0, value=0.15, step=0.05)

    with st.expander("💧 Watermark Logo", expanded=False):
        watermark_option = st.radio(
            "Watermark Source",
            ["None", "Ciekawostki", "Info24", "Custom Upload"],
            horizontal=True
        )
        
        uploaded_logo = None
        if watermark_option == "Ciekawostki":
            uploaded_logo = str(Path("assets/watermarks/watermark_ciekawostki.png").absolute())
        elif watermark_option == "Info24":
            uploaded_logo = str(Path("assets/watermarks/watermark_info24.png").absolute())
        elif watermark_option == "Custom Upload":
            uploaded_logo = st.file_uploader(
                "Upload Logo (PNG)", 
                type=["png"], 
                accept_multiple_files=False
            )

        st.write("Position Coordinates (X, Y):")
        col_lg_x, col_lg_y = st.columns(2)
        with col_lg_x:
            logo_x = st.slider("X (px)", 0, 1080, 700, key="lx")
        with col_lg_y:
            logo_y = st.slider("Y (px)", 0, 1920, 300, key="ly")
            
        logo_position = f"XY:{logo_x},{logo_y}"
        logo_opacity = st.slider("Logo Opacity", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

    with st.expander("💬 Dynamic Header", expanded=True):
        col_hd_style, col_hd_anim = st.columns(2)
        with col_hd_style:
            header_style = st.selectbox("Design Style", [
                "1. Neon Edge", 
                "2. Glassmorphic Ribbon", 
                "3. The Floating Pill", 
                "4. Split-Grid Panel", 
                "5. Double Stroke Outline"
            ], index=0)
        with col_hd_anim:
            header_animation = st.selectbox("Intro/Outro Animation", [
                "None",
                "1. Slide-in (Side)",
                "2. Pop-up (Bottom)"
            ], index=1)
        
        header_text = st.text_area("Header Text (Enter = Newline)", value="IRÁNSKI ATAK RAKIETOWY\\nNA DUBAJ", key="header_text_area")
        
        col_hd_colors1, col_hd_colors2 = st.columns(2)
        with col_hd_colors1:
            header_color = st.color_picker("Border Color", value="#FF6E00")
        with col_hd_colors2:
            header_bg_color = st.color_picker("Background Color", value="#000000")
        
        col_hd_op, col_hd_sc = st.columns(2)
        with col_hd_op:
            header_opacity = st.slider("Opacity", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
        with col_hd_sc:
            header_scale = st.slider("Scale (Size)", min_value=0.5, max_value=2.0, value=0.7, step=0.1, key="header_scale_slider")

        st.write("Position Coordinates (X, Y):")
        col_hd_x, col_hd_center, col_hd_y = st.columns([2, 1, 2])
        with col_hd_x:
            header_x = st.slider("X (px) ", 0, 1080, 65, key="hx")
        with col_hd_center:
            st.write("") # spacer for alignment
            def _center_x():
                # Measure the header width to compute true center
                from PIL import Image, ImageDraw, ImageFont
                _scale = st.session_state.get("header_scale_slider", 0.7)
                _text = st.session_state.get("header_text_area", "")
                _font_size = int(75 * _scale)
                _pad_x = int(40 * _scale)
                try:
                    _font = ImageFont.truetype("arialbd.ttf", _font_size)
                except IOError:
                    _font = ImageFont.load_default()
                _dd = ImageDraw.Draw(Image.new('RGBA', (1,1)))
                _max_w = 0
                for _line in _text.strip().split('\n'):
                    if _line.strip():
                        try:
                            l, t, r, b = _dd.textbbox((0,0), _line, font=_font)
                            _max_w = max(_max_w, r - l)
                        except:
                            _max_w = max(_max_w, len(_line) * int(35 * _scale))
                _box_w = _max_w + _pad_x * 2 + int(50 * _scale) * 2  # include glow_radius
                st.session_state["hx"] = max(0, (1080 - _box_w) // 2)
            st.button("Center X", key="center_x", on_click=_center_x)
        with col_hd_y:
            header_y = st.slider("Y (px) ", 0, 1920, 1300, key="hy")
        final_header_position = f"XY:{header_x},{header_y}"

    st.markdown("<div class='section-header'>Action</div>", unsafe_allow_html=True)
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
                    header_scale, header_color, header_bg_color, header_style, header_animation, video_bg_volume, progress_bar, status_text
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
    preview_image = render_unified_mockup(
        uploaded_logo, logo_position, logo_opacity, 
        header_text, final_header_position, header_opacity, header_scale, header_color, header_bg_color, header_style
    )
    
    st.markdown("<div class='preview-container'>", unsafe_allow_html=True)
    st.markdown("<div class='preview-label'>Live Preview</div>", unsafe_allow_html=True)
    st.image(preview_image, use_column_width=True)
    st.markdown("<div class='preview-res'>1080 × 1920 · 9:16</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

