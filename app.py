import streamlit as st
import os
import shutil
from pathlib import Path
from streamlit_sortables import sort_items
from positioning_resolver import resolve_header_top_left
import base64
from io import BytesIO
from PIL import Image
import cv2

try:
    import pillow_avif
except ImportError:
    pass

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
    
    /* ===== TIMELINE CARDS ===== */
    .media-card {
        background: #262626;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .media-thumb {
        width: 60px;
        height: 60px;
        border-radius: 4px;
        object-fit: cover;
        background: #111;
    }
    .media-info {
        flex-grow: 1;
    }
    .media-name {
        font-size: 0.85rem;
        font-weight: 500;
        color: #eee;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 200px;
    }
    .media-type {
        font-size: 0.7rem;
        color: #777;
        text-transform: uppercase;
    }
    
    /* ===== SORTABLE LIST CUSTOMIZATION ===== */
    /* Target the sortable list items if possible, otherwise we style the container */
    .sortable-container {
        max-width: 400px;
        margin-bottom: 2rem;
    }
    
    /* ===== HORIZONTAL FILMSTRIP ===== */
    .filmstrip-container {
        display: flex;
        overflow-x: auto;
        gap: 12px;
        padding: 10px 0;
        scroll-behavior: smooth;
        scrollbar-width: thin;
        scrollbar-color: #444 #1a1a1a;
    }
    .filmstrip-item {
        flex: 0 0 120px;
        background: #222;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 6px;
        text-align: center;
        transition: all 0.2s;
        position: relative;
    }
    .filmstrip-item:hover {
        border-color: #FF6E00;
        background: #2a2a2a;
    }
    .filmstrip-thumb {
        width: 100%;
        height: 160px;
        object-fit: cover;
        border-radius: 5px;
        margin-bottom: 6px;
    }
    .filmstrip-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: #bbb;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .filmstrip-index {
        position: absolute;
        top: -6px;
        left: -6px;
        background: #FF6E00;
        color: white;
        font-size: 0.65rem;
        font-weight: 900;
        width: 22px;
        height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        z-index: 10;
        box-shadow: 0 2px 5px rgba(0,0,0,0.5);
    }
    
    </style>
""", unsafe_allow_html=True)

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = "output.mp4"

def cleanup_temp_dir():
    """Removes the temp directory and its contents gracefully."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

def generate_video(image_files, audio_file, uploaded_logo, logo_position, logo_opacity, logo_scale, header_text, header_position, header_opacity, header_scale, header_color, header_bg_color, header_style, header_animation, video_bg_volume, header_custom_svg, header_font, progress_bar, status_text):
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
        logo_scale=logo_scale,
        header_text=header_text,
        header_position=header_position,
        header_opacity=header_opacity,
        header_scale=header_scale,
        header_color=header_color,
        header_bg_color=header_bg_color,
        header_style=header_style,
        header_animation=header_animation,
        video_bg_volume=video_bg_volume,
        header_custom_svg=header_custom_svg,
        header_font=header_font,
        progress_callback=progress_callback,
        status_callback=status_callback
    )
    
    progress_bar.progress(100)
    status_text.text("Done!")

def get_thumbnail(file):
    """Generates a base64 encoded thumbnail for an image or video."""
    if file.name in st.session_state.get('thumbnails', {}):
        return st.session_state['thumbnails'][file.name]
    
    try:
        if file.type.startswith('image'):
            img = Image.open(file)
            img.thumbnail((200, 200))
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            encoded = base64.b64encode(buffered.getvalue()).decode()
            thumb = f"data:image/jpeg;base64,{encoded}"
        elif file.type.startswith('video'):
            # Save temp file to read with cv2
            t_path = TEMP_DIR / f"thumb_tmp_{file.name}"
            with open(t_path, "wb") as f:
                f.write(file.getvalue())
            
            cap = cv2.VideoCapture(str(t_path))
            ret, frame = cap.read()
            cap.release()
            os.remove(t_path)
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img.thumbnail((200, 200))
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                encoded = base64.b64encode(buffered.getvalue()).decode()
                thumb = f"data:image/jpeg;base64,{encoded}"
            else:
                thumb = None
        else:
            thumb = None
    except Exception as e:
        print(f"Error generating thumbnail for {file.name}: {e}")
        thumb = None
    
    if 'thumbnails' not in st.session_state:
        st.session_state['thumbnails'] = {}
    st.session_state['thumbnails'][file.name] = thumb
    return thumb

# Mockup function
def render_unified_mockup(logo_file, logo_pos, logo_alpha, logo_scale, head_text, head_pos, head_alpha, head_scale, head_color, head_bg_color, head_style, head_custom_svg, head_font):
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
        
    if logo_scale != 1.0:
        new_w = int(lw * logo_scale)
        new_h = int(lh * logo_scale)
        if new_w > 0 and new_h > 0:
            logo_img = logo_img.resize((new_w, new_h), Resampling.LANCZOS)
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
        header_img, header_meta = generate_dynamic_header_img(
            head_text, mockup_scale, head_color, head_bg_color, head_alpha, head_style, head_pos, head_custom_svg, head_font,
            return_meta=True
        )
        
        # Position Header
        canvas_w, canvas_h = header_meta["canvas_size"]
        visual_bbox = header_meta["visual_bbox"]
        hx, hy = resolve_header_top_left(
            position_spec=head_pos,
            target_size=(W, H),
            canvas_size=(canvas_w, canvas_h),
            visual_bbox=visual_bbox,
            grid_margin=30,
            xy_scale=(W / 1080.0, H / 1920.0),
        )
            
        # Use paste with itself as mask for alpha compositing (compatible with different sizes)
        mockup.paste(header_img, (hx, hy), header_img)

    return mockup.convert("RGB")

# --- UI Layout Architecture ---
st.markdown("<h2 style='margin-bottom: 0.2rem;'>AI Video Automator</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #666; font-size: 0.85rem; margin-top: 0;'>Automated 9:16 video generation pipeline</p>", unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.markdown("<div class='section-header'>Upload Assets</div>", unsafe_allow_html=True)
    uploaded_images = st.file_uploader(
        "Upload Media (Images & Videos)", 
        type=["jpg", "jpeg", "png", "webp", "avif", "mp4", "mov"], 
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
        
        # Shorten names for the sortable list items
        def shorten_name(name, length=20):
            if len(name) <= length: return name
            ext = name.split('.')[-1]
            return name[:length-5] + "..." + ext

        # Prepare items for sort_items
        # We'll use "Index + Short Name" for the sortable bars to keep them clean
        original_labels = [f"{idx+1}. {shorten_name(img.name)}" for idx, img in enumerate(uploaded_images)]
        # Map labels back to full names
        label_to_fullname = {f"{idx+1}. {shorten_name(img.name)}": img.name for idx, img in enumerate(uploaded_images)}
        
        st.write("Rearrange the clips:")
        
        with st.container():
            st.markdown("<div class='sortable-container'>", unsafe_allow_html=True)
            sorted_labels = sort_items(original_labels, direction="vertical")
            st.markdown("</div>", unsafe_allow_html=True)
        
        if sorted_labels:
            ordered_images = [file_map[label_to_fullname[label]] for label in sorted_labels]
        else:
            ordered_images = uploaded_images
            
        # Display Visual Timeline (Filmstrip)
        st.markdown("### 🎞️ Visual Timeline")
        
        # IMPORTANT: Remove leading spaces in the HTML string to prevent markdown from treating it as a code block
        filmstrip_html = "<div class='filmstrip-container'>"
        for idx, img in enumerate(ordered_images):
            thumb = get_thumbnail(img)
            name = shorten_name(img.name, 15)
            
            thumb_tag = f"<img src='{thumb}' class='filmstrip-thumb'>" if thumb else "<div class='filmstrip-thumb' style='display:flex;align-items:center;justify-content:center;background:#111;font-size:0.5rem;'>NO THUMB</div>"
            
            # Construct HTML without leading spaces for each line
            item_html = "<div class='filmstrip-item'>"
            item_html += f"<div class='filmstrip-index'>{idx+1}</div>"
            item_html += thumb_tag
            item_html += f"<div class='filmstrip-label'>{name}</div>"
            item_html += "</div>"
            filmstrip_html += item_html
            
        filmstrip_html += "</div>"
        st.markdown(filmstrip_html, unsafe_allow_html=True)
        
        st.caption("Dragging items above updates this timeline automatically.")
    else:
        st.info("Upload media above to set their order.")
    
    st.markdown("<div class='section-header'>Configuration</div>", unsafe_allow_html=True)

    with st.expander("🎞️ Video Options", expanded=True):
        video_bg_volume = st.slider("Background Video Volume", min_value=0.0, max_value=1.0, value=0.15, step=0.05)

    with st.expander("💧 Watermark Logo", expanded=False):
        watermark_option = st.radio(
            "Watermark Source",
            ["None", "Ciekawostki", "Info24", "FactReactor", "Custom Upload"],
            horizontal=True
        )
        
        uploaded_logo = None
        if watermark_option == "Ciekawostki":
            uploaded_logo = str(Path("assets/watermarks/watermark_ciekawostki.png").absolute())
        elif watermark_option == "Info24":
            uploaded_logo = str(Path("assets/watermarks/watermark_info24.png").absolute())
        elif watermark_option == "FactReactor":
            uploaded_logo = str(Path("assets/watermarks/watermark_factreactor.png").absolute())
        elif watermark_option == "Custom Upload":
            uploaded_logo = st.file_uploader(
                "Upload Logo (PNG/WEBP/AVIF)", 
                type=["png", "webp", "avif"], 
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
        logo_scale = st.slider("Logo Scale (Size)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

    with st.expander("💬 Dynamic Header", expanded=True):
        col_hd_style, col_hd_anim = st.columns(2)
        with col_hd_style:
            header_style = st.selectbox("Design Style", [
                "1. Neon Edge", 
                "2. Glassmorphic Ribbon", 
                "3. The Floating Pill", 
                "4. Single News Banner", 
                "5. Multi-line News Banner",
                "6. Custom SVG",
                "7. Sharp Italic",
                "8. Warning Arch"
            ], index=0)
        
        col_hd_font, col_hd_dummy = st.columns(2)
        with col_hd_font:
            header_font = st.selectbox("Font Family", ["Arial", "Impact", "Verdana", "Georgia", "Courier New", "Tahoma"], index=1) # Default Impact
        
        header_custom_svg = ""
        if header_style == "6. Custom SVG":
            header_custom_svg = st.text_area("Custom SVG Code", placeholder="Paste SVG elements here (no <svg> tag needed)...", height=150)
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

        position_mode = st.radio(
            "Header Position Mode",
            ["Grid", "Custom (XY)"],
            horizontal=True,
            key="header_position_mode"
        )

        grid_positions = [
            "Top-Left", "Top-Center", "Top-Right",
            "Upper-Middle-Left", "Upper-Middle-Center", "Upper-Middle-Right",
            "Center-Left", "Center", "Center-Right",
            "Lower-Middle-Left", "Lower-Middle-Center", "Lower-Middle-Right",
            "Bottom-Left", "Bottom-Center", "Bottom-Right",
        ]

        if position_mode == "Grid":
            selected_grid = st.selectbox("Grid Position", grid_positions, index=7)
            final_header_position = f"GRID:{selected_grid}"
        else:
            st.write("Position Coordinates (X, Y):")
            col_hd_x, col_hd_y = st.columns(2)
            with col_hd_x:
                header_x = st.slider("X (px)", 0, 1080, 65, key="hx")
            with col_hd_y:
                header_y = st.slider("Y (px)", 0, 1920, 1300, key="hy")
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
                    logo_opacity, logo_scale, header_text, final_header_position, header_opacity, 
                    header_scale, header_color, header_bg_color, header_style, header_animation, video_bg_volume, 
                    header_custom_svg, header_font, progress_bar, status_text
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
        uploaded_logo, logo_position, logo_opacity, logo_scale,
        header_text, final_header_position, header_opacity, header_scale, header_color, header_bg_color, header_style, header_custom_svg, header_font
    )
    
    st.markdown("<div class='preview-container'>", unsafe_allow_html=True)
    st.markdown("<div class='preview-label'>Live Preview</div>", unsafe_allow_html=True)
    st.image(preview_image, use_column_width=True)
    st.markdown("<div class='preview-res'>1080 × 1920 · 9:16</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

