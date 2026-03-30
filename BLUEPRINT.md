# 1. PRODUCT OVERVIEW

## Problem
Creating engaging news videos for social media (TikTok, Reels, Shorts) is time-consuming. It requires synchronizing images, voiceovers, subtitles, and branding. Most tools are either too complex (Premiere Pro) or too manual (CapCut).

## User
Content creators, social media managers, and news aggregators who need to produce high-quality, branded news clips daily with minimal effort.

## Core Value Proposition
"From text and images to a viral-ready video in 60 seconds."
The app automates the "boring" parts (timing, transitions, overlays) while allowing enough customization (Dynamic Headers, Custom SVG) to maintain a unique brand identity.

## Use Cases
1. **Breaking News:** A creator hears a news story, grabs 3 images, writes the headline, and generates a video with a "Breaking News" header to post on TikTok immediately.
2. **Daily Roundup:** A news portal generates 5 short clips a day using a "Swiss Minimalist" style to maintain a consistent brand look across Instagram Reels.
3. **AI Narrated Stories:** A user provides a script for an AI voiceover; the app syncs the timing of images and headers to the spoken words automatically.

---

# 2. CORE FEATURES (MVP FIRST)

## MVP (Must Have)
### Image/Video Processing
- **Description:** Upload multiple images/videos and convert them into a 9:16 vertical video.
- **Inputs:** List of file paths (JPG, PNG, MP4).
- **Outputs:** A concatenated `ImageSequenceClip` / `CompositeVideoClip`.
- **Edge Cases:** Mixed aspect ratios (must be centered with blurred background), very short videos (must be handled gracefully).

### Dynamic Headers (The "Hook")
- **Description:** Automated text overlays that appear at specific times with animations.
- **Inputs:** String text, style selection (Neon, Glassmorphic, Custom SVG), position.
- **Outputs:** An animated `ImageClip` overlaying the video.
- **Edge Cases:** Text too long for the box (must auto-scale or wrap), special characters.

### Voiceover & Audio Sync
- **Description:** Sync the total duration of visual slides to the length of an uploaded audio file.
- **Inputs:** MP3/WAV file.
- **Outputs:** Final video matching the audio duration perfectly.
- **Edge Cases:** Audio shorter than the minimum slide duration (must trim audio or speed up visuals).

## V2 (Later)
- **Automatic Subtitles:** Transcribe audio and overlay "Alex Hormozi" style captions.
- **AI Voiceover Integration:** API connection to ElevenLabs/OpenAI TTS.
- **Cloud Rendering:** Move from local processing to AWS/GCP workers.
- **Template Library:** Pre-saved combinations of colors, fonts, and SVG styles.

---

# 3. USER FLOW

1. **Upload Phase:**
   - User uploads 3-10 images or videos.
   - User uploads a voiceover file (MP3).
2. **Configuration Phase:**
   - User chooses a "Design Style" for the header.
   - User enters the "Headline Text".
   - User selects the logo and its position.
3. **Preview Phase:**
   - User checks the "Live Preview" (static frame or short clip) to see if the layout looks good.
   - User adjusts colors/offsets if necessary.
4. **Generation Phase:**
   - User clicks "Build Video".
   - App renders the video using MoviePy.
   - User downloads the result.

---

# 4. SYSTEM ARCHITECTURE

## Components
- **Frontend (Web):** Streamlit (Python-based) for rapid prototyping and easy AI modification.
- **Backend (Engine):** MoviePy + CairoSVG + OpenCV.
- **Database:** None for MVP (Stateless). Session state handles temporary data.

## Communication
- User interactions trigger Python callbacks.
- Logic is strictly separated into `app.py` (UI) and `video_engine.py` (Business Logic/Rendering).

## Why this?
Streamlit allows a solo developer to build a full-featured web app in a single language (Python), making it extremely easy for AI to understand the entire context and make changes across UI and Engine simultaneously.

---

# 5. DATA MODEL (Internal State)

### Project Configuration (Object)
- `media_files`: List of strings (paths to temp files).
- `audio_file`: String (path to voiceover).
- `header_config`: Object {
    - `text`: String,
    - `style`: String (enum),
    - `custom_svg`: String (XML),
    - `color`: Hex String,
    - `position`: String (enum/XY)
}
- `rendering_options`: Object {
    - `bg_volume`: Float,
    - `resolution`: Tuple (1080, 1920)
}

---

# 6. API DESIGN (Internal Functions)

*Since this is a Streamlit app, "API" refers to core function signatures.*

### `generate_dynamic_header_img`
- **Goal:** Render the header plate.
- **Input:** `text`, `scale`, `color_hex`, `style`, `custom_svg`.
- **Output:** PIL Image object.

### `process_video_pipeline`
- **Goal:** The "Main Orchestrator".
- **Input:** All configuration objects.
- **Output:** Path to the generated `.mp4` file.

---

# 7. FRONTEND STRUCTURE

## Screens
1. **Main Dashboard:** Single-page app with 3 columns:
   - **Left:** Media Upload & Order (Streamlit Sortables).
   - **Middle:** Configuration Panels (Expanders for Header, Logo, Audio).
   - **Right:** Live Mockup Preview & Final Download button.

## Key Components
- `HeaderConfigPanel`: Conditional UI for Custom SVG vs Presets.
- `LivePreviewMockup`: Real-time PIL-based assembly of the current frame.
- `ProgressBar`: Visual feedback during MoviePy rendering.

---

# 8. BUSINESS LOGIC

1. **Duration Rule:** `total_duration = max(min_duration, audio.duration)`.
2. **Layout Rule:** Header must be constrained to safe zones (not too close to edges).
3. **SVG Rule:** Placeholders in Custom SVG `{text}`, `{color_hex}` are injected via string replacement before rendering.
4. **Math Rule:** `op-add` and `op-sub` in SVG allow AI to define relative positions.

---

# 9. PROJECT STRUCTURE

```text
/
├── app.py              # Main UI Logic (Streamlit)
├── video_engine.py      # Core Rendering Engine (MoviePy/CV2)
├── requirements.txt    # Python dependencies
├── BLUEPRINT.md        # Single Source of Truth
├── custom_svg_prompt.md # User guide for AI prompt generation
├── /assets             # Static assets (fonts, default logos)
└── /temp               # Temporary storage for uploads (clean scheduled)
```

---

# 10. DEVELOPMENT PLAN

1. **Stage 1: Core Engine** - Implement basic Image-to-Video conversion in `video_engine.py`.
2. **Stage 2: UI Foundation** - Setup Streamlit app.py with file uploaders.
3. **Stage 3: The Mockup Engine** - Implement `render_unified_mockup` for real-time visual feedback.
4. **Stage 4: Dynamic Headers** - Add preset SVG styles and the SVG-to-PNG renderer.
5. **Stage 5: Custom SVG & Math** - Implement the math parser for `{op-add}`.
6. **Stage 6: Final Polish** - Add progress bars, error handling, and crossfades.

---

# 11. RISKS & SIMPLIFICATIONS

- **Risk:** MoviePy memory leaks on long videos. **Simplification:** Limit MVP to 60-second clips.
- **Risk:** CairoSVG rendering inconsistencies. **Simplification:** Use a standard font (Arial) as fallback.
- **Simplified:** No User Auth or Database for MVP. Everything runs in the current session.

---

# 12. FUTURE EXTENSIONS

- **Templates:** JSON-based style definitions.
- **Audio Generation:** Direct integration with OpenAI Whisper (subtitles) and ElevenLabs (voice).
- **Editor Mode:** A timeline-based UI for micro-adjustments of clips.
