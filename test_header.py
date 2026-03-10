from video_engine import generate_dynamic_header_img
try:
    styles = [
        "1. Neon Edge",
        "2. Glassmorphic Ribbon",
        "3. Floating Pill",
        "4. Split-Grid Panel",
        "5. Double Stroke Box"
    ]
    
    for i, style in enumerate(styles):
        img = generate_dynamic_header_img(
            text=f"STYLE TEST\n{style.upper()}", 
            scale=1.0, 
            color_hex="#FF6E00", 
            bg_color_hex="#000000",
            opacity=0.9, 
            style=style, 
            header_position="Center"
        )
        img.save(f"test_out_{i+1}.png")
        print(f"Image {style} generated successfully: {img.size}")
except Exception as e:
    import traceback
    traceback.print_exc()
