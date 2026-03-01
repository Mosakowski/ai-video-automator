import sys
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

def generate_tiktok_header(text, opacity=1.0):
    lines = text.strip().split('\n')
    
    pad_x = 40
    pad_y = 20
    line_spacing = -10 # Negative spacing to bunch them up like in the reference
    
    try:
        font = ImageFont.truetype('arialbd.ttf', 70)
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
            tw = len(line) * 35
            th = 70
            
        box_w = tw + pad_x * 2
        box_h = th + pad_y * 2
        line_dims.append({'text': line, 'tw': tw, 'th': th, 'bw': box_w, 'bh': box_h})
        
        if box_w > total_w: total_w = box_w
        total_h += box_h + line_spacing
        
    total_h -= line_spacing # remove last spacing
    
    # Add padding for glow effect
    glow_radius = 25
    canvas_w = total_w + glow_radius * 2
    canvas_h = total_h + glow_radius * 2
    
    img = Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))
    glow_layer = Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))
    shapes_layer = Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))
    text_layer = Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))
    
    glow_draw = ImageDraw.Draw(glow_layer)
    shapes_draw = ImageDraw.Draw(shapes_layer)
    text_draw = ImageDraw.Draw(text_layer)
    
    current_y = glow_radius
    
    for dim in line_dims:
        # Align right as in reference image!
        x_offset = glow_radius + (total_w - dim['bw'])
        
        box_rect = [x_offset, current_y, x_offset + dim['bw'], current_y + dim['bh']]
        
        # Glow (Thicker outline drawn on blur layer)
        glow_draw.rounded_rectangle(box_rect, radius=20, outline=(255, 40, 40, 255), width=15)
        
        # Main Shape (dark blue-grey with sharp red border)
        shapes_draw.rounded_rectangle(box_rect, radius=20, fill=(30, 35, 45, int(255*opacity)), outline=(255, 60, 60, int(255*opacity)), width=3)
        
        # Text
        tx = x_offset + pad_x
        ty = current_y + pad_y - 12 # manual vertical tweak for Arial
        
        # Draw a slight drop shadow behind text to make it pop like in the reference
        text_draw.text((tx+3, ty+3), dim['text'], fill=(0, 0, 0, int(150*opacity)), font=font)
        text_draw.text((tx, ty), dim['text'], fill=(255, 255, 255, int(255*opacity)), font=font)
        
        current_y += dim['bh'] + line_spacing
        
    # Apply Blur to glow layer
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(glow_radius // 2))
    
    # Composite
    img = Image.alpha_composite(img, glow_layer)
    img = Image.alpha_composite(img, shapes_layer)
    img = Image.alpha_composite(img, text_layer)
    
    img.save('test_header_2.png')
    print('Saved test_header_2.png', canvas_w, canvas_h)

generate_tiktok_header("ATAK IRANSKICH BOMB\nNA DUBAJ")
