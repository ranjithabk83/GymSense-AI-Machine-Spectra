"""
GYMSENSE AI - App Icon Generator
"""

import os
from PIL import Image, ImageDraw, ImageFont

os.makedirs("static/icons", exist_ok=True)

def create_icon(size: int, output_path: str):
    img = Image.new("RGBA", (size, size), (7, 10, 20, 255))
    draw = ImageDraw.Draw(img)

    # Rounded background with subtle gradient
    margin = size * 0.08
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=size * 0.22,
        fill=(14, 22, 43, 255),
        outline=(255, 46, 147, 180),
        width=int(size * 0.03)
    )

    # Ambient glow circle
    cx, cy = size / 2, size / 2
    r = size * 0.28
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=(255, 46, 147, 40)
    )

    # Stylized Dumbbell + Sensor Ring Symbol
    # Central Bar
    bar_w = size * 0.44
    bar_h = size * 0.07
    draw.rounded_rectangle(
        [cx - bar_w/2, cy - bar_h/2, cx + bar_w/2, cy + bar_h/2],
        radius=int(bar_h/2),
        fill=(255, 255, 255, 255)
    )

    # Left Weight Plates
    pw = size * 0.08
    ph = size * 0.32
    draw.rounded_rectangle(
        [cx - bar_w/2 - pw/2, cy - ph/2, cx - bar_w/2 + pw/2, cy + ph/2],
        radius=int(pw/3),
        fill=(255, 46, 147, 255)
    )

    # Right Weight Plates
    draw.rounded_rectangle(
        [cx + bar_w/2 - pw/2, cy - ph/2, cx + bar_w/2 + pw/2, cy + ph/2],
        radius=int(pw/3),
        fill=(0, 242, 254, 255)
    )

    # Sensor Pulse Ring
    draw.arc(
        [cx - r*1.15, cy - r*1.15, cx + r*1.15, cy + r*1.15],
        start=45,
        end=315,
        fill=(255, 46, 147, 220),
        width=int(size * 0.025)
    )

    img.save(output_path, "PNG")
    print(f"[OK] Generated {output_path} ({size}x{size})")

create_icon(192, "static/icons/icon-192.png")
create_icon(512, "static/icons/icon-512.png")
create_icon(180, "static/icons/apple-touch-icon.png")
