from PIL import Image, ImageDraw, ImageFont
import json

def save_image_with_metadata(image, filepath, metadata):
    """Save image and metadata as JSON"""
    image.save(filepath, "PNG")
    json_path = filepath.replace(".png", ".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

def apply_watermark(image, text="AI Generated"):
    """Add watermark text to image"""
    watermark_image = image.copy()
    draw = ImageDraw.Draw(watermark_image)
    width, height = watermark_image.size
    fontsize = int(width / 32)
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except:
        font = ImageFont.load_default()
    draw.text((10, height - fontsize - 10), text, fill=(255, 255, 255), font=font)
    return watermark_image


