import re
from PIL import Image, ImageDraw, ImageFont
import os


def image_annotate(image_path, font_size=30, font_type='Ubuntu-R.ttf'):
    for root, _, files in os.walk(image_path):
        for file in files:
            image = Image.open(os.path.join(image_path, '{}'.format(file)))

            padding = 30
            new_size = (image.width + 2 * padding, image.height + 2 * padding)
            new_image = Image.new("RGB", new_size, "black")
            new_image.paste(image, (padding, padding))

            draw = ImageDraw.Draw(new_image)

            text = 'epoch : ' + file.split('_')[1].split('.')[0]
            text_color = (255, 0, 0)

            font = ImageFont.truetype(font_type, size=font_size)
            draw.text((0, 0), text, fill=text_color, font=font)

            new_image.save(os.path.join(image_path, '{}'.format(file)))


def sort_key(name):
    numbers = re.findall(r'\d+', name)

    return int(numbers[0]) if numbers else 0


def image_to_gif(image_path):
    for _, _, files in os.walk(image_path):
        image_files = sorted(files, key=sort_key)

    images = []

    for file in image_files:
        img = Image.open(os.path.join(image_path, file))
        images.append(img)

    images[0].save('sample.gif', save_all=True, append_images=images[1:], duration=300, loop=0)
