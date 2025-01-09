import inference
import os
from PIL import Image, ImageDraw

def call_model(images):
    model = inference.get_model(
        "roadcracknewversion/1",
        api_key=os.getenv("ROBOFLOW_API_KEY")
    )

    prediction = model.infer(image=images)
    predictions = prediction[0].predictions

    image = Image.open(images)
    draw = ImageDraw.Draw(image)

    cropped_images = []
    view_bbox = []

    for index, item in enumerate(predictions):
        if item.confidence > 0.5:
            x1 = int(item.x - item.width / 2)
            y1 = int(item.y - item.height / 2)
            x2 = int(item.x + item.width / 2)
            y2 = int(item.y + item.height / 2)

            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped_image)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            # text = f"No {index} - {item.class_name} - {item.confidence:.2f}"
            # draw.text((x1, y1), text, fill="red")

            padding = 12

            available_padding_top = y1
            available_padding_bottom = image.height - y2

            actual_padding_top = min(padding, available_padding_top)
            actual_padding_bottom = min(padding, available_padding_bottom)

            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - actual_padding_top)
            x2_padded = min(image.width, x2 + padding)
            y2_padded = min(image.height, y2 + actual_padding_bottom)

            padded_cropped_image = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
            view_bbox.append(padded_cropped_image)

    # image.show()

    return cropped_images, view_bbox, image
