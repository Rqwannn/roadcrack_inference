import inference
import os
from PIL import Image, ImageDraw

def call_model(images):
    model = inference.get_model(
        "crackdetection-oufte/2",
        api_key=os.getenv("ROBOFLOW_API_KEY")
    )

    prediction = model.infer(image=images)
    predictions = prediction[0].predictions

    image = Image.open(images)
    draw = ImageDraw.Draw(image)

    cropped_images = []

    for item in predictions:
        if item.confidence > 0.8:
            x1 = int(item.x - item.width / 2)
            y1 = int(item.y - item.height / 2)
            x2 = int(item.x + item.width / 2)
            y2 = int(item.y + item.height / 2)

            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped_image)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            text = f"{item.class_name} - {item.confidence:.2f}"
            draw.text((x1, y1), text, fill="red")

    image.show()

    return cropped_images, image
