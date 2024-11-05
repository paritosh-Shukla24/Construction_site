import base64
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="7OYXBrvSIIBJHtvLytfM"
)

# Load the image file and encode it as base64
with open("image2.jpeg", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

# Perform inference using the base64-encoded image
result = CLIENT.infer(image_base64, model_id="bismillah-e3vym/4")

# Decode the base64 image for visualization
image_data = base64.b64decode(image_base64)
image = Image.open(io.BytesIO(image_data))

# Define colors for each class ID
color_map = {
    0: "red",        # Helmet
    1: "blue",       # Another class, e.g., Hat
    2: "green",      # Another class, e.g., Mask
    3: "purple",     # Person
    4: "orange"      # Vest
    # Add more class IDs and colors as needed
}

# Draw bounding boxes and labels on the image
draw = ImageDraw.Draw(image)

for prediction in result['predictions']:
    x, y = prediction['x'], prediction['y']
    width, height = prediction['width'], prediction['height']
    confidence = prediction['confidence']
    label = prediction['class']
    class_id = prediction['class_id']

    # Assign color based on the class ID
    color = color_map.get(class_id, "white")  # Default to white if class ID not in color map

    # Calculate bounding box coordinates
    left = x - width / 2
    top = y - height / 2
    right = x + width / 2
    bottom = y + height / 2

    # Draw the bounding box
    draw.rectangle([left, top, right, bottom], outline=color, width=3)

    # Set dynamic font size based on bounding box width
    font_size = int(width / 10)  # Adjust the divisor as needed for font scaling
    font_size = max(15, min(font_size, 40))  # Limit font size between 15 and 40
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Load custom font with dynamic size
    except IOError:
        font = ImageFont.load_default()  # Fallback to default if custom font fails

    # Draw the label with confidence score above the bounding box
    label_text = f"{label} ({confidence:.2f})"
    
    # Calculate text position above the bounding box
    text_bbox = draw.textbbox((left, top), label_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_position = (left, top - text_height - 5)

    # Draw a filled rectangle for the text background for readability
    draw.rectangle(
        [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
        fill=color
    )
    # Draw the text (label and confidence score)
    draw.text(text_position, label_text, fill="white", font=font)

# Display the image with bounding boxes and labels
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.show()
