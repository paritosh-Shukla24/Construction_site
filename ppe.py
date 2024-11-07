# from ultralytics import YOLO
# import cv2
# import matplotlib.pyplot as plt

# # Load your custom-trained model
# model = YOLO('best.pt')  # Update 'best.pt' with the path to your model if needed

# # Load the image for testing
# image_path = 'first.jpeg'  # Replace with your image file path
# image = cv2.imread(image_path)

# # Run the model on the image
# results = model(image)

# # Display the results
# annotated_image = results[0].plot()

# # Use matplotlib to display the image
# plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

from ultralytics import YOLO
import cv2

# Load your custom-trained model
model = YOLO('best.pt')  # Update 'best.pt' with the path to your model

# Load the video for testing
video_path = '2048246-hd_1920_1080_24fps.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object to save the output
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec for .mp4 format
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run the model on the frame
    results = model(frame)
    
    # Get the annotated frame
    annotated_frame = results[0].plot()

    # Write the frame to the output video
    out.write(annotated_frame)

    # Display the frame (optional)
    cv2.imshow('YOLO Video', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
