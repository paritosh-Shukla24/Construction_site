# from ultralytics import YOLO
# import cv2
# import dash
# from dash import dcc, html
# from dash.dependencies import Output, Input
# import dash_bootstrap_components as dbc
# import base64
# import numpy as np
# import threading

# # Load YOLO model
# model = YOLO('best.pt')  # Replace with your model file path if necessary

# # Initialize Dash app
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# # Layout of the dashboard
# app.layout = dbc.Container([
#     dbc.Row([
#         dbc.Col(html.H2("Real-Time Safety Compliance Dashboard"), width=12)
#     ]),
#     dbc.Row([
#         dbc.Col(dcc.Graph(id="live-video-feed", config={"displayModeBar": False}), width=8),
#         dbc.Col(html.Div(id="detection-sidebar"), width=4),
#     ]),
#     dcc.Interval(id="interval-component", interval=500, n_intervals=0)  # Update every 500 ms
# ])

# # Video capture setup
# video_path = 0  # 0 for webcam or replace with video file path
# cap = cv2.VideoCapture(video_path)

# # Negative labels to watch for
# negative_labels = {'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'}

# # Function to process frame and run YOLO model
# def process_frame():
#     ret, frame = cap.read()
#     if not ret:
#         return None, []
    
#     # Detect objects in the frame
#     results = model(frame)
#     annotated_frame = results[0].plot()  # Annotated frame with bounding boxes
    
#     # Prepare the frame for display
#     _, buffer = cv2.imencode('.jpg', annotated_frame)
#     frame_encoded = base64.b64encode(buffer).decode('utf-8')

#     # Collect detection information for negative labels only
#     detection_info = []
#     for result in results[0].boxes:
#         label = result.cls
#         confidence = round(float(result.conf), 2)
        
#         # Debugging: print detected labels to console
#         print(f"Detected label: {label}, confidence: {confidence}")
        
#         # Only add if label is in negative_labels
#         if label in negative_labels:
#             detection_info.append({
#                 "label": label,
#                 "confidence": confidence
#             })
    
#     return frame_encoded, detection_info

# # Update video feed and detection sidebar
# @app.callback(
#     [Output("live-video-feed", "figure"), Output("detection-sidebar", "children")],
#     [Input("interval-component", "n_intervals")]
# )
# def update_dashboard(n_intervals):
#     frame_encoded, detection_info = process_frame()
    
#     # Create image for display
#     figure = {
#         "data": [{
#             "x": [0],
#             "y": [0],
#             "type": "image",
#             "source": f"data:image/jpeg;base64,{frame_encoded}",
#             "xref": "x",
#             "yref": "y",
#             "sizing": "stretch",
#             "layer": "below",
#         }],
#         "layout": {
#             "xaxis": {"visible": False},
#             "yaxis": {"visible": False},
#             "margin": {"t": 0, "b": 0, "l": 0, "r": 0}
#         }
#     }

#     # Update sidebar with negative detection details
#     sidebar_content = [html.H4("Negative Detections")]
#     if detection_info:
#         for detection in detection_info:
#             sidebar_content.append(html.Div([
#                 html.P(f"Label: {detection['label']}"),
#                 html.P(f"Confidence: {detection['confidence']}")
#             ], style={"marginBottom": "10px"}))
#     else:
#         sidebar_content.append(html.P("No negative detections found."))

#     return figure, sidebar_content

# # Run the app
# if __name__ == "__main__":
#     app.run_server(debug=True)

from ultralytics import YOLO
import cv2
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import base64
import numpy as np

# Load YOLO model
model = YOLO('best.pt')  # Replace with your model file path if necessary

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Label mappings
labels_dict = {
    0: "Hardhat",
    1: "Mask",
    2: "Safety Vest",
    3: "NO-Hardhat",
    4: "NO-Mask",
    5: "NO-Safety Vest",
    6: "Person"
}

# Layout of the dashboard
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Real-Time Safety Compliance Dashboard"), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="live-video-feed", config={"displayModeBar": False}), width=8),
        dbc.Col(html.Div(id="detection-sidebar"), width=4),
    ]),
    dcc.Interval(id="interval-component", interval=500, n_intervals=0)  # Update every 500 ms
])

# Video capture setup
video_path = 0  # 0 for webcam or replace with video file path
cap = cv2.VideoCapture(video_path)

# Negative labels to watch for
negative_labels = {'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'}

# Function to process frame and run YOLO model
def process_frame():
    ret, frame = cap.read()
    if not ret:
        return None, []

    # Detect objects in the frame
    results = model(frame)
    annotated_frame = results[0].plot()  # Annotated frame with bounding boxes

    # Encode frame for display in Dash app
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')

    # Collect detection information for negative labels only
    detection_info = []
    for result in results[0].boxes:
        label_id = int(result.cls)
        label = labels_dict.get(label_id, "Unknown")
        confidence = round(float(result.conf), 2)
        
        # Debugging: print detected labels to console
        print(f"Detected label: {label}, confidence: {confidence}")
        
        # Only add if label is in negative_labels
        if label in negative_labels:
            detection_info.append({
                "label": label,
                "confidence": confidence
            })
    
    return frame_encoded, detection_info

# Update video feed and detection sidebar
@app.callback(
    [Output("live-video-feed", "figure"), Output("detection-sidebar", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_dashboard(n_intervals):
    frame_encoded, detection_info = process_frame()
    
    # Create image for display
    figure = {
        "data": [{
            "x": [0],
            "y": [0],
            "type": "image",
            "source": f"data:image/jpeg;base64,{frame_encoded}",
            "xref": "x",
            "yref": "y",
            "sizing": "stretch",
            "layer": "below",
        }],
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "margin": {"t": 0, "b": 0, "l": 0, "r": 0}
        }
    }

    # Update sidebar with negative detection details
    sidebar_content = [html.H4("Negative Detections")]
    if detection_info:
        for detection in detection_info:
            sidebar_content.append(html.Div([
                html.P(f"Label: {detection['label']}"),
                html.P(f"Confidence: {detection['confidence']}"),
            ], style={"marginBottom": "10px"}))
    else:
        sidebar_content.append(html.P("No negative detections found."))

    return figure, sidebar_content

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
