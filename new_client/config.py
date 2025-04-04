import os

# Server Configuration
SERVER = {
    'host': 'localhost',
    'port': 8765,
    'url': f'ws://localhost:8765'
}

# Camera Configuration
CAMERA = {
    'width': 640,
    'height': 480,
    'fps': 30
}

# Face Detection Configuration
FACE_DETECTION = {
    'model_path': 'yolov8n-face-lindevs.pt',
    'confidence_threshold': 0.5,
    'box_color': (0, 255, 0),  # Green color for bounding boxes
    'box_thickness': 2
}

# UI Configuration
UI = {
    'window_title': 'Face Authentication System',
    'window_size': (800, 600),
    'button_height': 50,
    'camera_size': (640, 480),
    'status_bar_height': 30,
    'font_size': 16
}

# File Paths
PATHS = {
    'models': 'models',
    'temp': 'temp'
}

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        PATHS['models'],
        PATHS['temp']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True) 