import os

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Prajwal2608',
    'database': 'face_auth_db'
}

# Face Recognition Configuration
FACE_RECOGNITION = {
    'model_name': 'Facenet',
    'similarity_threshold': 0.7,
    'face_db_path': 'faceDB',
    'doctor_folder': 'DOCTORS',
    'patient_folder': 'PATIENTS',
    'debug_folder': 'debug',  # New: folder for debug images
    'temp_image_name': 'temp.jpg'  # New: name for temporary image
}

# Server Configuration
SERVER = {
    'host': 'localhost',
    'port': 8765
}

# File Naming Configuration
FILE_NAMING = {
    'separator': '_',
    'extension': '.jpg'
}

# Debug Configuration
DEBUG = {
    'save_received_images': True,  # Whether to save received images for debugging
    'image_format': 'jpg',  # Format for saved debug images
    'timestamp_format': '%Y%m%d_%H%M%S'  # Format for timestamp in debug filenames
}

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        FACE_RECOGNITION['face_db_path'],
        os.path.join(FACE_RECOGNITION['face_db_path'], FACE_RECOGNITION['doctor_folder']),
        os.path.join(FACE_RECOGNITION['face_db_path'], FACE_RECOGNITION['patient_folder']),
        os.path.join(FACE_RECOGNITION['face_db_path'], FACE_RECOGNITION['debug_folder'])
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True) 