# Face Recognition-based Hospital Access Control System

## Project Overview

This project implements a Face Recognition-based Hospital Access Control System designed to enhance security and streamline access management in healthcare facilities. The system uses facial recognition technology to authenticate doctors, medical personnel, and patients, ensuring that only authorized individuals can access specific areas or patient information.

## Architecture

The system follows a client-server architecture with the following components:

1. **Client-side:**
   - Face detection and embedding generation module (to be implemented)
   - User interface for registration and authentication (to be implemented)
   - WebSocket client for real-time communication with the server

2. **Server-side:**
   - FastAPI web server
   - WebSocket server for real-time authentication
   - MySQL database for storing user information and face embeddings
   - Authentication and access control logic

3. **Database:**
   - MySQL database for storing user data, face embeddings, and access permissions

## Technologies Used

- Backend:
  - Python 3.8+
  - FastAPI (web framework)
  - WebSockets (for real-time communication)
  - MySQL (database)
  - NumPy (for numerical operations on embeddings)
- Frontend (to be developed):
  - React.js or Vue.js (for building the user interface)
  - WebSocket client library
- Face Recognition:
  - FaceNet or a similar deep learning model for generating face embeddings
- Testing:
  - pytest for backend testing

## Workflow

1. **User Registration:**
   - A new user (doctor, medical personnel, or patient) stands in front of the camera.
   - The system captures the face and generates an embedding.
   - User details and face embedding are sent to the server.
   - The server stores the information in the database.

2. **Authentication:**
   - A doctor/medical personnel stands in front of the camera.
   - The system captures the face and generates an embedding.
   - A patient stands in front of the camera.
   - The system captures the patient's face and generates an embedding.
   - Both embeddings are sent to the server via WebSocket.
   - The server verifies the identities and checks access permissions.
   - The server sends back the authentication result.
   - The system grants or denies access based on the result.

3. **Access Management:**
   - Administrators can grant or revoke access permissions for doctors/medical personnel to specific patients.
   - This is done through the `create_access` API endpoint.

## Setup and Installation

1. Clone the repository:



2. Set up a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up the MySQL database:
- Create a new database named `hospital_db`.
- Run the SQL script in `database_schema.sql` to create the necessary tables.

5. Configure the database connection:
- Update the database connection details in `database.py`.

6. Run the FastAPI server:

```bash
uvicorn main:app --reload
```


## API Endpoints

- WebSocket: `/ws/auth` (for real-time authentication)
- POST `/register_face` (for user registration)
- POST `/create_access` (for managing access permissions)

## Testing

Run the test suite using pytest:

```bash
py testing.py
```

## Future Enhancements

- Implement the client-side face detection and embedding generation.
- Develop a user-friendly frontend interface.
- Implement multi-factor authentication for higher security.
- Integrate with hospital management systems.
- Implement an admin panel for managing users and permissions.

## Security Considerations

- Face embeddings are stored as binary data in the database.
- Implement proper error handling and input validation.
- Use HTTPS for all communications.
- Implement rate limiting to prevent brute-force attacks.
- Regularly update and patch all dependencies.

## Conclusion

This Face Recognition-based Hospital Access Control System provides a secure and efficient way to manage access in healthcare facilities. By following the setup instructions and understanding the workflow, you can implement and customize this system to meet specific healthcare security needs.
