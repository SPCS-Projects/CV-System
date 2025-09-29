import streamlit as st
import cv2
import time
from core_logic import process_frame, cleanup_resources, KNOWN_FACES_DIR # Import necessary functions and variables
import os

st.title("Real-time Visual Detection System")

# Sidebar for controls
st.sidebar.header("Controls")
start_button = st.sidebar.button("Start Camera")
stop_button = st.sidebar.button("Stop Camera")

# Placeholder for the video feed
frame_window = st.image([], caption="Live Camera Feed", use_column_width=True)

# Placeholder for status messages
status_text = st.sidebar.empty()

# Session state to manage camera status
if 'camera_started' not in st.session_state:
    st.session_state.camera_started = False

if start_button and not st.session_state.camera_started:
    st.session_state.camera_started = True
    status_text.info("Starting camera...")
    cap = cv2.VideoCapture(0) # 0 for default camera

    if not cap.isOpened():
        status_text.error("Error: Could not open video stream.")
        st.session_state.camera_started = False
    else:
        status_text.success("Camera opened successfully. Press 'Stop Camera' to quit.")
        while st.session_state.camera_started:
            ret, frame = cap.read()
            if not ret:
                status_text.error("Error: Could not read frame.")
                break

            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)

            # Process the frame using the core logic
            processed_frame = process_frame(frame)

            # Display the processed frame in Streamlit
            frame_window.image(processed_frame, channels="BGR")

            # Small delay to prevent high CPU usage and allow Streamlit to update
            time.sleep(0.01)

            # Check if stop button was pressed (Streamlit re-runs the script on button press)
            if stop_button:
                break

        cap.release()
        cleanup_resources() # Clean up MediaPipe resources
        st.session_state.camera_started = False
        status_text.info("Camera stopped.")
elif stop_button and st.session_state.camera_started:
    st.session_state.camera_started = False
    status_text.info("Stopping camera...")

# Display known faces directory status
st.sidebar.markdown(f"**Known Faces Directory:** `{KNOWN_FACES_DIR}`")
if not os.path.exists(KNOWN_FACES_DIR) or not os.listdir(KNOWN_FACES_DIR):
    st.sidebar.warning("No known faces found. Create subfolders in 'known_faces' with images for recognition.")
else:
    st.sidebar.success(f"Found {len(os.listdir(KNOWN_FACES_DIR))} known face folders.")

