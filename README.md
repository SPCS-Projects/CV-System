# Visual Detection Project

This project aims to develop a real-time visual detection system capable of identifying individuals, recognizing faces, determining emotional expressions, and tracking behavior from video input. The system is designed to run efficiently on a Raspberry Pi with a connected camera.

## Features

-   **Person Detection:** Identify and track individuals in the video feed.
-   **Face Recognition:** Recognize known individuals by name or label as 'unknown'.
-   **Emotion Detection:** Determine facial expressions (e.g., sad, angry, happy, neutral).
-   **Behavior Tracking:** Monitor and analyze the behavior of detected individuals.
-   **Real-time Preview:** Display a live video feed with bounding boxes around detected faces.

## Technologies Used

-   **OpenCV:** For video processing and basic computer vision tasks.
-   **MediaPipe:** For efficient face detection and landmarking.
-   **DeepFace:** For robust face recognition and emotion analysis.
-   **ONNXRuntime:** (Optional) For optimizing model inference on Raspberry Pi.
-   **Streamlit:** For creating an interactive web-based user interface.
-   **Python:** The primary programming language.

## Hardware Requirements

-   Raspberry Pi (e.g., Raspberry Pi 4 Model B or newer for better performance)
-   Compatible USB Camera or Raspberry Pi Camera Module

## Raspberry Pi Optimization Notes

For optimal performance on a Raspberry Pi, consider the following:

-   **Model Selection:** DeepFace offers various models for face recognition and emotion analysis. Lighter models (e.g., `Facenet512` or `OpenFace` for recognition, and simpler emotion models) might perform better on resource-constrained devices like the Raspberry Pi. The current implementation uses `VGG-Face` for recognition and default emotion models, which can be computationally intensive.
-   **ONNXRuntime:** While `ONNXRuntime` is included in `requirements.txt`, its direct integration with DeepFace for model inference requires converting DeepFace's internal models to ONNX format, which is a more advanced optimization step not directly implemented in this initial version. However, `ONNXRuntime` can be used for other custom ONNX models if you choose to replace parts of DeepFace or MediaPipe with ONNX-optimized alternatives.
-   **Resolution and Frame Rate:** Reducing the camera resolution and frame rate can significantly improve performance.
-   **Headless Operation:** Running the Streamlit application in headless mode (without a graphical display) can save resources.
-   **Virtual Environment:** Always use a virtual environment to manage dependencies and avoid conflicts.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/visual-detection-project.git
    cd visual-detection-project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application, navigate to the project directory in your terminal and execute:

```bash
streamlit run app.py
```

This will open the application in your web browser. If running on a Raspberry Pi without a graphical interface, you might need to access it from another device on the same network using the IP address and port displayed in the terminal.

### Adding Known Faces

To enable face recognition, you need to populate the `known_faces` directory:

1.  Create subdirectories within `known_faces` for each person you want to recognize. The subdirectory name will be the person's name (e.g., `known_faces/John_Doe`).
2.  Place one or more clear images of the person's face (JPG or PNG format) inside their respective subdirectory. The more images from different angles and lighting conditions, the better the recognition accuracy.

Example:

```
visual_detection_project/
├── app.py
├── core_logic.py
├── requirements.txt
├── README.md
└── known_faces/
    ├── Alice/
    │   └── alice_1.jpg
    │   └── alice_2.png
    └── Bob/
        └── bob_face.jpg
```

## Project Structure

-   `app.py`: Main Streamlit application script.
-   `core_logic.py`: Contains the core computer vision and AI logic.
-   `requirements.txt`: Lists all Python dependencies.
-   `README.md`: Project documentation.
-   `known_faces/`: Directory for storing images of known individuals for face recognition.

## AI Models and Their Management

This project leverages pre-trained models from MediaPipe and DeepFace. These libraries handle the download and management of their respective models automatically upon first use.

### MediaPipe

-   **Model:** MediaPipe Face Detection (specifically `model_selection=0` which is a short-range model optimized for faces that are relatively close to the camera).
-   **Location:** MediaPipe models are typically downloaded and cached by the library itself, often in a user's application data directory (e.g., `~/.cache/mediapipe` on Linux-like systems). You generally do not need to manually download or place these models.
-   **Direct Link:** Not applicable, as MediaPipe manages its own model downloads.

### DeepFace

DeepFace is a wrapper that uses several state-of-the-art models for face recognition and emotion analysis. When you first run DeepFace functions (like `DeepFace.verify` or `DeepFace.analyze`), it will automatically download the necessary models if they are not already present.

-   **Face Recognition Model (used in `recognize_face`):** `VGG-Face`
    -   **Download Trigger:** The first time `DeepFace.verify(..., model_name="VGG-Face", ...)` is called.
    -   **Location:** DeepFace models are typically downloaded to `~/.deepface/weights`.
    -   **Direct Link:** DeepFace downloads these models from its GitHub releases or other specified URLs. You can find more details on the DeepFace GitHub repository: [https://github.com/serengil/deepface](https://github.com/serengil/deepface)

-   **Emotion Analysis Model (used in `detect_emotion`):** DeepFace uses an ensemble of models for emotion analysis. The primary model for emotion is usually based on a Convolutional Neural Network (CNN) trained on datasets like FER-2013.
    -   **Download Trigger:** The first time `DeepFace.analyze(..., actions=["emotion"], ...)` is called.
    -   **Location:** Similar to face recognition models, these are downloaded to `~/.deepface/weights`.
    -   **Direct Link:** Managed by DeepFace.

### ONNXRuntime

`ONNXRuntime` is included for potential future optimizations. It does not come with pre-trained models for this specific application. If you were to use it, you would typically convert existing models (e.g., from TensorFlow or PyTorch) into the ONNX format and then load them with `onnxruntime`.

-   **Model:** None provided directly by this project.
-   **Location:** Custom ONNX models would be placed in a designated `models/` directory (which is mentioned in the `Project Structure` but not yet created as it's optional).
-   **Direct Link:** Not applicable, as this is for custom ONNX models you might create or acquire.

**Important Note:** The initial download of these models (especially DeepFace models) can take some time and requires an internet connection. Once downloaded, they are cached locally and do not need to be downloaded again.
