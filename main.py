import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import os
import time
from collections import deque

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Directory for known faces
KNOWN_FACES_DIR = "known_faces"

# Ensure known_faces directory exists
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Load known faces (simple approach: each subfolder is a person's name)
known_faces_db = {}
for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if os.path.isdir(person_dir):
        known_faces_db[person_name] = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith((".jpg", ".png"))]

print(f"Loaded {len(known_faces_db)} known individuals.")

def recognize_face(face_image):
    try:
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        temp_face_path = "temp_face.jpg"
        cv2.imwrite(temp_face_path, face_image_rgb)

        for person_name, face_paths in known_faces_db.items():
            for known_face_path in face_paths:
                result = DeepFace.verify(img1_path=temp_face_path, img2_path=known_face_path, model_name="VGG-Face", enforce_detection=False, distance_metric='cosine')
                if result["verified"]:
                    os.remove(temp_face_path)
                    return person_name
        os.remove(temp_face_path)
        return "Unknown"
    except Exception as e:
        if os.path.exists(temp_face_path):
            os.remove(temp_face_path)
        return "Unknown"

def detect_emotion(face_image):
    try:
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        demography = DeepFace.analyze(img_path=face_image_rgb, actions=["emotion"], enforce_detection=False, prog_bar=False)
        
        if demography and isinstance(demography, list) and len(demography) > 0:
            return demography[0]["dominant_emotion"]
        return "Unknown Emotion"
    except Exception as e:
        return "Unknown Emotion"

# Behavior tracking data structure
# Stores last N emotions for each recognized person
person_behavior = {}
EMOTION_HISTORY_LENGTH = 10 # Keep track of last 10 emotions

def track_behavior(person_name, emotion):
    if person_name == "Unknown":
        return

    if person_name not in person_behavior:
        person_behavior[person_name] = {
            "emotion_history": deque(maxlen=EMOTION_HISTORY_LENGTH),
            "last_seen": time.time()
        }
    
    person_behavior[person_name]["emotion_history"].append(emotion)
    person_behavior[person_name]["last_seen"] = time.time()

    # Simple behavior analysis: check for frequent emotion changes or prolonged single emotion
    # This can be expanded significantly.
    if len(person_behavior[person_name]["emotion_history"]) == EMOTION_HISTORY_LENGTH:
        unique_emotions = set(person_behavior[person_name]["emotion_history"])
        if len(unique_emotions) > EMOTION_HISTORY_LENGTH / 2: # More than half unique emotions
            # print(f"{person_name} is showing varied emotions.")
            pass
        elif len(unique_emotions) == 1:
            # print(f"{person_name} is consistently {list(unique_emotions)[0]}.")
            pass

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Camera opened successfully. Press \'q\' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        current_frame_recognized_people = set()

        if results.detections:
            for detection in results.detections:
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), \
                             int(bbox_c.width * iw), int(bbox_c.height * ih)

                x = max(0, x)
                y = max(0, y)
                w = min(iw - x, w)
                h = min(ih - y, h)

                if w > 0 and h > 0:
                    face_img = frame[y:y+h, x:x+w]

                    person_name = recognize_face(face_img)
                    emotion = detect_emotion(face_img)

                    track_behavior(person_name, emotion)
                    current_frame_recognized_people.add(person_name)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                    text = f'{person_name} ({emotion})'
                    cv2.putText(frame, text, (x, y - 10), \
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        # Clean up behavior data for people not seen recently
        for person_name in list(person_behavior.keys()):
            if person_name not in current_frame_recognized_people and (time.time() - person_behavior[person_name]["last_seen"]) > 5: # 5 seconds timeout
                del person_behavior[person_name]

        cv2.imshow('Visual Detection System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()

if __name__ == '__main__':
    main()

