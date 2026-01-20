# phase5_authentication.py

import time
import cv2
from utils.hand_tracker import HandTracker
from utils.gesture_recognizer import GestureRecognizer

# simple in-memory "stored gesture password"
stored_pattern = []

def record_gesture_pattern(tracker, recognizer, duration=5):
    print("[INFO] Recording gesture password...")
    start = time.time()
    pattern = []

    cap = cv2.VideoCapture(0)

    while time.time() - start < duration:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        hands_data = tracker.process(img, draw=True)

        current_gesture = "None"
        if hands_data:
            fingers = recognizer.get_finger_states(hands_data[0])
            current_gesture = recognizer.get_static_gesture(fingers)
            pattern.append(current_gesture)

        cv2.putText(img, f"Recording: {current_gesture}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.imshow("Record Gesture Pattern", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Recorded pattern:", pattern)
    return pattern

def authenticate_gesture_pattern(tracker, recognizer, stored, duration=5):
    print("[INFO] Perform gesture pattern to authenticate...")
    start = time.time()
    pattern = []

    cap = cv2.VideoCapture(0)

    while time.time() - start < duration:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        hands_data = tracker.process(img, draw=True)

        current_gesture = "None"
        if hands_data:
            fingers = recognizer.get_finger_states(hands_data[0])
            current_gesture = recognizer.get_static_gesture(fingers)
            pattern.append(current_gesture)

        cv2.putText(img, f"Auth: {current_gesture}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.imshow("Authenticate Gesture Pattern", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Entered pattern:", pattern)

    # naive comparison
    return pattern == stored

def main():
    tracker = HandTracker()
    recognizer = GestureRecognizer()

    global stored_pattern
    stored_pattern = record_gesture_pattern(tracker, recognizer, duration=5)

    success = authenticate_gesture_pattern(tracker, recognizer, stored_pattern, duration=5)
    if success:
        print("[AUTH] Authentication SUCCESS")
    else:
        print("[AUTH] Authentication FAILED")

if __name__ == "__main__":
    main()
