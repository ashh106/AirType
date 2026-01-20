# phase2_gesture_recognition.py

import cv2
from utils.hand_tracker import HandTracker
from utils.gesture_recognizer import GestureRecognizer

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    recognizer = GestureRecognizer(history_len=15)

    last_swipe = None

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        hands_data = tracker.process(img, draw=True)

        for hand in hands_data:
            fingers = recognizer.get_finger_states(hand)
            static_gesture = recognizer.get_static_gesture(fingers)
            recognizer.update_index_history(hand)
            swipe = recognizer.get_swipe()

            if swipe:
                last_swipe = swipe

            txt1 = f"Static: {static_gesture}"
            txt2 = f"Swipe: {last_swipe if last_swipe else 'None'}"

            cv2.putText(img, txt1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, txt2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        fps = tracker.get_fps()
        cv2.putText(
            img, f"FPS: {int(fps)}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )

        cv2.imshow("Phase 2 - Gesture Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
