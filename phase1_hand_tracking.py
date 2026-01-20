# phase1_hand_tracking.py

import cv2
from utils.hand_tracker import HandTracker
from utils.gesture_recognizer import GestureRecognizer

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    recognizer = GestureRecognizer()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        hands_data = tracker.process(img, draw=True)

        for hand in hands_data:
            fingers = recognizer.get_finger_states(hand)
            gesture = recognizer.get_static_gesture(fingers)

            # Display finger states and gesture
            text = f"Fingers: {fingers}  Gesture: {gesture}"
            cv2.putText(
                img,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # FPS
        fps = tracker.get_fps()
        cv2.putText(
            img, f"FPS: {int(fps)}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )

        cv2.imshow("Phase 1 - Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
