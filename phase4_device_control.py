# phase4_device_control.py

import cv2
from utils.hand_tracker import HandTracker
from utils.gesture_recognizer import GestureRecognizer
from utils.system_control import (
    set_system_volume,
    set_screen_brightness,
    toggle_play_pause,
    next_track,
    previous_track,
    open_application,
)

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
        h, w, _ = img.shape

        for hand in hands_data:
            lm = {id: (x, y) for id, x, y in hand["landmarks"]}

            # Example: palm distance (thumb tip to pinky tip) → volume
            thumb_tip = lm[4]
            pinky_tip = lm[20]
            palm_dist = recognizer.distance(thumb_tip, pinky_tip)
            volume = int(max(0, min(100, (palm_dist - 20) / (200 - 20) * 100)))
            set_system_volume(volume)

            # Example: pinch distance thumb-index → brightness
            index_tip = lm[8]
            pinch_dist = recognizer.distance(thumb_tip, index_tip)
            brightness = int(max(0, min(100, (pinch_dist - 20) / (120 - 20) * 100)))
            set_screen_brightness(brightness)

            fingers = recognizer.get_finger_states(hand)
            static_gesture = recognizer.get_static_gesture(fingers)

            if static_gesture == "Peace":
                toggle_play_pause()
            elif static_gesture == "Point":
                open_application("chrome")

        cv2.imshow("Phase 4 - Device Control (stub)", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
