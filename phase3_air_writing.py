# phase3_air_writing.py

import cv2
import numpy as np
from utils.hand_tracker import HandTracker
from utils.gesture_recognizer import GestureRecognizer
import math


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_palm_width(lm_dict):
    """
    Approximate palm width using landmarks 5 (index base) and 17 (pinky base).
    """
    p1 = lm_dict[5]
    p2 = lm_dict[17]
    return dist(p1, p2)


def is_strict_pinch(lm_dict, id1, id2, palm_width, tight_factor=0.32, align_factor=0.18):
    """
    More accurate pinch:
    - distance between fingertips relative to palm size
    - vertical alignment constraint to ensure tips really "meet"
    """
    p1 = lm_dict[id1]
    p2 = lm_dict[id2]

    d = dist(p1, p2)
    max_dist = palm_width * tight_factor
    if d > max_dist:
        return False

    # vertical alignment: top points meet properly (y close)
    dy = abs(p1[1] - p2[1])
    if dy > palm_width * align_factor:
        return False

    return True


def enhance_low_light(img):
    """
    Simple low-light enhancement:
    - if frame is too dark, boost brightness/contrast slightly
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray.mean()

    if mean_intensity < 70:  # adjust threshold if needed
        alpha = 1.4  # contrast
        beta = 25    # brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img


def main():
    cap = cv2.VideoCapture(0)
    # slightly higher confidence → more stable tracking
    tracker = HandTracker(detection_confidence=0.7, tracking_confidence=0.7)
    recognizer = GestureRecognizer()

    CANVAS_H, CANVAS_W = 600, 800
    GRID_SPACING = 50

    # Stroke storage for undo/redo
    strokes = []          # list of strokes, each stroke is list of (x, y)
    undone_strokes = []   # for redo
    current_stroke = []

    # Gesture timing counters
    draw_hold_frames = 0
    erase_hold_frames = 0
    undo_hold_frames = 0
    redo_hold_frames = 0

    DRAW_ACTIVATE_FRAMES = 6    # frames to start drawing
    ERASE_ACTIVATE_FRAMES = 6   # frames to trigger erase-all
    UNDO_ACTIVATE_FRAMES = 6
    REDO_ACTIVATE_FRAMES = 6

    drawing_active = False
    smoothed_point = None

    while True:
        success, raw_img = cap.read()
        if not success:
            break

        raw_img = cv2.flip(raw_img, 1)

        # Low-light enhancement for better landmark detection
        img = enhance_low_light(raw_img.copy())
        base_img = img.copy()

        # Process for landmarks (no drawing)
        hands_data = tracker.process(img, draw=False)

        # Small hand view with skeleton (use enhanced image)
        hand_view = base_img.copy()
        tracker.process(hand_view, draw=True)

        h, w, _ = img.shape

        # Base canvas (rebuild every frame)
        canvas_base = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255

        # Draw grid on canvas
        for x in range(0, CANVAS_W, GRID_SPACING):
            cv2.line(canvas_base, (x, 0), (x, CANVAS_H), (230, 230, 230), 1)
        for y in range(0, CANVAS_H, GRID_SPACING):
            cv2.line(canvas_base, (0, y), (CANVAS_W, y), (230, 230, 230), 1)

        # Redraw all committed strokes
        for stroke in strokes:
            for i in range(1, len(stroke)):
                cv2.line(canvas_base, stroke[i - 1], stroke[i], (0, 0, 0), 5)

        # Gesture states for this frame (raw detections)
        draw_gesture_present = False
        erase_gesture_present = False
        undo_gesture_present = False
        redo_gesture_present = False

        index_tip_canvas_pos = None

        # ---- Gesture detection ----
        if hands_data:
            for hand in hands_data:
                lm = {id: (x, y) for id, x, y in hand["landmarks"]}
                handedness = hand["handedness"]   # 'Left' or 'Right'

                palm_width = get_palm_width(lm)

                # Map index fingertip from camera space → canvas space
                idx_x, idx_y = lm[8]
                idx_canvas_x = int(idx_x / w * CANVAS_W)
                idx_canvas_y = int(idx_y / h * CANVAS_H)
                index_tip_canvas_pos = (idx_canvas_x, idx_canvas_y)

                # Strict pinch detection using adaptive threshold
                is_draw_pinch = is_strict_pinch(lm, 4, 8, palm_width)          # thumb-index
                is_index_middle_pinch = is_strict_pinch(lm, 8, 12, palm_width)
                is_index_ring_pinch = is_strict_pinch(lm, 8, 16, palm_width)

                # Draw gesture: any hand, thumb-index pinch
                if is_draw_pinch:
                    draw_gesture_present = True

                # Erase gesture: LEFT hand index-middle pinch (hold)
                if handedness == "Left" and is_index_middle_pinch:
                    erase_gesture_present = True

                # Undo gesture: RIGHT hand index-middle pinch (hold)
                if handedness == "Right" and is_index_middle_pinch:
                    undo_gesture_present = True

                # Redo gesture: RIGHT hand index-ring pinch (hold)
                if handedness == "Right" and is_index_ring_pinch:
                    redo_gesture_present = True

        # ---- Enforce mutual exclusivity with priority ----
        # Priority: erase > undo > redo > draw
        current_intent = None

        if erase_gesture_present:
            current_intent = "erase"
            draw_gesture_present = False
            undo_gesture_present = False
            redo_gesture_present = False
        elif undo_gesture_present:
            current_intent = "undo"
            draw_gesture_present = False
            erase_gesture_present = False
            redo_gesture_present = False
        elif redo_gesture_present:
            current_intent = "redo"
            draw_gesture_present = False
            erase_gesture_present = False
            undo_gesture_present = False
        elif draw_gesture_present:
            current_intent = "draw"
        else:
            current_intent = None

        # ---- Gesture timing logic with exclusivity ----

        # Draw activation
        if current_intent == "draw":
            draw_hold_frames += 1
        else:
            draw_hold_frames = 0
            drawing_active = False
            # close stroke if we had one
            if current_stroke:
                strokes.append(current_stroke)
                current_stroke = []
                undone_strokes.clear()  # clear redo stack when new stroke is committed

        if draw_hold_frames >= DRAW_ACTIVATE_FRAMES:
            drawing_active = True

        # Erase (clear all strokes) when left-hand gesture held
        if current_intent == "erase":
            erase_hold_frames += 1
        else:
            erase_hold_frames = 0

        if erase_hold_frames == ERASE_ACTIVATE_FRAMES:
            # trigger erase once per gesture hold
            strokes.clear()
            undone_strokes.clear()
            current_stroke = []
            drawing_active = False
            smoothed_point = None
            print("[ACTION] Erase all strokes")

        # Undo
        if current_intent == "undo":
            undo_hold_frames += 1
        else:
            undo_hold_frames = 0

        if undo_hold_frames == UNDO_ACTIVATE_FRAMES:
            if strokes:
                last = strokes.pop()
                undone_strokes.append(last)
                print("[ACTION] Undo")

        # Redo
        if current_intent == "redo":
            redo_hold_frames += 1
        else:
            redo_hold_frames = 0

        if redo_hold_frames == REDO_ACTIVATE_FRAMES:
            if undone_strokes:
                restored = undone_strokes.pop()
                strokes.append(restored)
                print("[ACTION] Redo")

        # ------ Drawing with smoothing ------

        if drawing_active and index_tip_canvas_pos is not None:
            if smoothed_point is None:
                smoothed_point = index_tip_canvas_pos
            else:
                # simple exponential smoothing for smoother strokes
                alpha = 0.3
                sx = int((1 - alpha) * smoothed_point[0] + alpha * index_tip_canvas_pos[0])
                sy = int((1 - alpha) * smoothed_point[1] + alpha * index_tip_canvas_pos[1])
                smoothed_point = (sx, sy)

            # append smoothed point to current stroke
            current_stroke.append(smoothed_point)
        else:
            smoothed_point = None

        # Re-draw strokes including current_stroke
        for stroke in strokes:
            for i in range(1, len(stroke)):
                cv2.line(canvas_base, stroke[i - 1], stroke[i], (0, 0, 0), 5)
        if current_stroke:
            for i in range(1, len(current_stroke)):
                cv2.line(canvas_base, current_stroke[i - 1], current_stroke[i], (0, 0, 0), 5)

        # ------ Build display canvas (pointer + instructions + mode) ------

        display_canvas = canvas_base.copy()

        # Pointer: only visual, not permanent; color depends on mode
        if index_tip_canvas_pos is not None:
            color = (150, 0, 150)   # idle default
            if current_intent == "draw":
                color = (0, 200, 0)
            elif current_intent == "erase":
                color = (0, 0, 255)
            elif current_intent == "undo":
                color = (0, 165, 255)  # orange
            elif current_intent == "redo":
                color = (255, 0, 0)

            cv2.circle(display_canvas, index_tip_canvas_pos, 7, color, -1)

        # Instructions overlay
        instructions = [
            "Draw: pinch Thumb + Index (hold)",
            "Erase ALL: LEFT hand pinch Index + Middle (hold)",
            "Undo: RIGHT hand pinch Index + Middle (hold)",
            "Redo: RIGHT hand pinch Index + Ring (hold)",
            "Tips: Good lighting improves accuracy",
            "Keys: Q=quit, C=clear, S=save"
        ]
        y0 = 20
        for line in instructions:
            cv2.putText(display_canvas, line, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
            y0 += 18

        # Current mode label
        mode_text = "MODE: IDLE"
        mode_color = (128, 128, 128)
        if current_intent == "draw":
            mode_text = "MODE: DRAW"
            mode_color = (0, 200, 0)
        elif current_intent == "erase":
            mode_text = "MODE: ERASE"
            mode_color = (0, 0, 255)
        elif current_intent == "undo":
            mode_text = "MODE: UNDO"
            mode_color = (0, 165, 255)
        elif current_intent == "redo":
            mode_text = "MODE: REDO"
            mode_color = (255, 0, 0)

        cv2.putText(display_canvas, mode_text, (10, CANVAS_H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

        # ------ Show windows ------

        hand_small = cv2.resize(hand_view, (320, 240))
        cv2.imshow("Hand View", hand_small)
        cv2.imshow("AirType Canvas", display_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            strokes.clear()
            undone_strokes.clear()
            current_stroke = []
            drawing_active = False
            smoothed_point = None
            print("[ACTION] Clear canvas (keyboard)")
        if key == ord("s"):
            cv2.imwrite("airtype_drawing.png", canvas_base)
            print("[INFO] Canvas saved as airtype_drawing.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
