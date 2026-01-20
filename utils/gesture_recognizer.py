# utils/gesture_recognizer.py

import math
from collections import deque

class GestureRecognizer:
    def __init__(self, history_len=10):
        # store index fingertip positions for dynamic gestures
        self.index_history = deque(maxlen=history_len)

        # Mediapipe landmark indices
        self.TIP_IDS = [4, 8, 12, 16, 20]
        self.FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    @staticmethod
    def distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def get_finger_states(self, hand_data):
        """
        Returns list of 0/1 for [thumb, index, middle, ring, pinky]
        1 = up/extended, 0 = down
        """
        lm = {id: (x, y) for id, x, y in hand_data["landmarks"]}

        fingers = []

        # Thumb: compare x coordinates depending on handedness
        thumb_tip = lm[4]
        thumb_ip = lm[3]
        if hand_data["handedness"] == "Right":
            fingers.append(1 if thumb_tip[0] > thumb_ip[0] else 0)
        else:
            fingers.append(1 if thumb_tip[0] < thumb_ip[0] else 0)

        # Other fingers: compare y of tip vs pip (tip higher → up)
        for tip_id, pip_id in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            tip_y = lm[tip_id][1]
            pip_y = lm[pip_id][1]
            fingers.append(1 if tip_y < pip_y else 0)

        return fingers

    def get_static_gesture(self, finger_states):
        """
        Return a string name for some basic gestures.
        """
        f = finger_states  # [thumb, index, middle, ring, pinky]

        if f == [0, 0, 0, 0, 0]:
            return "Fist"
        if f == [1, 1, 1, 1, 1]:
            return "OpenPalm"
        if f == [0, 1, 0, 0, 0]:
            return "Point"
        if f == [0, 1, 1, 0, 0]:
            return "Peace"
        if f == [1, 0, 0, 0, 0]:
            return "ThumbsUp"

        return "Unknown"

    def update_index_history(self, hand_data):
        lm = {id: (x, y) for id, x, y in hand_data["landmarks"]}
        index_tip = lm[8]
        self.index_history.append(index_tip)

    def get_swipe(self, min_disp=80):
        """
        crude swipe detection using index history.
        returns 'SwipeLeft', 'SwipeRight', 'SwipeUp', 'SwipeDown', or None
        """
        if len(self.index_history) < 2:
            return None

        x0, y0 = self.index_history[0]
        x1, y1 = self.index_history[-1]
        dx = x1 - x0
        dy = y1 - y0

        if abs(dx) > abs(dy):
            if abs(dx) > min_disp:
                return "SwipeRight" if dx > 0 else "SwipeLeft"
        else:
            if abs(dy) > min_disp:
                return "SwipeDown" if dy > 0 else "SwipeUp"

        return None

    def is_pinch(self, hand_data, id1, id2, thresh=40):
        lm = {id: (x, y) for id, x, y in hand_data["landmarks"]}
        p1 = lm[id1]
        p2 = lm[id2]
        d = self.distance(p1, p2)
        return d < thresh
