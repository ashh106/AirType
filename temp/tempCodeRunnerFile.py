# import cv2
# import mediapipe as mp
# import numpy as np
# import math
# import time
# import os

# # =========================
# # Helper math functions
# # =========================

# def dist(p1, p2):
#     return math.dist(p1, p2)

# def snap_to_grid(point, grid_size=4):
#     x, y = point
#     x = int(round(x / grid_size) * grid_size)
#     y = int(round(y / grid_size) * grid_size)
#     return (x, y)

# def smooth_point(prev_point, new_point, alpha=0.7):
#     if prev_point is None:
#         return new_point
#     px, py = prev_point
#     nx, ny = new_point
#     sx = int(alpha * px + (1 - alpha) * nx)
#     sy = int(alpha * py + (1 - alpha) * ny)
#     return (sx, sy)

# # =========================
# # Shape recognition
# # =========================

# def recognize_shape(points, min_size=40):
#     """
#     Very simple heuristic:
#     - takes a stroke (list of (x,y))
#     - approx as circle or rectangle if it looks like it
#     """
#     if len(points) < 10:
#         return points  # too small to classify

#     xs = [p[0] for p in points]
#     ys = [p[1] for p in points]
#     minx, maxx = min(xs), max(xs)
#     miny, maxy = min(ys), max(ys)

#     w = maxx - minx
#     h = maxy - miny

#     if w < min_size or h < min_size:
#         return points  # ignore tiny strokes

#     # approximate perimeter
#     perim = 0.0
#     for i in range(1, len(points)):
#         perim += dist(points[i - 1], points[i])
#     if perim == 0:
#         return points

#     area_bbox = w * h
#     circularity = 4 * math.pi * area_bbox / (perim ** 2 + 1e-6)
#     aspect = w / (h + 1e-6)

#     # Circle-ish: nearly square + relatively high circularity
#     if 0.8 <= aspect <= 1.25 and circularity > 2.5:
#         cx = (minx + maxx) // 2
#         cy = (miny + maxy) // 2
#         r = int((w + h) / 4)
#         new_pts = []
#         for a in range(0, 360, 5):
#             rad = math.radians(a)
#             x = cx + int(r * math.cos(rad))
#             y = cy + int(r * math.sin(rad))
#             new_pts.append((x, y))
#         return new_pts

#     # Rectangle-ish: not very circular
#     if circularity < 2.0:
#         new_pts = [
#             (minx, miny),
#             (maxx, miny),
#             (maxx, maxy),
#             (minx, maxy),
#             (minx, miny)
#         ]
#         return new_pts

#     return points  # default: leave as is

# # =========================
# # Main AirDraw class
# # =========================

# class AirDrawApp:
#     def __init__(self):
#         self.cap = cv2.VideoCapture(0)
#         self.mpHands = mp.solutions.hands
#         self.hands = self.mpHands.Hands(
#             max_num_hands=2,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.7
#         )
#         self.mpDraw = mp.solutions.drawing_utils

#         # strokes: list of { "points": [(x,y)...], "color":(b,g,r), "thickness":int }
#         self.strokes = []
#         self.current_stroke = None

#         self.canvas = None
#         self.brush_thickness = 4
#         self.min_thickness = 2
#         self.max_thickness = 25

#         self.colors = [
#             (0, 0, 255),    # red
#             (0, 255, 0),    # green
#             (255, 0, 0),    # blue
#             (0, 255, 255),  # yellow
#             (255, 0, 255),  # magenta
#             (255, 255, 255) # white
#         ]
#         self.color_index = 0

#         # States
#         self.drawing = False
#         self.erasing = False
#         self.eraser_size = 35
#         self.last_point = None

#         # Gesture timers / flags
#         self.clear_frames = 0
#         self.clear_threshold = 60  # ~2 sec at 30fps
#         self.save_cooldown_frames = 0

#         self.thickness_pinch_prev = False
#         self.color_select_mode = False
#         self.color_base_y = None

#     # ------------- pinch helper -------------

#     def is_pinch(self, lm, id1, id2, w, h, threshold=40):
#         x1 = int(lm[id1].x * w)
#         y1 = int(lm[id1].y * h)
#         x2 = int(lm[id2].x * w)
#         y2 = int(lm[id2].y * h)
#         d = math.dist((x1, y1), (x2, y2))
#         return d < threshold, (x1, y1), (x2, y2)

#     # ------------- finger / palm helpers -------------

#     def fingers_up(self, lm, handedness):
#         """
#         Very basic finger detection.
#         Returns list [thumb, index, middle, ring, pinky] as 1 (up) / 0 (down)
#         """
#         tips = [4, 8, 12, 16, 20]
#         pips = [3, 6, 10, 14, 18]

#         fingers = []

#         # Thumb: depends on left/right; use x instead of y
#         if handedness == "Right":
#             fingers.append(1 if lm[tips[0]].x < lm[pips[0]].x else 0)
#         else:
#             fingers.append(1 if lm[tips[0]].x > lm[pips[0]].x else 0)

#         # Other fingers: tip.y < pip.y -> up
#         for i in range(1, 5):
#             fingers.append(1 if lm[tips[i]].y < lm[pips[i]].y else 0)

#         return fingers

#     def is_palm_open(self, lm, handedness):
#         fingers = self.fingers_up(lm, handedness)
#         # at least 4 fingers up
#         return sum(fingers[1:]) >= 4

#     def is_thumbs_up(self, lm, handedness):
#         fingers = self.fingers_up(lm, handedness)
#         thumb, idx, mid, ring, pinky = fingers
#         # thumb up, all other fingers down
#         if thumb == 1 and (idx + mid + ring + pinky) == 0:
#             return True
#         return False

#     # ------------- drawing logic -------------

#     def start_new_stroke(self, start_point):
#         self.current_stroke = {
#             "points": [start_point],
#             "color": self.colors[self.color_index],
#             "thickness": self.brush_thickness
#         }
#         self.last_point = start_point

#     def add_point_to_stroke(self, raw_point):
#         if self.current_stroke is None:
#             return  # avoid crash
#         smoothed = smooth_point(self.last_point, raw_point)
#         snapped = snap_to_grid(smoothed)
#         self.current_stroke["points"].append(snapped)
#         self.last_point = snapped

#     def end_stroke(self):
#         if self.current_stroke and len(self.current_stroke["points"]) > 1:
#             # shape recognition
#             pts = self.current_stroke["points"]
#             pts = recognize_shape(pts)
#             self.current_stroke["points"] = pts
#             self.strokes.append(self.current_stroke)
#         self.current_stroke = None
#         self.last_point = None

#     def erase_at(self, x, y):
#         new_strokes = []
#         for stroke in self.strokes:
#             new_points = []
#             for p in stroke["points"]:
#                 if dist(p, (x, y)) > self.eraser_size:
#                     new_points.append(p)
#             if len(new_points) > 1:
#                 stroke["points"] = new_points
#                 new_strokes.append(stroke)
#         self.strokes = new_strokes

#         # also erase from current stroke if drawing
#         if self.current_stroke is not None:
#             new_points = []
#             for p in self.current_stroke["points"]:
#                 if dist(p, (x, y)) > self.eraser_size:
#                     new_points.append(p)
#             if len(new_points) > 1:
#                 self.current_stroke["points"] = new_points
#             else:
#                 self.current_stroke = None

#     # ------------- UI / overlay -------------

#     def draw_strokes_on_canvas(self):
#         if self.canvas is None:
#             return
#         self.canvas[:] = 0
#         for stroke in self.strokes:
#             pts = stroke["points"]
#             if len(pts) < 2:
#                 continue
#             for i in range(1, len(pts)):
#                 cv2.line(
#                     self.canvas,
#                     pts[i - 1],
#                     pts[i],
#                     stroke["color"],
#                     stroke["thickness"]
#                 )
#         if self.current_stroke is not None:
#             pts = self.current_stroke["points"]
#             for i in range(1, len(pts)):
#                 cv2.line(
#                     self.canvas,
#                     pts[i - 1],
#                     pts[i],
#                     self.current_stroke["color"],
#                     self.current_stroke["thickness"]
#                 )

#     def save_canvas(self):
#         if self.canvas is None:
#             return
#         folder = "airdraw_saves"
#         os.makedirs(folder, exist_ok=True)
#         filename = time.strftime("%Y%m%d_%H%M%S") + ".png"
#         path = os.path.join(folder, filename)
#         # save only the canvas (no camera) on white background
#         h, w, _ = self.canvas.shape
#         white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
#         # strokes are drawn on black canvas; make them visible on white
#         mask = self.canvas > 0
#         white_bg[mask] = self.canvas[mask]
#         cv2.imwrite(path, white_bg)
#         print(f"Saved drawing to {path}")

#     # ------------- main processing -------------

#     def process_hand(self, img, handLms, handedness_label):
#         h, w, _ = img.shape
#         lm = handLms.landmark

#         # Basic fingertip coordinates
#         ix = int(lm[8].x * w)
#         iy = int(lm[8].y * h)

#         # -------- Gestures --------

#         # 1) Clear screen: open palm for some frames
#         if self.is_palm_open(lm, handedness_label):
#             self.clear_frames += 1
#             if self.clear_frames == self.clear_threshold:
#                 self.strokes = []
#                 self.current_stroke = None
#                 self.last_point = None
#                 print("Canvas cleared")
#         else:
#             self.clear_frames = 0

#         # 2) Save drawing: thumbs up (with cooldown)
#         if self.save_cooldown_frames > 0:
#             self.save_cooldown_frames -= 1

#         if self.is_thumbs_up(lm, handedness_label) and self.save_cooldown_frames == 0:
#             self.save_canvas()
#             self.save_cooldown_frames = 60  # ~2 sec cooldown

#         # 3) Thickness control: right hand middle + ring
#         if handedness_label == "Right":
#             thick_pinch, _, _ = self.is_pinch(lm, 12, 16, w, h, threshold=35)
#             if thick_pinch and not self.thickness_pinch_prev:
#                 # on pinch "edge" increase thickness
#                 self.brush_thickness = min(self.brush_thickness + 2, self.max_thickness)
#             self.thickness_pinch_prev = thick_pinch
#         else:
#             self.thickness_pinch_prev = False

#         # 4) Color selection: right hand thumb + middle pinch, move up/down
#         if handedness_label == "Right":
#             color_pinch, _, _ = self.is_pinch(lm, 4, 12, w, h, threshold=35)
#             if color_pinch:
#                 if not self.color_select_mode:
#                     self.color_select_mode = True
#                     self.color_base_y = iy
#                 else:
#                     dy = iy - self.color_base_y
#                     # move up -> next color, move down -> previous color
#                     if dy < -40:  # moved up
#                         self.color_index = (self.color_index + 1) % len(self.colors)
#                         self.color_base_y = iy
#                         print("Color changed to index", self.color_index)
#                     elif dy > 40:  # moved down
#                         self.color_index = (self.color_index - 1) % len(self.colors)
#                         self.color_base_y = iy
#                         print("Color changed to index", self.color_index)
#             else:
#                 self.color_select_mode = False
#                 self.color_base_y = None

#         # 5) Eraser: left hand index + middle pinch
#         if handedness_label == "Left":
#             erase_pinch, _, _ = self.is_pinch(lm, 8, 12, w, h, threshold=40)
#             self.erasing = erase_pinch
#             if self.erasing:
#                 # visible eraser
#                 cv2.circle(img, (ix, iy), self.eraser_size, (255, 255, 255), 2)
#                 self.erase_at(ix, iy)
#         # only left hand controls eraser
#         if handedness_label == "Right":
#             # do not let right hand accidentally overwrite erasing
#             pass

#         # 6) Drawing: right hand thumb + index pinch
#         if handedness_label == "Right" and not self.color_select_mode:
#             draw_pinch, _, _ = self.is_pinch(lm, 4, 8, w, h, threshold=40)

#             if draw_pinch and not self.drawing:
#                 # Start drawing
#                 start_pt = snap_to_grid((ix, iy))
#                 self.start_new_stroke(start_pt)
#                 self.drawing = True

#             elif draw_pinch and self.drawing:
#                 if self.current_stroke is not None:
#                    self.add_point_to_stroke((ix, iy))
#                 else:
#                       # stroke was cleared or ended, restart safely
#                       start_pt = snap_to_grid((ix, iy))
#                       self.start_new_stroke(start_pt)
#                       self.drawing = True


#             elif not draw_pinch and self.drawing:
#                 # Finish stroke when fingers separate
#                 self.end_stroke()
#                 self.drawing = False

#         # if it's left hand or no draw pinch: ensure if drawing state gets off when needed
#         if handedness_label == "Left":
#             # ensure left hand doesn't cause drawing
#             pass

#     def run(self):
#         if not self.cap.isOpened():
#             print("Could not open webcam")
#             return

#         first_frame = True

#         while True:
#             success, img = self.cap.read()
#             if not success:
#                 break

#             img = cv2.flip(img, 1)

#             if first_frame:
#                 h, w, _ = img.shape
#                 self.canvas = np.zeros_like(img)
#                 first_frame = False

#             # MediaPipe processing
#             imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             results = self.hands.process(imgRGB)

#             if results.multi_hand_landmarks:
#                 for idx, handLms in enumerate(results.multi_hand_landmarks):
#                     handedness_label = "Right"
#                     if results.multi_handedness and len(results.multi_handedness) > idx:
#                         handedness_label = results.multi_handedness[idx].classification[0].label

#                     # draw landmarks for debug
#                     self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

#                     # process gestures / drawing for each hand
#                     self.process_hand(img, handLms, handedness_label)

#             # Draw strokes on separate canvas
#             self.draw_strokes_on_canvas()

#             # Overlay canvas on camera
#             if self.canvas is not None:
#                 output = cv2.addWeighted(img, 0.5, self.canvas, 0.8, 0)
#             else:
#                 output = img

#             # Small HUD
#             cv2.rectangle(output, (0, 0), (260, 110), (0, 0, 0), -1)
#             cv2.putText(output, f"Color: {self.color_index}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#             cv2.putText(output, f"Thick: {self.brush_thickness}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#             cv2.putText(output, "Draw: R thumb+index", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
#             cv2.putText(output, "Erase: L index+middle", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

#             cv2.imshow("AirDraw", output)
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break

#         self.cap.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     app = AirDrawApp()
#     app.run()


# # part 3 ----------------------------------------------




# import cv2
# import mediapipe as mp
# import numpy as np
# import math
# import time
# import os

# # =========================
# # Helper math functions
# # =========================

# def dist(p1, p2):
#     return math.dist(p1, p2)

# def snap_to_grid(point, grid_size=4):
#     x, y = point
#     x = int(round(x / grid_size) * grid_size)
#     y = int(round(y / grid_size) * grid_size)
#     return (x, y)

# def smooth_point(prev_point, new_point, alpha=0.7):
#     if prev_point is None:
#         return new_point
#     px, py = prev_point
#     nx, ny = new_point
#     sx = int(alpha * px + (1 - alpha) * nx)
#     sy = int(alpha * py + (1 - alpha) * ny)
#     return (sx, sy)

# def make_rect_points(minx, miny, maxx, maxy):
#     return [
#         (minx, miny),
#         (maxx, miny),
#         (maxx, maxy),
#         (minx, maxy),
#         (minx, miny)
#     ]

# def make_circle_points(minx, miny, maxx, maxy, step_deg=5):
#     cx = (minx + maxx) // 2
#     cy = (miny + maxy) // 2
#     r = int((maxx - minx + maxy - miny) / 4)
#     pts = []
#     for a in range(0, 360, step_deg):
#         rad = math.radians(a)
#         x = cx + int(r * math.cos(rad))
#         y = cy + int(r * math.sin(rad))
#         pts.append((x, y))
#     return pts

# # =========================
# # Main app
# # =========================

# class AirDrawApp:
#     def __init__(self):
#         self.cap = cv2.VideoCapture(0)
#         self.mpHands = mp.solutions.hands
#         self.hands = self.mpHands.Hands(
#             max_num_hands=2,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.7
#         )
#         self.mpDraw = mp.solutions.drawing_utils

#         # Drawing data
#         self.strokes = []       # list of { "points": [...], "color":(b,g,r), "thickness":int }
#         self.redo_stack = []    # for redo
#         self.current_stroke = None

#         self.canvas = None

#         # Brush / color
#         self.brush_thickness = 4
#         self.min_thickness = 2
#         self.max_thickness = 25

#         self.colors = [
#             (0, 0, 255),     # red
#             (0, 255, 0),     # green
#             (255, 0, 0),     # blue
#             (0, 255, 255),   # yellow
#             (255, 0, 255),   # magenta
#             (255, 255, 255)  # white
#         ]
#         self.color_index = 0

#         # Modes / states
#         self.active_tool = "brush"   # "brush", "eraser", "circle", "rect"
#         self.eraser_size = 35
#         self.drawing = False
#         self.last_point = None

#         # UI layout
#         self.toolbar_height = 80
#         self.color_bar_height = 30

#         # UI click handling
#         self.ui_pinch_prev = False

#     # ------------- basic helpers -------------

#     def is_pinch(self, lm, id1, id2, w, h, threshold=40):
#         x1 = int(lm[id1].x * w)
#         y1 = int(lm[id1].y * h)
#         x2 = int(lm[id2].x * w)
#         y2 = int(lm[id2].y * h)
#         d = math.dist((x1, y1), (x2, y2))
#         return d < threshold, (x1, y1), (x2, y2)

#     # ------------- strokes -------------

#     def start_new_stroke(self, start_point):
#         self.current_stroke = {
#             "points": [start_point],
#             "color": self.colors[self.color_index],
#             "thickness": self.brush_thickness
#         }
#         self.last_point = start_point

#     def add_point_to_stroke(self, raw_point):
#         if self.current_stroke is None:
#             return
#         smoothed = smooth_point(self.last_point, raw_point)
#         snapped = snap_to_grid(smoothed)
#         self.current_stroke["points"].append(snapped)
#         self.last_point = snapped

#     def end_stroke(self):
#         if self.current_stroke is None:
#             return
#         pts = self.current_stroke["points"]
#         if len(pts) > 1:
#             xs = [p[0] for p in pts]
#             ys = [p[1] for p in pts]
#             minx, maxx = min(xs), max(xs)
#             miny, maxy = min(ys), max(ys)

#             # shape modes
#             if self.active_tool == "circle":
#                 pts = make_circle_points(minx, miny, maxx, maxy)
#             elif self.active_tool == "rect":
#                 pts = make_rect_points(minx, miny, maxx, maxy)

#             self.current_stroke["points"] = pts
#             self.strokes.append(self.current_stroke)
#             # New stroke invalidates redo stack
#             self.redo_stack = []

#         self.current_stroke = None
#         self.last_point = None

#     def erase_at(self, x, y):
#         p = (x, y)
#         new_strokes = []
#         for stroke in self.strokes:
#             new_points = [pt for pt in stroke["points"] if dist(pt, p) > self.eraser_size]
#             if len(new_points) > 1:
#                 stroke["points"] = new_points
#                 new_strokes.append(stroke)
#         self.strokes = new_strokes

#         # Also impact current stroke if any
#         if self.current_stroke is not None:
#             new_points = [pt for pt in self.current_stroke["points"] if dist(pt, p) > self.eraser_size]
#             if len(new_points) > 1:
#                 self.current_stroke["points"] = new_points
#             else:
#                 self.current_stroke = None
#                 self.drawing = False

#     # ------------- undo / redo / clear / save -------------

#     def undo(self):
#         # cancel current drawing if any
#         if self.current_stroke is not None and self.drawing:
#             self.current_stroke = None
#             self.drawing = False
#             return

#         if self.strokes:
#             stroke = self.strokes.pop()
#             self.redo_stack.append(stroke)
#             print("Undo")

#     def redo(self):
#         if self.redo_stack:
#             stroke = self.redo_stack.pop()
#             self.strokes.append(stroke)
#             print("Redo")

#     def clear_all(self):
#         self.strokes = []
#         self.redo_stack = []
#         self.current_stroke = None
#         self.drawing = False
#         print("Canvas cleared")

#     def save_canvas(self):
#         if self.canvas is None:
#             return
#         folder = "airdraw_saves"
#         os.makedirs(folder, exist_ok=True)
#         filename = time.strftime("%Y%m%d_%H%M%S") + ".png"
#         path = os.path.join(folder, filename)
#         h, w, _ = self.canvas.shape
#         white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
#         mask = self.canvas > 0
#         white_bg[mask] = self.canvas[mask]
#         cv2.imwrite(path, white_bg)
#         print(f"Saved drawing to {path}")

#     # ------------- drawing canvas -------------

#     def draw_strokes_on_canvas(self):
#         if self.canvas is None:
#             return
#         self.canvas[:] = 0
#         # existing strokes
#         for stroke in self.strokes:
#             pts = stroke["points"]
#             if len(pts) < 2:
#                 continue
#             for i in range(1, len(pts)):
#                 cv2.line(self.canvas, pts[i - 1], pts[i], stroke["color"], stroke["thickness"])
#         # current stroke
#         if self.current_stroke is not None:
#             pts = self.current_stroke["points"]
#             for i in range(1, len(pts)):
#                 cv2.line(
#                     self.canvas,
#                     pts[i - 1],
#                     pts[i],
#                     self.current_stroke["color"],
#                     self.current_stroke["thickness"]
#                 )

#     # ------------- UI (toolbar + colors) -------------

#     def get_toolbar_layout(self, w, h):
#         """
#         Returns a list of buttons with rects:
#         [ {id, label, x1, y1, x2, y2}, ... ]
#         """
#         toolbar_top = h - self.toolbar_height
#         tool_ids = ["brush", "eraser", "circle", "rect", "thin", "thick", "clear", "save"]
#         labels = ["Brush", "Eraser", "Circle", "Rect", "-", "+", "Clear", "Save"]
#         n = len(tool_ids)
#         cell_w = int(w / n)

#         buttons = []
#         for i, (tid, label) in enumerate(zip(tool_ids, labels)):
#             x1 = i * cell_w + 5
#             x2 = (i + 1) * cell_w - 5
#             y1 = toolbar_top + 5
#             y2 = h - 5
#             buttons.append({
#                 "id": tid,
#                 "label": label,
#                 "x1": x1,
#                 "y1": y1,
#                 "x2": x2,
#                 "y2": y2
#             })
#         return buttons

#     def get_color_layout(self, w, h):
#         """
#         Color bar above toolbar: returns list of color buttons
#         """
#         toolbar_top = h - self.toolbar_height
#         color_top = toolbar_top - self.color_bar_height - 5
#         color_bottom = toolbar_top - 5
#         n = len(self.colors)
#         cell_w = int(w / n)

#         buttons = []
#         for i in range(n):
#             x1 = i * cell_w + 10
#             x2 = (i + 1) * cell_w - 10
#             y1 = color_top
#             y2 = color_bottom
#             buttons.append({
#                 "index": i,
#                 "x1": x1,
#                 "y1": y1,
#                 "x2": x2,
#                 "y2": y2
#             })
#         return buttons

#     def draw_toolbar_and_colors(self, frame):
#         h, w, _ = frame.shape
#         toolbar_top = h - self.toolbar_height
#         color_buttons = self.get_color_layout(w, h)
#         tool_buttons = self.get_toolbar_layout(w, h)

#         # Background for toolbar
#         cv2.rectangle(frame, (0, toolbar_top), (w, h), (20, 20, 20), -1)

#         # Color bar background
#         if color_buttons:
#             y1 = color_buttons[0]["y1"] - 2
#             y2 = color_buttons[0]["y2"] + 2
#             cv2.rectangle(frame, (0, y1), (w, y2), (40, 40, 40), -1)

#         # Draw color buttons
#         for cb in color_buttons:
#             color = self.colors[cb["index"]]
#             selected = (cb["index"] == self.color_index)
#             thickness = 3 if selected else 1
#             cv2.rectangle(
#                 frame,
#                 (cb["x1"], cb["y1"]),
#                 (cb["x2"], cb["y2"]),
#                 color,
#                 -1
#             )
#             cv2.rectangle(
#                 frame,
#                 (cb["x1"], cb["y1"]),
#                 (cb["x2"], cb["y2"]),
#                 (255, 255, 255),
#                 thickness
#             )

#         # Draw tool buttons
#         for tb in tool_buttons:
#             is_active = (
#                 (tb["id"] == self.active_tool) or
#                 (tb["id"] == "thin") or
#                 (tb["id"] == "thick")
#             )
#             # box
#             cv2.rectangle(
#                 frame,
#                 (tb["x1"], tb["y1"]),
#                 (tb["x2"], tb["y2"]),
#                 (80, 80, 80) if is_active else (60, 60, 60),
#                 -1
#             )
#             cv2.rectangle(
#                 frame,
#                 (tb["x1"], tb["y1"]),
#                 (tb["x2"], tb["y2"]),
#                 (200, 200, 200),
#                 1
#             )
#             # text
#             cv2.putText(
#                 frame,
#                 tb["label"],
#                 (tb["x1"] + 5, tb["y1"] + 25),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (255, 255, 255),
#                 1
#             )

#         # Small status text on top-left
#         cv2.rectangle(frame, (0, 0), (260, 60), (0, 0, 0), -1)
#         cv2.putText(
#             frame, f"Tool: {self.active_tool}",
#             (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
#         )
#         cv2.putText(
#             frame, f"Thick: {self.brush_thickness}",
#             (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
#         )

#     def handle_ui_click(self, x, y, w, h):
#         toolbar_top = h - self.toolbar_height
#         color_buttons = self.get_color_layout(w, h)
#         tool_buttons = self.get_toolbar_layout(w, h)

#         # Check color bar first
#         for cb in color_buttons:
#             if cb["x1"] <= x <= cb["x2"] and cb["y1"] <= y <= cb["y2"]:
#                 self.color_index = cb["index"]
#                 print("Color changed to index", self.color_index)
#                 return

#         # Then tools
#         if y >= toolbar_top:
#             for tb in tool_buttons:
#                 if tb["x1"] <= x <= tb["x2"] and tb["y1"] <= y <= tb["y2"]:
#                     tid = tb["id"]
#                     if tid in ["brush", "eraser", "circle", "rect"]:
#                         self.active_tool = tid
#                         print("Tool:", tid)
#                     elif tid == "thin":
#                         self.brush_thickness = max(self.min_thickness, self.brush_thickness - 2)
#                         print("Thickness:", self.brush_thickness)
#                     elif tid == "thick":
#                         self.brush_thickness = min(self.max_thickness, self.brush_thickness + 2)
#                         print("Thickness:", self.brush_thickness)
#                     elif tid == "clear":
#                         self.clear_all()
#                     elif tid == "save":
#                         self.save_canvas()
#                     return

#     # ------------- hand processing -------------

#     def process_left_hand(self, img, handLms, handedness_label):
#         h, w, _ = img.shape
#         lm = handLms.landmark
#         ix = int(lm[8].x * w)
#         iy = int(lm[8].y * h)

#         # left hand eraser: index + middle pinch
#         erase_pinch, _, _ = self.is_pinch(lm, 8, 12, w, h, threshold=40)
#         if erase_pinch:
#             cv2.circle(img, (ix, iy), self.eraser_size, (255, 255, 255), 2)
#             self.erase_at(ix, iy)

#     def process_right_hand(self, img, handLms, handedness_label):
#         h, w, _ = img.shape
#         lm = handLms.landmark
#         ix = int(lm[8].x * w)
#         iy = int(lm[8].y * h)

#         toolbar_top = h - self.toolbar_height
#         color_top = toolbar_top - self.color_bar_height - 5

#         draw_pinch, _, _ = self.is_pinch(lm, 4, 8, w, h, threshold=40)

#         if draw_pinch:
#             in_color_bar = color_top <= iy < toolbar_top
#             in_toolbar = iy >= toolbar_top

#             # If pointing at UI area → treat as click
#             if in_color_bar or in_toolbar:
#                 if not self.ui_pinch_prev:
#                     self.handle_ui_click(ix, iy, w, h)
#                 self.ui_pinch_prev = True

#                 # If we were drawing, end stroke when going into UI
#                 if self.drawing:
#                     self.end_stroke()
#                     self.drawing = False
#                 return
#             else:
#                 # Drawing region
#                 self.ui_pinch_prev = False

#                 # If eraser tool selected, erase with right hand too
#                 if self.active_tool == "eraser":
#                     self.erase_at(ix, iy)
#                     return

#                 # Brush / circle / rect draw
#                 if not self.drawing:
#                     start_pt = snap_to_grid((ix, iy))
#                     self.start_new_stroke(start_pt)
#                     self.drawing = True
#                 else:
#                     if self.current_stroke is not None:
#                         self.add_point_to_stroke((ix, iy))
#         else:
#             # no pinch
#             self.ui_pinch_prev = False
#             if self.drawing:
#                 self.end_stroke()
#                 self.drawing = False

#     # ------------- main loop -------------

#     def run(self):
#         if not self.cap.isOpened():
#             print("Could not open webcam")
#             return

#         first_frame = True

#         while True:
#             success, img = self.cap.read()
#             if not success:
#                 break

#             img = cv2.flip(img, 1)

#             if first_frame:
#                 h, w, _ = img.shape
#                 self.canvas = np.zeros_like(img)
#                 first_frame = False

#             imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             results = self.hands.process(imgRGB)

#             if results.multi_hand_landmarks:
#                 for idx, handLms in enumerate(results.multi_hand_landmarks):
#                     handedness_label = "Right"
#                     if results.multi_handedness and len(results.multi_handedness) > idx:
#                         handedness_label = results.multi_handedness[idx].classification[0].label

#                     self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

#                     if handedness_label == "Left":
#                         self.process_left_hand(img, handLms, handedness_label)
#                     else:
#                         self.process_right_hand(img, handLms, handedness_label)

#             # Draw strokes
#             self.draw_strokes_on_canvas()

#             # Combine camera + canvas
#             if self.canvas is not None:
#                 output = cv2.addWeighted(img, 0.5, self.canvas, 0.8, 0)
#             else:
#                 output = img

#             # Draw toolbar and colors
#             self.draw_toolbar_and_colors(output)

#             cv2.imshow("AirDraw", output)

#             key = cv2.waitKey(1) & 0xFF
#             # Quit
#             if key in (27, ord('q')):  # ESC or q
#                 break
#             # Undo (Ctrl+Z or z)
#             if key in (26, ord('z')):
#                 self.undo()
#             # Redo (Ctrl+Y or y)
#             if key in (25, ord('y')):
#                 self.redo()
#             # Clear
#             if key == ord('c'):
#                 self.clear_all()
#             # Save
#             if key == ord('s'):
#                 self.save_canvas()

#         self.cap.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     app = AirDrawApp()
#     app.run()






# # part 4 ----------------------------------------------






import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import copy

# =========================
# Helper math functions
# =========================

def dist(p1, p2):
    return math.dist(p1, p2)

def snap_to_grid(point, grid_size=4):
    x, y = point
    x = int(round(x / grid_size) * grid_size)
    y = int(round(y / grid_size) * grid_size)
    return (x, y)

def smooth_point(prev_point, new_point, alpha=0.7):
    if prev_point is None:
        return new_point
    px, py = prev_point
    nx, ny = new_point
    sx = int(alpha * px + (1 - alpha) * nx)
    sy = int(alpha * py + (1 - alpha) * ny)
    return (sx, sy)

def make_rect_points(minx, miny, maxx, maxy):
    return [
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
        (minx, miny)
    ]

def make_circle_points(minx, miny, maxx, maxy, step_deg=5):
    cx = (minx + maxx) // 2
    cy = (miny + maxy) // 2
    r = int((maxx - minx + maxy - miny) / 4)
    pts = []
    for a in range(0, 360, step_deg):
        rad = math.radians(a)
        x = cx + int(r * math.cos(rad))
        y = cy + int(r * math.sin(rad))
        pts.append((x, y))
    return pts

# =========================
# Main app
# =========================

class AirDrawApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Drawing data
        self.strokes = []            # list of { "points": [...], "color":(b,g,r), "thickness":int }
        self.current_stroke = None

        # History for undo/redo (full canvas states)
        self.history = [copy.deepcopy(self.strokes)]
        self.history_pos = 0

        self.canvas = None

        # Brush / color
        self.brush_thickness = 4
        self.min_thickness = 2
        self.max_thickness = 25

        self.colors = [
            (0, 0, 255),     # red
            (0, 255, 0),     # green
            (255, 0, 0),     # blue
            (0, 255, 255),   # yellow
            (255, 0, 255),   # magenta
            (255, 255, 255)  # white
        ]
        self.color_index = 0

        # Modes / states
        self.active_tool = "brush"   # "brush", "eraser", "circle", "rect"
        self.eraser_size = 35
        self.drawing = False
        self.last_point = None

        # Erasing session tracking (for history)
        self.erasing_session_active = False

        # UI layout
        self.toolbar_height = 80
        self.color_bar_height = 30

        # UI hover / hold-selection
        self.ui_hover_target = None        # ("color", index) or ("tool", id)
        self.ui_hover_start_time = None
        self.ui_hold_confirmed = False
        self.ui_hold_duration = 1.0        # seconds

        # Grid overlay
        self.show_grid = False

    # ------------- history (undo / redo) -------------

    def push_history(self):
        # when we have a new state (after action), append snapshot
        if self.history_pos < len(self.history) - 1:
            self.history = self.history[:self.history_pos + 1]
        self.history.append(copy.deepcopy(self.strokes))
        self.history_pos += 1

        # limit history length
        max_len = 50
        if len(self.history) > max_len:
            excess = len(self.history) - max_len
            self.history = self.history[excess:]
            self.history_pos -= excess

    def undo(self):
        # move one step back in history
        if self.history_pos > 0:
            self.history_pos -= 1
            self.strokes = copy.deepcopy(self.history[self.history_pos])
            self.current_stroke = None
            self.drawing = False
            print("Undo")

    def redo(self):
        if self.history_pos < len(self.history) - 1:
            self.history_pos += 1
            self.strokes = copy.deepcopy(self.history[self.history_pos])
            self.current_stroke = None
            self.drawing = False
            print("Redo")

    # ------------- basic helpers -------------

    def is_pinch(self, lm, id1, id2, w, h, threshold=40):
        x1 = int(lm[id1].x * w)
        y1 = int(lm[id1].y * h)
        x2 = int(lm[id2].x * w)
        y2 = int(lm[id2].y * h)
        d = math.dist((x1, y1), (x2, y2))
        return d < threshold, (x1, y1), (x2, y2)

    # ------------- strokes -------------

    def start_new_stroke(self, start_point):
        self.current_stroke = {
            "points": [start_point],
            "color": self.colors[self.color_index],
            "thickness": self.brush_thickness
        }
        self.last_point = start_point

    def add_point_to_stroke(self, raw_point):
        if self.current_stroke is None:
            return
        smoothed = smooth_point(self.last_point, raw_point)
        snapped = snap_to_grid(smoothed)
        self.current_stroke["points"].append(snapped)
        self.last_point = snapped

    def end_stroke(self):
        if self.current_stroke is None:
            return
        pts = self.current_stroke["points"]
        if len(pts) > 1:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)

            if self.active_tool == "circle":
                pts = make_circle_points(minx, miny, maxx, maxy)
            elif self.active_tool == "rect":
                pts = make_rect_points(minx, miny, maxx, maxy)

            self.current_stroke["points"] = pts
            self.strokes.append(self.current_stroke)
            self.push_history()  # record new stroke in history

        self.current_stroke = None
        self.last_point = None

    def erase_at(self, x, y):
        p = (x, y)
        new_strokes = []
        for stroke in self.strokes:
            new_points = [pt for pt in stroke["points"] if dist(pt, p) > self.eraser_size]
            if len(new_points) > 1:
                stroke["points"] = new_points
                new_strokes.append(stroke)
        self.strokes = new_strokes

        if self.current_stroke is not None:
            new_points = [pt for pt in self.current_stroke["points"] if dist(pt, p) > self.eraser_size]
            if len(new_points) > 1:
                self.current_stroke["points"] = new_points
            else:
                self.current_stroke = None
                self.drawing = False

    # ------------- clear / save -------------

    def clear_all(self):
        self.strokes = []
        self.current_stroke = None
        self.drawing = False
        self.push_history()
        print("Canvas cleared")

    def save_canvas(self):
        if self.canvas is None:
            return
        # redraw strokes to be sure it's fresh
        self.draw_strokes_on_canvas()
        h, w, _ = self.canvas.shape
        white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        # mask where strokes exist
        mask = np.any(self.canvas != 0, axis=2)
        white_bg[mask] = self.canvas[mask]

        folder = "airdraw_saves"
        os.makedirs(folder, exist_ok=True)
        filename = time.strftime("%Y%m%d_%H%M%S") + ".png"
        path = os.path.join(folder, filename)
        cv2.imwrite(path, white_bg)
        print(f"Saved drawing to {path}")

    # ------------- drawing canvas -------------

    def draw_strokes_on_canvas(self):
        if self.canvas is None:
            return
        self.canvas[:] = 0
        for stroke in self.strokes:
            pts = stroke["points"]
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                cv2.line(self.canvas, pts[i - 1], pts[i], stroke["color"], stroke["thickness"])
        if self.current_stroke is not None:
            pts = self.current_stroke["points"]
            for i in range(1, len(pts)):
                cv2.line(
                    self.canvas,
                    pts[i - 1],
                    pts[i],
                    self.current_stroke["color"],
                    self.current_stroke["thickness"]
                )

    # ------------- UI (toolbar + colors) -------------

    def get_toolbar_layout(self, w, h):
        toolbar_top = h - self.toolbar_height
        tool_ids = ["brush", "eraser", "circle", "rect", "thin", "thick", "clear", "save"]
        labels = ["Brush", "Eraser", "Circle", "Rect", "-", "+", "Clear", "Save"]
        n = len(tool_ids)
        cell_w = int(w / n)

        buttons = []
        for i, (tid, label) in enumerate(zip(tool_ids, labels)):
            x1 = i * cell_w + 5
            x2 = (i + 1) * cell_w - 5
            y1 = toolbar_top + 5
            y2 = h - 5
            buttons.append({
                "id": tid,
                "label": label,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })
        return buttons

    def get_color_layout(self, w, h):
        toolbar_top = h - self.toolbar_height
        color_top = toolbar_top - self.color_bar_height - 5
        color_bottom = toolbar_top - 5
        n = len(self.colors)
        cell_w = int(w / n)

        buttons = []
        for i in range(n):
            x1 = i * cell_w + 10
            x2 = (i + 1) * cell_w - 10
            y1 = color_top
            y2 = color_bottom
            buttons.append({
                "index": i,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })
        return buttons

    def draw_toolbar_and_colors(self, frame):
        h, w, _ = frame.shape
        toolbar_top = h - self.toolbar_height
        color_buttons = self.get_color_layout(w, h)
        tool_buttons = self.get_toolbar_layout(w, h)

        # toolbar background
        cv2.rectangle(frame, (0, toolbar_top), (w, h), (20, 20, 20), -1)

        # color bar background
        if color_buttons:
            y1 = color_buttons[0]["y1"] - 2
            y2 = color_buttons[0]["y2"] + 2
            cv2.rectangle(frame, (0, y1), (w, y2), (40, 40, 40), -1)

        # color buttons
        for cb in color_buttons:
            color = self.colors[cb["index"]]
            selected = (cb["index"] == self.color_index)
            thickness = 3 if selected else 1
            cv2.rectangle(
                frame,
                (cb["x1"], cb["y1"]),
                (cb["x2"], cb["y2"]),
                color,
                -1
            )
            cv2.rectangle(
                frame,
                (cb["x1"], cb["y1"]),
                (cb["x2"], cb["y2"]),
                (255, 255, 255),
                thickness
            )

        # tool buttons
        for tb in tool_buttons:
            is_active = (tb["id"] == self.active_tool)
            cv2.rectangle(
                frame,
                (tb["x1"], tb["y1"]),
                (tb["x2"], tb["y2"]),
                (80, 80, 80) if is_active else (60, 60, 60),
                -1
            )
            cv2.rectangle(
                frame,
                (tb["x1"], tb["y1"]),
                (tb["x2"], tb["y2"]),
                (200, 200, 200),
                1
            )
            cv2.putText(
                frame,
                tb["label"],
                (tb["x1"] + 5, tb["y1"] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

        # status box
        cv2.rectangle(frame, (0, 0), (260, 60), (0, 0, 0), -1)
        cv2.putText(
            frame, f"Tool: {self.active_tool}",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        cv2.putText(
            frame, f"Thick: {self.brush_thickness}",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

    def hit_test_ui(self, x, y, w, h):
        toolbar_top = h - self.toolbar_height

        color_buttons = self.get_color_layout(w, h)
        for cb in color_buttons:
            if cb["x1"] <= x <= cb["x2"] and cb["y1"] <= y <= cb["y2"]:
                return ("color", cb["index"])

        if y >= toolbar_top:
            tool_buttons = self.get_toolbar_layout(w, h)
            for tb in tool_buttons:
                if tb["x1"] <= x <= tb["x2"] and tb["y1"] <= y <= tb["y2"]:
                    return ("tool", tb["id"])
        return None

    def apply_ui_action(self, kind, value):
        if kind == "color":
            self.color_index = value
            print("Color changed to index", self.color_index)
            return

        if kind == "tool":
            tid = value
            if tid in ["brush", "eraser", "circle", "rect"]:
                self.active_tool = tid
                print("Tool:", tid)
            elif tid == "thin":
                self.brush_thickness = max(self.min_thickness, self.brush_thickness - 2)
                print("Thickness:", self.brush_thickness)
            elif tid == "thick":
                self.brush_thickness = min(self.max_thickness, self.brush_thickness + 2)
                print("Thickness:", self.brush_thickness)
            elif tid == "clear":
                self.clear_all()
            elif tid == "save":
                self.save_canvas()

    # ------------- hand processing -------------

    def process_left_hand(self, img, handLms, handedness_label):
        h, w, _ = img.shape
        lm = handLms.landmark
        ix = int(lm[8].x * w)
        iy = int(lm[8].y * h)

        # eraser: index + middle pinch
        erase_pinch, _, _ = self.is_pinch(lm, 8, 12, w, h, threshold=40)
        if erase_pinch:
            if not self.erasing_session_active:
                self.erasing_session_active = True
            cv2.circle(img, (ix, iy), self.eraser_size, (255, 255, 255), 2)
            self.erase_at(ix, iy)
        else:
            if self.erasing_session_active:
                self.erasing_session_active = False
                self.push_history()

    def process_right_hand(self, img, handLms, handedness_label):
        h, w, _ = img.shape
        lm = handLms.landmark
        ix = int(lm[8].x * w)
        iy = int(lm[8].y * h)

        # visible pointer (green)
        cv2.circle(img, (ix, iy), 8, (0, 255, 0), 2)

        toolbar_top = h - self.toolbar_height
        color_top = toolbar_top - self.color_bar_height - 5

        draw_pinch, _, _ = self.is_pinch(lm, 4, 8, w, h, threshold=40)

        if draw_pinch:
            in_color_bar = color_top <= iy < toolbar_top
            in_toolbar = iy >= toolbar_top

            if in_color_bar or in_toolbar:
                # hover / hold selection
                target = self.hit_test_ui(ix, iy, w, h)

                if target != self.ui_hover_target:
                    self.ui_hover_target = target
                    self.ui_hover_start_time = time.time() if target else None
                    self.ui_hold_confirmed = False
                else:
                    if target and not self.ui_hold_confirmed and self.ui_hover_start_time is not None:
                        if time.time() - self.ui_hover_start_time >= self.ui_hold_duration:
                            self.apply_ui_action(*target)
                            self.ui_hold_confirmed = True

                if self.drawing:
                    self.end_stroke()
                    self.drawing = False
                return
            else:
                # left UI region
                self.ui_hover_target = None
                self.ui_hover_start_time = None
                self.ui_hold_confirmed = False

            # tool: eraser with right hand too
            if self.active_tool == "eraser":
                if not self.erasing_session_active:
                    self.erasing_session_active = True
                self.erase_at(ix, iy)
                return

            # drawing modes (brush / circle / rect)
            if not self.drawing:
                start_pt = snap_to_grid((ix, iy))
                self.start_new_stroke(start_pt)
                self.drawing = True
            else:
                if self.current_stroke is not None:
                    self.add_point_to_stroke((ix, iy))
        else:
            # pinch released
            if self.drawing:
                self.end_stroke()
                self.drawing = False

            if self.erasing_session_active and self.active_tool == "eraser":
                self.erasing_session_active = False
                self.push_history()

            self.ui_hover_target = None
            self.ui_hover_start_time = None
            self.ui_hold_confirmed = False

    # ------------- main loop -------------

    def run(self):
        if not self.cap.isOpened():
            print("Could not open webcam")
            return

        first_frame = True

        while True:
            success, img = self.cap.read()
            if not success:
                break

            img = cv2.flip(img, 1)

            if first_frame:
                h, w, _ = img.shape
                self.canvas = np.zeros_like(img)
                first_frame = False

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for idx, handLms in enumerate(results.multi_hand_landmarks):
                    handedness_label = "Right"
                    if results.multi_handedness and len(results.multi_handedness) > idx:
                        handedness_label = results.multi_handedness[idx].classification[0].label

                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

                    if handedness_label == "Left":
                        self.process_left_hand(img, handLms, handedness_label)
                    else:
                        self.process_right_hand(img, handLms, handedness_label)

            # draw strokes to canvas
            self.draw_strokes_on_canvas()

            # combine camera + canvas
            if self.canvas is not None:
                output = cv2.addWeighted(img, 0.5, self.canvas, 0.8, 0)
            else:
                output = img

            # optional grid overlay
            if self.show_grid:
                h, w, _ = output.shape
                step = 50
                for x in range(0, w, step):
                    cv2.line(output, (x, 0), (x, h), (60, 60, 60), 1)
                for y in range(0, h, step):
                    cv2.line(output, (0, y), (w, y), (60, 60, 60), 1)

            # UI
            self.draw_toolbar_and_colors(output)

            cv2.imshow("AirDraw", output)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):
                break
            if key in (26, ord('z')):
                self.undo()
            if key in (25, ord('y')):
                self.redo()
            if key == ord('c'):
                self.clear_all()
            if key == ord('s'):
                self.save_canvas()
            if key == ord('g'):
                self.show_grid = not self.show_grid
                print("Grid:", self.show_grid)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AirDrawApp()
    app.run()
