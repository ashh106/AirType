import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
# this is a formality before using this module
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
  sucess, img = cap.read() 
  
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# RGB class is used because mediapipe uses RGB images
  results = hands.process(imgRGB)
  # for loop to check multiple hand and extraxt one by one
  # print(results.multi_hand_landmarks)
  if results.multi_hand_landmarks:
      for handLms in results.multi_hand_landmarks:
          mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
          

  cv2.imshow("Image", img)
  cv2.waitKey(1)
  

  # create an object from class hand

