import numpy as np
import cv2 
import random


BLUE = ([99, 150, 68], [106, 255, 255])
GREEN = ([65, 150, 47], [80, 255, 255])
ORANGE = ([4, 163, 136], [8, 255, 255])
YELLOW = ([24, 138, 88], [34, 255, 255])


def add_to_mask(mask, color):
    lower = np.array(color[0], dtype="uint8")
    upper = np.array(color[1], dtype="uint8")
    mask += cv2.inRange(hsv, lower, upper)
    

def is_between(center, color):
    le = np.less_equal(hsv[center[::-1]], color[1])
    ge = np.greater_equal(hsv[center[::-1]], color[0])
    if np.all([le, ge]):
        return True
    return False


def define_color(center):
    if is_between(center, BLUE):
        return 'B'
    if is_between(center, GREEN):
        return 'G'
    if is_between(center, YELLOW):
        return 'Y'
    if is_between(center, ORANGE):
        return 'O'


cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, 0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

guessed = ['B', 'G', 'Y', 'O']
random.shuffle(guessed)
guessed = guessed[:3]

while cap.isOpened(): 
    colors_order = {}
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"GOAL: {guessed}",(10,30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255)) 

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype = "uint8")
    add_to_mask(mask, BLUE)
    add_to_mask(mask, GREEN)
    add_to_mask(mask, YELLOW)
    add_to_mask(mask, ORANGE)


    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    for c in contours:
        (x,y),radius = cv2.minEnclosingCircle(c)
        center = (int(x),int(y))
        radius = int(radius)
        if radius > 30:
            cv2.circle(frame,center,radius,(0,255,0),2)
            clr = define_color(center)
            colors_order[clr] = center
    
    colors_order = {k: v for k, v in sorted(colors_order.items(), key=lambda item: item[1])}
    colors_order = list(colors_order.keys())
    cv2.putText(frame, f"CURRENT: {colors_order}",(10,60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))  
    if guessed == colors_order:
        cv2.putText(frame, "YOU WON", (10, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255))  
    key = cv2.waitKey(1)
    if key == ord('d'):
        break
    cv2.imshow('Camera', frame)

cap.release()
cv2.destroyAllWindows()