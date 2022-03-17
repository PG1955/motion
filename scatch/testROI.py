import numpy as np
import cv2
cap = cv2.VideoCapture(0)

# print(arr)
while True:
    _, arr = cap.read()
    roi = arr[-48, -640:-10, :-10]

    cv2.imshow('arr', arr)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
print(arr.shape)
roi = arr[-48,-640:-10,:-10]

print(roi.shape)

cv2.destroyAllWindows()






