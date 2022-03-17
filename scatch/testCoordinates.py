import numpy as np
import cv2

height = 240
width = 320

yp = 10
xp = 95
y = int((yp * height)/100)
x = int((xp * width)/100)
b = 10
print('x:{} y:{}'.format(x,y))

frame = np.zeros((height, width, 3), np.uint8)
frame.fill(200)
graph = np.zeros((y, x, 3), np.uint8)

while True:
    # Specify the region of interest.
    roi = frame[-abs(y + b):-abs(b),-abs(x + b):-abs(b), :]
    # put the graph on the roi
    roi[:] = graph
    cv2.imshow('Frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()







