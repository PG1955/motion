import cv2
import numpy as np

class Graph:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.graph = np.zeros((height, width, 3), np.uint8)
    def update_frame(self, value):
        if value < 0:
            value = 0
        elif value >= self.height:
            value = self.height - 1
        new_graph = np.zeros((self.height, self.width, 3), np.uint8)
        new_graph[:,:-1,:] = self.graph[:,1:,:]
        new_graph[self.height - value:,-1,:] = 255
        self.graph = new_graph
    def get_graph(self):
        return self.graph

# Setup camera
cap = cv2.VideoCapture(0)
# Set a smaller resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
graph = Graph(620, 48)
prev_frame = np.zeros((480, 640), np.uint8)
while True:
    # Capture frame-by-frame
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), None)
    diff = cv2.absdiff(prev_frame, gray)
    difference = np.sum(diff)
    prev_frame = gray
    graph.update_frame(int(difference/42111))
    roi = frame[-58:-10, -630:-10,:]
    roi[:] = graph.get_graph()
    cv2.putText(frame, "...wanted a live graph", (20, 430), cv2.FONT_HERSHEY_PLAIN, 1.8, (200, 200, 200), 2)
    cv2.putText(frame, "...measures motion in frame", (20, 460), cv2.FONT_HERSHEY_PLAIN, 1.8, (200, 200, 200), 2)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()