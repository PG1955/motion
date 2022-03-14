import cv2


class libcamera:

    def __init__(self):
        self.width = None
        self.height = None
        self.rotation = None
        self.cap = None
        self.d = None
        return None

    def initCamera(self, width, height, RGB888, buffercount, rotation):
        self.cap = cv2.VideoCapture(0)
        self.width = width
        self.height = height
        self.rotation = rotation
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        return False

    def startCamera(self):
        return True

    def set(self, FrameDurationLimits, param):
        pass

    def readFrame(self):
        self.data = data(self.cap)
        return True, self.data

    def read(self):
        return self.cap.read()

    def returnFrameBuffer(self):
        pass

    def stopCamera(self):
        pass

    def closeCamera(self):
        pass

    def get(self, index):
        return self.cap.get(index)


class data:
    def __init__(self, capture):
        self.capture = capture
        self.r, self.frame = self.capture.read()
        return None


    def imageData(self):
        return self.frame

class PixelFormat:
    RGB888 = None


class FrameDurationLimits:
    pass

class data:
    def __init__(self, cap):
        self.cap = cap
        self.image = self.cap.read()
        return None

    def imageData(self):
        return self.image
