import cv2

cap = cv2.VideoCapture(0)
# 480	×	234 16:9
# 320 X 240 4:3

# cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,234)

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print(height, width)


while True:
    _, frame = cap.read()
    # Display the frame
    cv2.imshow('Camera', frame)

    # Wait for 25ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(frame.shape)
# release the camera from video capture
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()