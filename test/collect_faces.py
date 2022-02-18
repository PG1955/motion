import cv2 as cv

haar_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

capture = cv.VideoCapture(0)

def capture_frames():
    while True:
        isTrue, frame = capture.read()

        # if cv.waitKey(20) & 0xFF==ord('d'):
        # This is the preferred way - if `isTrue` is false (the frame could
        # not be read, or we're at the end of the video), we immediately
        # break from the loop.
        if isTrue:
            # cv.imshow('Video', frame)
            if cv.waitKey(20) & 0xff == ord('s'):
                # save_face(frame)
                print('Save Face')
            find_faces(frame)
            if cv.waitKey(20) & 0xFF == ord('d'):
                break
        else:
            break

def find_faces(frame):
    # Resize the image.
    scale_percent = 50  # percent of original sizeys
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    small_frame = cv.resize(frame, dim, interpolation=cv.INTER_CUBIC)

    # Get greyscale copy.
    gray = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
    # cv.imshow('Person', gray)

    # Detect faces in the frame.
    # Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]

        # label, confidence = face_recognizer.predict(faces_roi)
        # print(f'Label = {people[label]} with a confidence of {confidence}')

        # cv.putText(small_frame, str(people[label]), (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(small_frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    cv.imshow('Video', small_frame)


capture_frames()

capture.release()
cv.destroyAllWindows()
