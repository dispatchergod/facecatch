import cv2


def main():
    video = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("face_finder.xml")

    while True:
        check, frame = video.read()

        faces = face_cascade.detectMultiScale(frame,
                                              scaleFactor=1.2,
                                              minNeighbors=5)
        for x, y, w, h in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imshow("Capturing", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
