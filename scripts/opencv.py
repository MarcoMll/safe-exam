import cv2

capture = cv2.VideoCapture(0)  # 0 is the default camera
ret, frame = capture.read()  # ret (frame was read), (frame itself)
print(ret, frame)

capture.release()  # always release the capture after you are done with it
