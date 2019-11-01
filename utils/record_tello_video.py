import cv2

camera = cv2.VideoCapture('udp://127.0.0.1:11111')
video_size = (960, 720)
out = cv2.VideoWriter('tello.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, video_size)

while(True):
    ret, frame = camera.read()

    if ret == True:
        out.write(frame)

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

camera.release()
out.release()
cv2.destroyAllWindows()
