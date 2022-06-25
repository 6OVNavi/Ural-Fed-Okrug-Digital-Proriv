
from tkinter import *
from time import *
import numpy as np
import cv2
import sys
#sys.path.append('D:\!!УФА ВЩМГОР\hackaton\yolov5face')
sys.path.insert(1, 'yolov5face')
import detect_face

#import main
def start():
    global var
    var = True
    cap = cv2.VideoCapture('IMG_0003.MOV')  # 'test_home.mov'

    # the output will be written to output.avi
    out = cv2.VideoWriter(
        'output.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        15.,
        (640, 480))
    #cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # resizing for faster detection
        if frame is None:
            break
        frame = cv2.resize(frame, (640, 480))
        # using a greyscale picture, also for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes=detect_face.detect_one('face.pt', frame, device='cpu')
        #print(faces)
        for i in boxes:

            i = list(map(int, i))
            print(i)
            xA = i[0];
            yA = i[1];
            xB = i[2];
            yB = i[3]

            # display the detected boxes in the colour picture
            cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)

        # Write the output video
        out.write(frame.astype('uint8'))
        # Display the resulting frame
        #cv2.imshow('frame', frame)

        '''if cv2.waitKey(1) & 0xFF == ord('q'):
            break'''

    # When everything done, release the capture
    cap.release()
    # and release the output
    out.release()
def end():
    global var
    var=False


window = Tk()
window.geometry("300x400")
window.title("Emotion Detectrotion")
b1 = Button(text="Start recording",
            width=15, height=2)
b1.config(command=start)
b1.pack()
b2 = Button(text="End recording",
            width=15, height=2)
b2.config(command=end)
b2.pack()
window.mainloop()
