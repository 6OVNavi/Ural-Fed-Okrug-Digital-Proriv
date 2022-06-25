from tkinter import *
from time import *
import numpy as np
import cv2
import sys
import torch
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten
from keras.optimizers import Adam, RMSprop, SGD
from keras import regularizers
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
import datetime
import threading
from PIL import Image
#from keras.utils import plot_model
#sys.path.append('D:\!!УФА ВЩМГОР\hackaton\yolov5face')
sys.path.insert(1, 'yolov5face')
import detect_face

#import main
row, col = 48, 48
classes = 7
def get_model(input_size, classes=7):
    # Initialising the CNN
    model = tf.keras.models.Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_size))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(classes, activation='softmax'))

    # Compliling the model
    '''model.compile(optimizer=Adam(lr=0.0001, decay=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])'''
    return model

clf = get_model((row,col,1), classes)

#clf.build((row,col,1, 0))

clf.load_weights('fernet_bestweight.ckpt.index')
#clf.load_state_dict(torch.load('fernet_bestweight.h5'))


def start():

    global var
    var = True

    cap = cv2.VideoCapture('IMG_0002.MOV')  # VIDEO/WEBCAM/PHOTO PATH

    # the output will be written to output.avi
    out = cv2.VideoWriter(
        'output.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        15.,
        (640, 480))
    #cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    while (var):
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
            xA = i[0];
            yA = i[1];
            xB = i[2];
            yB = i[3]
            img=frame[yA:yB, xA:xB]
            img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #img=img.reshape(48, 48, 1)
            img=Image.fromarray(img)
            img=img.resize((48, 48), Image.ANTIALIAS)
            img=np.array(img)
            img = img.reshape(1, 48, 48, 1)
            img=img.astype(float)
            predict=clf.predict(img)
            max=[]
            index=[]
            for i in range(len(predict[0])):
                if len(max)<3:
                    max.append(predict[0][i])
                    index.append(i)
                else:
                    max.sort()
                    if max[0]<predict[0][i]:
                        max[0]=predict[0][i]
                        index[0]=i




            # display the detected boxes in the colour picture
            '''cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
'''
        # Write the output video
        out.write(frame.astype('uint8'))
        #out.write(frame.astype('uint8'))
        # Display the resulting frame
        #cv2.imshow('frame', frame)

        '''if cv2.waitKey(1) & 0xFF == ord('q'):
            break'''

    # When everything done, release the capture
    cap.release()
    # and release the output
    out.release()
def real_start():
    thread = threading.Thread(target=start)
    thread.daemon = True
    thread.start()
def end():
    global var
    var=False


window = Tk()
window.geometry("300x400")
window.title("Emotion Detectrotion")
b1 = Button(text="Start recording",
            width=15, height=2)
b1.config(command=real_start)
b1.pack()
b2 = Button(text="End recording",
            width=15, height=2)
b2.config(command=end)
b2.pack()
window.mainloop()
