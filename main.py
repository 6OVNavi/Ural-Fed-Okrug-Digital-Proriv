from tkinter import *
from time import *
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import pandas as pd
import torch
import tensorflow as tf
from tensorflow import keras
#from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten
from keras.optimizers import Adam, RMSprop, SGD
from keras import regularizers
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
import datetime
import threading
import time
from PIL import Image
#from keras.utils import plot_model
#sys.path.append('D:\!!УФА ВЩМГОР\hackaton\yolov5face')
sys.path.insert(1, 'yolov5face')
import detect_face

#import main
row, col = 48, 48
classes = 7
emotions=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
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

clf.load_weights('train71val55.ckpt')
#clf.load_state_dict(torch.load('fernet_bestweight.h5'))


def start():
    global df
    d={'id': [-1],
       'time': [-1],
       'em1': [-1],
       'em2': [-1],
       'em3': [-1]
       }
    df=pd.DataFrame(data=d)

    start_=time.time()
    global var
    var = True

    cap = cv2.VideoCapture('IMG_0004.MOV')  # VIDEO/WEBCAM/PHOTO PATH

    # the output will be written to output.avi
    out = cv2.VideoWriter(
        'output.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        15.,
        (640, 480))
    #cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    borders=[]
    count=0
    while (var):
        if count%10==0:

            # Capture frame-by-frame
            ret, frame = cap.read()
            # resizing for faster detection
            if frame is None:
                break
            frame = cv2.resize(frame, (640, 640))
            # using a greyscale picture, also for faster detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # detect people in the image
            # returns the bounding boxes for the detected objects
            boxes=detect_face.detect_one('face.pt', frame, device='cpu')


            #print(faces)
            for i in boxes:
                destroy=False
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
                maxx=[]
                index=[]

                for i in range(len(predict[0])):
                    if len(maxx)<3:
                        maxx.append(predict[0][i])
                        index.append(i)
                    else:
                        maxx.sort()
                        if maxx[0]<predict[0][i]:
                            maxx[0]=predict[0][i]
                            index[0]=i
                print(index)
                print(maxx)
                for j in range(len(borders)):
                    if borders[j][0]<=xA<=borders[j][2] and borders[j][0]<=xB<=borders[j][2] and borders[j][1]<=yA<=borders[j][3] and borders[j][1]<=yB<=borders[j][3]:
                        id_=j
                        time_=time.time()-start_
                        em1_=emotions[index[2]]
                        em2_=emotions[index[1]]
                        em3_=emotions[index[0]]
                        df.loc[len(df.index)] = [id_, time_, em1_, em2_, em3_]
                        destroy=True
                        break
                if not destroy:

                    id_ =len(borders)
                    #print(id_)
                    time_ = time.time() - start_
                    em1_ = emotions[index[2]]
                    em2_ = emotions[index[1]]
                    em3_ = emotions[index[0]]
                    borders.append((max(0,xA-25), max(0,yA-25), min(640,xB+25), min(640,yB+25)))
                    df.loc[len(df.index)] = [id_, time_, em1_, em2_, em3_]


                # display the detected boxes in the colour picture
                cv2.rectangle(frame, (xA, yA), (xB, yB),
                              (0, 255, 0), 2)
                cv2.putText(frame, emotions[index[2]],
                            (xA, yB),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            3,
                            2
                            )
            # Write the output video
            out.write(frame.astype('uint8'))
            #out.write(frame.astype('uint8'))
            # Display the resulting frame
            #cv2.imshow('frame', frame)
            count+=1
            '''if cv2.waitKey(1) & 0xFF == ord('q'):
                break'''

            cv2.imshow('frame', frame)

            cv2.waitKey(0)
    # When everything done, release the capture
    cap.release()
    # and release the output
    out.release()
def real_start():
    thread = threading.Thread(target=start)
    thread.daemon = True
    thread.start()
def end():
    global df
    df.to_csv('result.csv', index=False)
    global var
    var=False

def chart():
    plt.pie(frame, labels=my_labels, autopct='%1.1f%%')
    plt.title('My Tasks')
    plt.axis('equal')
    plt.show()
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

xw = Text(window, height=1,
          width=15,
          bg="light yellow")
xw.insert(INSERT, "X")
xw.pack()
b3 = Button(text="End",
            width=15, height=2)
b3.config(command=end)
b3.pack()

window.mainloop()
