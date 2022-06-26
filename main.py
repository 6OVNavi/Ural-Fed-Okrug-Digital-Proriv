from tkinter import *
from time import *
import numpy as np
import cv2
import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
#from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
import datetime
import threading
import time
from PIL import Image
#from keras.utils import plot_model
#sys.path.append('D:\!!УФА ВЩМГОР\hackaton\yolov5face')
sys.path.insert(1, 'yolov5face')
import detect_face
matplotlib.use('TkAgg')
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

    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

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
#clf=clf.load_model('train71val55.ckpt')
clf.load_weights('ferNet.h5')


def start():
    spath = path.get(1.0, END)
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
    try:
        cap = cv2.VideoCapture(int(spath))  # VIDEO/WEBCAM/PHOTO PATH
    except:
        cap = cv2.VideoCapture(spath)
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
        ret, frame = cap.read()
        count += 1
        if count%3==0:
            
            
            # Capture frame-by-frame
            
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
                maxx=[0, 0, 0]
                index=[-1, -1, -1]

                for j in range(len(predict[0])):
                    if maxx[0]<=predict[0][j]:
                        if maxx[1]<=predict[0][j]:
                            if maxx[2]<=predict[0][j]:
                                maxx[2]=predict[0][j]
                                index[2]=j
                                continue
                            maxx[1]=predict[0][j]
                            index[1]=j
                            continue
                        maxx[0]=predict[0][j]
                        index[0]=j

                print('pred:', predict)
                print('ind', index)
                print('max:', maxx)
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
                cv2.putText(frame, f'{id_}:{emotions[index[2]]}',
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

            '''if cv2.waitKey(1) & 0xFF == ord('q'):
                break'''

            cv2.imshow('frame', frame)
            #cv2.destroyAllWindows()
            cv2.waitKey(40)






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
    global var
    var=False
    df=df[1:]
    df.to_csv('result.csv', index=False)

def chart():
    s = xw.get(1.0, END)
    df=pd.read_csv('result.csv')
    print(df[df['id']==int(s)]['em1'].value_counts().index)
    temp_df=df.copy()
    #temp_df=temp_df.drop(columns=[])
    plt.figure(figsize=(8, 6))
    plt.pie(df[df['id']==int(s)]['em1'].value_counts(), labels=df[df['id']==int(s)]['em1'].value_counts().index , autopct='%1.1f%%')
    plt.title('График эмоций ученика')
    plt.axis('equal')
    plt.show()
    plt.plot(df['time'], df['em1'], label='Наиболее важная')
    plt.plot(df['time'], df['em2'], alpha=0.5, label='Менее важная')
    plt.plot(df['time'], df['em3'], alpha=0.3, label='Незначительная')
    plt.legend(loc='upper center')
    plt.title('Эмоции сквозь время')
    plt.xlabel('Время в секундах')
    plt.ylabel('Эмоции')
    plt.show()

window = Tk()
window.geometry("300x400")
window.title("Emotion Detectrotion")
path = Text(window, height=2,
          width=30,
          bg="light yellow")
path.insert(INSERT, "Путь к файлу или веб-камера(укажите 0)")
path.pack()
b1 = Button(text="Начать запись",
            width=15, height=2)
b1.config(command=real_start)
b1.pack()
b2 = Button(text="Прекратить запись",
            width=15, height=2)
b2.config(command=end)
b2.pack()

xw = Text(window, height=1,
          width=30,
          bg="light yellow")
xw.insert(INSERT, "Укажите ID ученика")
xw.pack()
b3 = Button(text="Создать график",
            width=15, height=2)
b3.config(command=chart)
b3.pack()

window.mainloop()
