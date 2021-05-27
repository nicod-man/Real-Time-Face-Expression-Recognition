#from flask import Flask, render_template, Response
from camera import VideoCamera
from model import FacialExpressionModel
from keras.models import load_model

#import tensorflow as tf
import keras
import os, sys, time, socket
import threading 
import argparse
import numpy as np
import cv2



#print("Tensorflow version %s" %tf.__version__)
print("Keras version %s" %keras.__version__)


datadir = 'dataset'
modelsdir = os.getcwd()

model = FacialExpressionModel("model.json", "model_weights.h5")
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX


classnames = ["Angry", "Disgust", "Fear", "Happy",
              "Neutral", "Sad", "Surprise"]

default_server_port = 9250
height = 1280
width = 959



"""
Predict class of an image
"""
def predictImage(model, imagefile):
    global classnames

    inp = inputImage(imagefile)
    print(inp)
    if inp is not None:
        gray_fr = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))
            probability, emotion = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            return (probability, emotion)
    else:
        return (0,'error')


    # inp = inputImage(imagefile)
    # if inp is not None:
    #     pr = model.predict(inp)
    #     return (np.max(pr), classnames[np.argmax(pr)])
    # else:
    #     return (0, 'error')

"""
Load an image and return input data for the network
"""

# , target_size=(height, width)
def inputImage(imagefile):
    try:
        img = load_img(imagefile, color_mode="rgb", target_size=(height,width))
        arr = img_to_array(img) / 255
        inp = np.array([arr])  # Convert single image to a batch.
        return inp
    except:
        return None


"""
Load a trained model
"""

def loadModel(modelname):
    global modelsdir
    filename = os.path.join(modelsdir, '%s.h5' %modelname)
    try:
        model = load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:    
        print("\nModel file %s not found!!!\n" %modelname)
        model = None
    return model


class ModelServer(threading.Thread):

    def __init__(self, port, model):
        threading.Thread.__init__(self)

        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(3) # timeout when listening (exit with CTRL+C)
        
        # Bind the socket to the port
        server_address = ('', port)
        self.sock.bind(server_address)
        self.sock.listen(1)

        self.model = model


        print("Server running on port %d" %port)
        
        self.dorun = True # server running
        self.connection = None  # connection object

    def stop(self):
        self.dorun = False

    def connect(self):
        connected = False
        while (self.dorun and not connected):
            try:
                self.connection, client_address = self.sock.accept()
                self.connection.settimeout(3)
                connected = True
                print('Connection from %s' %str(client_address))
            except:
                pass 

    # buf may contain a first chunk of data
    def recvall(self, count, chunk):
        buf = chunk
        count -= len(buf)
        while count>0:
            newbuf = self.connection.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf


    def run(self):

        imgsize = -1
        res = 'none 0.0'
        while (self.dorun):
            self.connect()  
            try:
                # Receive data
                while (self.dorun):
                    try:
                        data = self.connection.recv(256)
                        data = data.strip()
                    except socket.timeout:
                        data = "***"
                    except Exception as e:
                        print(e)
                        data = None

                    buf = b''
                    if (type(data)!=str):
                        k = data.find(b'\n')
                        if (k<0):
                            data = data.decode('utf-8')
                        elif (len(data)>k+1):
                            buf = data[k+2:]
                            data = data[0:k].decode('utf-8')

                            # print('Eval image [%s]' %v[1])
                            # (p,c) = predictImage(self.model,v[1])
                            # print("Predicted: %s, prob: %.3f" %(c,p))
                            # res = "%s %.3f" %(c,p)
                            # ressend = (res+'\n\r').encode('UTF-8')
                            # self.connection.send(ressend)

                    if (data!=None and data!="" and data!="***"):
                        self.received = data
                        print('Received: %s' %data)
                        v = data.split(' ')
                        if v[0]=='EVAL' and len(v)>1:
                            print('Evaluating frame [%s]' %v[1])
                            (p,c) = predictImage(self.model,v[1])
                            print("Predicted: %s, probability: %.3f" %(c,p))
                            res = "%s %.3f" %(c,p)
                            ressend = (res+'\n\r').encode('UTF-8')
                            self.connection.send(ressend)
                        elif (data == None or data==""):
                            break
            finally:
                print('Connection closed.')
                # Clean up the connection
                if (self.connection != None):
                    self.connection.close()
                    self.connection = None

    # wait for Keyboard interrupt
    def spin(self):
        while (self.dorun):
            try:
                time.sleep(120)
            except KeyboardInterrupt:
                print("Exit")
                self.dorun = False


"""
Start prediction server
"""
def startServer(port, model):
    print("Starting server on port %d" %port)
    mserver = ModelServer(port, model)
    mserver.start()
    mserver.spin() 
    mserver.stop()


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-modelname", type=str, default=None,
                        help="Model name to load/save")
    parser.add_argument('--server', default = False, action ='store_true', 
                        help='Start in server mode')
    parser.add_argument('-server_port', type=int, default=default_server_port, 
                        help='server port (default: %d)' %default_server_port)

    args = parser.parse_args()

    if (args.modelname == None):
        print("Please specify a model name and an operation to perform.")
        sys.exit(1)

    elif (args.server):
        model = loadModel(args.modelname)
        startServer(args.server_port, model)
    else:
        print("Please specify a model name and an operation to perform.")
        sys.exit(1)