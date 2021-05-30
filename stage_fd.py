from model import FacialExpressionModel
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from cv2 import *
import keras
import os, sys, time, socket
import threading
import argparse
import numpy as np
import cv2

modelsdir = 'models'

model_loaded = FacialExpressionModel("models/model.json", "models/fer.h5")
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

default_server_port = 9251

"""
Predict class of an image
"""


def detectPeople(model, imagefile):
    global classnames

    inp = inputImage(imagefile)
    if inp is not None:
        faces = facec.detectMultiScale(inp, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(inp, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return inp
    else:
        return (0, 'error')


"""
Load an image and return input data for the network
"""


def inputImage(imagefile):
    try:
        #IMREAD_GRAYSCALE
        gray = cv2.imread(imagefile, IMREAD_COLOR)
        gray = np.array(gray, dtype='uint8')
        return gray
    except:
        return None


"""
Load a trained model
"""


def loadModel(modelname):
    global modelsdir
    filename = os.path.join(modelsdir, '%s.h5' % modelname)
    try:
        model = keras.models.load_model(filename)
        print("\nModel loaded successfully from file %s\n" % filename)
    except OSError:
        print("\nModel file %s not found!!!\n" % modelname)
        model = None
    return model


class ModelServer(threading.Thread):

    def __init__(self, port, model):
        threading.Thread.__init__(self)

        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(3)  # timeout when listening (exit with CTRL+C)

        # Bind the socket to the port
        server_address = ('', port)
        self.sock.bind(server_address)
        self.sock.listen(1)

        self.model = model

        print("Server running on port %d" % port)

        self.dorun = True  # server running
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
                print('Connection from %s' % str(client_address))
            except:
                pass

                # buf may contain a first chunk of data

    def recvall(self, count, chunk):
        buf = chunk
        count -= len(buf)
        while count > 0:
            newbuf = self.connection.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def run(self):

        imgsize = -1
        res = 'none 0.0'
        while (self.dorun):
            self.connect()  # wait for connection
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
                    if (type(data) != str):
                        k = data.find(b'\n')
                        if (k < 0):
                            data = data.decode('utf-8')
                        elif (len(data) > k + 1):
                            buf = data[k + 2:]
                            data = data[0:k].decode('utf-8')

                    if (data != None and data != "" and data != "***"):
                        self.received = data
                        print('Received: %s' % data)
                        v = data.split(' ')
                        if v[0] == 'EVAL' and len(v) > 1:
                            print("\n-----Detecting people------\n")
                            people_detected = detectPeople(self.model, v[1])

                            # Without resizing the window, it will fit the whole screen
                            cv2.namedWindow("detected", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("detected", 959, 1280)

                            cv2.imshow("detected", people_detected)
                            cv2.waitKey(6000)
                            cv2.destroyAllWindows()

                            res = "People detected!"
                            ressend = (res + '\n\r').encode('UTF-8')
                            self.connection.send(ressend)
                        else:
                            print('Received: %s' % data)
                    elif (data == None or data == ""):
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
    print("Starting server on port %d" % port)
    mserver = ModelServer(port, model)
    mserver.start()
    mserver.spin()
    mserver.stop()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-modelname", type=str, default=None,
                        help="Model name to load/save")
    parser.add_argument('--server', default=False, action='store_true',
                        help='Start in server mode')
    parser.add_argument('-server_port', type=int, default=default_server_port,
                        help='server port (default: %d)' % default_server_port)
    parser.add_argument("-predict", type=str, default=None,
                        help="Image file to predict")

    args = parser.parse_args()

    if (args.modelname == None):
        print("Please specify a model name and an operation to perform.")
        sys.exit(1)
    elif (args.server):
        model = loadModel(args.modelname)
        startServer(args.server_port, model)
    elif (args.predict != None):
        print("\n -----Detecting people:------\n")
        model = loadModel(args.modelname)
        people_detected = detectPeople(model, args.predict)

        # Without resizing the window, it will fit the whole screen
        cv2.namedWindow("detected",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("detected",(959,1280))

        cv2.imshow("detected",people_detected)
        cv2.waitKey(0)
        cv2.imwrite("prova.jpg",people_detected)
    else:
        print("Please specify a model name and an operation to perform.")
        sys.exit(1)