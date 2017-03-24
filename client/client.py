import socket
import sys
import ctypes
import struct
import cv2
import numpy as np
import threading
from multiprocessing.pool import ThreadPool

packetSize = 1300
dataType = {0 : np.uint8,
            1 : np.int8,
            2 : np.uint16,
            3 : np.int16,
            4 : np.int32,
            5 : np.float32,
            6 : np.float64}
dataArrayLock = threading.Lock()
image = []


def receiveData(sock):
    global image
    handshakeData = [None]*4
    for i in range(0,4):
        try:
            handshakeEntry = struct.unpack('<I', sock.recv(4))[0]
            handshakeData[i] = handshakeEntry
        except struct.error:
            print "Disconnected during handshake."
            return False

    print "Image size: " + str(handshakeData[0]) + ", type number: " + str(handshakeData[1]) + ", rows: " + str(handshakeData[2]) + ", cols: " + str(handshakeData[3])

    #Receive the image.
    bytesLeftToReceive = handshakeData[0]
    buffer=b''
    while(True):
        nextPacketSize=packetSize
        if bytesLeftToReceive < packetSize:
            nextPacketSize = bytesLeftToReceive

        received=sock.recv(nextPacketSize)
        if received == "":
            print "Disconnected during data transmission"
            return False
        buffer+=received
        bytesLeftToReceive-=len(received)

        if bytesLeftToReceive == 0:
            print "Received "+ str(len(buffer)) + " bytes"
            break

    #Image received, convert to a byte array and decode

    imageBytes = np.fromstring(buffer, dataType[handshakeData[1]])
    dataArrayLock.acquire()
    image = np.reshape(imageBytes,(handshakeData[2], handshakeData[3]))
    dataArrayLock.release()
    return True




def main():
    pool = ThreadPool(1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ("10.34.193.123", 4097)
    print "Attempting to connect to the server..."
    while(True):
        try:
            sock.connect(server_address)
        except socket.error:
            continue
        else:
            print "Connection established."
        while(True):
            #May be an overkill to use threadpool? It's quicker than having the worker wait on a semaphore
            socketMap = [sock]
            dataReceived = pool.map(receiveData, socketMap)
            if not dataReceived[0]:
                break
            else:
                dataArrayLock.acquire()
                small = cv2.resize(image, (0,0), fx=0.3, fy=0.3)
                dataArrayLock.release()
                cv2.imshow("Middle camera preview", small)
                cv2.waitKey(1)

if __name__ == "__main__":
    main()