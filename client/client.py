import socket
import sys
import ctypes
import struct
import cv2
import numpy as np

packetSize = 1300
dataType = {0 : np.uint8,
            1 : np.int8,
            2 : np.uint16,
            3 : np.int16,
            4 : np.int32,
            5 : np.float32,
            6 : np.float64}


def receiveData(sock):
    handshakeData = [None]*4
    for i in range(0,4):
        try:
            handshakeEntry = struct.unpack('<I', sock.recv(4))[0]
            handshakeData[i] = handshakeEntry
        except struct.error:
            print "Disconnected during handshake."
            return None

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
            return None
        buffer+=received
        bytesLeftToReceive-=len(received)

        if bytesLeftToReceive == 0:
            print "Received "+ str(len(buffer)) + " bytes"
            break

    #Image received, convert to a byte array and decode

    imageBytes = np.fromstring(buffer, dataType[handshakeData[1]])
    image = np.reshape(imageBytes,(handshakeData[2], handshakeData[3]))
    return image




def main():
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
            image = receiveData(sock)
            if image is None:
                break
            else:
                small = cv2.resize(image, (0,0), fx=0.3, fy=0.3)
                cv2.imshow("Real-time view", small)
                cv2.waitKey(100)

if __name__ == "__main__":
    main()