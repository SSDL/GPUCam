import socket
import sys
import ctypes
import struct

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = ("10.34.196.111", 4097)
print "Attempting to connect to the server..."
while(True):
    try:
        sock.connect(server_address)
    except socket.error:
        continue
    else:
        print "Connection established."
        break

while(True):
    while(True):
        try:
            imageSize = struct.unpack('<I', sock.recv(4))[0]
        except struct.error:
            print "Disconnected during handshake."
            sys.exit(0)
        else:
            print "Received 4 bytes, image size:" + str(imageSize)
            break

    #Receive the image.
