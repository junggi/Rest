import socket

HOST = 'localhost'
PORT = 1234

s = socket.socket()
s.connect((HOST, PORT))
while 1:
    a = raw_input("Sup ")
    if a != "0":
        s.send(a)
        data = s.recv(1024)
        print "Received, ",data
    else:
        break
s.close()
