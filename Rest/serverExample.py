import socket, select


open_sockets = []
listening_socket = socket.socket( socket.AF_INET, socket.SOCK_STREAM )
listening_socket.bind( ("", 1234) )
listening_socket.listen(5)

while True:
    rlist, wlist, xlist = select.select( [listening_socket] + open_sockets, [], [] )
    for i in rlist:
        if i is listening_socket:
            new_socket, addr = listening_socket.accept()
            open_sockets.append(new_socket)
        else:
            data = i.recv(1024)
            if data == "":
                open_sockets.remove(i)
                print "Connection closed"
            else:
                print repr(data)
                

                
##import socket
##import thread
##
##HOST = 'localhost'
##PORT = 1234
##
##def clientThread(conn):
##    print "client connected"
##    while 1:
##        print "WHOAAAA"
##        data = conn.recv(1024)
##        if not data:
##            break
##        conn.sendall(data)
##    print "closing..."
##    conn.close()
##
##
##
##s = socket.socket()
##s.bind((HOST,PORT))
##while 1:
##    s.listen(1)
##    print "waiting for client..."
##    conn,addr=s.accept()
##    thread.start_new_thread(clientThread, (conn,))
##print "bye"
##conn.close()
