import socket
import numpy as np
from ApneaDetector.RespData.RespData import RespData
from ApneaDetector.ApneaDetector import detectApneas
# 10.0.31.190
exp = 3
# Rounds to 10^exp-th decimal. Currently set to 3.
# Change if parameters or output need more precision
def roundTo(r):
    return int(r*(10**exp))/(10.0**exp)

# load a .resp file, with log
def getRespData(fileNum):
    respFileName = "RespFiles/ALG_ ("+fileNum+").resp"
    data = RespData(respFileName)
    
    #Get matching peak data
    temp = data.getFeature('upperPeakTimes')
    upperPeakTimes = np.zeros(data.data.shape[0])           #pair peaks down
    upperPeakTimes[temp] = 1                                #cut down to length
    upperPeakTimes = np.nonzero(upperPeakTimes)[0]          #get index
    
    data.loadLog("Logs/ALG_ ("+fileNum+").txt")

    return data

# main client.
# f is a function of resp data, and a parameter set.
# client assumes that server knows how many parameters
# that f uses. Parameter set is of form (p1, p2,...)
def client(f):
    HOST = '10.0.31.190'
##    HOST = 'localhost'
    PORT = 1234
    currentFileNum = None
    resp = None

##    with socket.socket() as s:
    s = socket.socket()
    s.connect((HOST, PORT))
    
    while 1:

        # wait for new point to compute
        msg = s.recv(1024)    # msg = compute fileNum p1 p2 p3...
##        while msg[len(msg)-1] != "\n":
        while msg.strip().count(" ") != 6:
            print 1
##            and (roundTo(float(msg.rsplit(maxsplit = 1)[1])) not in np.arange(30.0,80,10)):
            msg += s.recv(1024)
        while roundTo(float(msg.strip().rsplit(" ",1)[1])) not in np.arange(30.0,80,10):
            print 2
            msg += s.recv(1024)
        print "received:", msg
        tokens = msg.strip().split(" ")
        if tokens[0] == "compute":
            pSet = tokens[2:]
            fileNum = tokens[1]
            
            # beginning
            if currentFileNum == None:
                currentFileNum = fileNum
                resp = getRespData(fileNum)
            # load new file
            elif currentFileNum != fileNum:
                currentFileNum = fileNum
                resp = getRespData(fileNum)


            # compute output. 
            pSet = np.array([float(x) for x in pSet])
            output = f(resp, pSet)
            
            csMsg = "store "+fileNum
            for p in pSet:
                csMsg += " "+str(roundTo(p))
            csMsg += " "+str(roundTo(output))
            
            print "sent:", csMsg
##            s.sendall(csMsg+"\n")
            s.sendall(csMsg)
        elif tokens[0] == "done":
            print "done"
            s.close()
            break
        else:
            print "check message"
            s.sendall("resend\n")
    raw_input("Done. Press enter to exit.")

if __name__ == "__main__":
    a = 1
    if a == 0:
        def f(resp, pSet):
            return pSet[0]
        client(f)
    if a == 1:
        def f(resp, pSet):
            options = {
                'dropTime':         pSet[0],
                'dropPercent':      pSet[1],
                'topAverageRate':   pSet[2],
                'eventLengthMin':   pSet[3],
                'eventLengthMax':   pSet[4],
                'staging':          "LAB",
            }
            return detectApneas(resp, options)[0]
        client(f)
