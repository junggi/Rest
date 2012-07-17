import socket, select
import Queue
import sqlite3
import time
import threading
import numpy as np

exp = 3
# Rounds to 10^exp-th decimal. Currently set to 3.
# Change if parameters or output need more precision
def roundTo(r):
    return int(r*(10**exp))/(10.0**exp)

# called from main thread. helper method.
# Makes the next compute message to be sent to a client
def createComputeMsg(parameterSets, computing, lock, count, n):
    if not parameterSets.empty():
        msg = "compute"
        nextPSet = parameterSets.get()
        for i in range(len(nextPSet)):
            p = nextPSet[i]
            if not i:
                msg += " "+str(int(p))
                nextPSet[i] = int(p)
            else:
                msg += " "+str(roundTo(p))
                nextPSet[i] = roundTo(p)
##        print "nextPSet", nextPSet
        with lock:
            computing+=[nextPSet]
    else:
        print 1
        # if the parameterSets queue is empty, check the computing list.
        # if a client disconnects before sending back data, the pset will
        # remain on the computing list, and we must recompute it. 
        with lock:
            if not count[0] == n:
                for c in computing:
                    parameterSets.put(c)
                computing = []
                msg = "compute"
                nextPSet = parameterSets.get()
                for i in range(len(nextPSet)):
                    p = nextPSet[i]
                    if not i:
                        msg += " "+str(p)
                    else:
                        msg += " "+str(roundTo(p))
                computing += [nextPSet]
            # if computing is also empty, then we're done
            else:
                msg = "done"
                print "Done."
    return msg+"\n"

# called from main thread. helper method.
def createTable(parameterList, dbName):
    conn = sqlite3.connect(dbName)
    try:
        c = conn.cursor()
        try:
            query = "CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, patient text,"
            for p in parameterList:
                query += " "+p+" num,"
            query += " output num)"
            c.execute(query)
            conn.commit()
            c.close()
        except:
            c.close()
            raise
        conn.close()
    except:
        conn.close()
        raise

# called from store thread. Puts data into database
def insertInto(tokens, computing, parameterList, dbName, lock, count):
    fileNum = tokens[1]
    patientName = "Alg_ ("+str(fileNum)+")"
    pSet = tokens[2:len(tokens)-1]
    pSet = [float(x) for x in pSet]
    output = tokens[len(tokens)-1]
    
    lock.acquire()

    # confirm that we got back data for the right pSet
    if [int(fileNum)]+pSet in computing:
        computing.remove([int(fileNum)]+pSet)
        lock.release()
        
        conn = sqlite3.connect(dbName)
        try:
            c = conn.cursor()
            try:
                query = "SELECT COUNT(id) FROM results WHERE patient='"+patientName+"'"
                for i in range(len(pSet)):
                    query += " AND "+str(parameterList[i]) +"="+str(pSet[i])
                a = c.execute(query)
                if a.fetchone()[0] == 0:                
                    query = "INSERT INTO results (patient,"
                    for p in parameterList:
                        query += " "+p+","
                    query += " output) VALUES ("
                    query += "'"+patientName+"',"
                    for p in pSet:
                        query += " "+str(p)+","
                    query += " "+str(output)+")"
                    c.execute(query)
    ##                    print "stored,",tokens
                    count[0] = count[0]+1
                    conn.commit()
                    c.close()
            except:
                c.close()
                raise
            conn.close()
        except:
            conn.close()
            raise
    else:
        lock.release()


# Thread to handle storing data into the database. Reduces the time that clients wait for new pSet
class storingThread(threading.Thread):

    def __init__(self, storeQueue,computing, parameterList, dbName, lock, count):
        threading.Thread.__init__(self)
        self.queue = storeQueue
        self.computing = computing
        self.pList = parameterList
        self.dbName = dbName
        self.lock = lock
        self.count = count
        
    def run(self):
        while True:
            tokens = self.queue.get()
            insertInto(tokens, self.computing, self.pList, self.dbName, self.lock, self.count)
            
# main server/thread
# parameterSets is the queue of parameter sets that we want the client wants to compute in.
#   A parameter set is of the form (file number, parameter1, parameter2,...)
# parameterList is the list of names of the parameters. These will be names of database table columns.
#   Do not include column for file name/patient. 
# dbName is directory/name of database to save data in.
def server(parameterSets, parameterList, dbName):
    createTable(parameterList, dbName)
    print "Created table"
    n = parameterSets.qsize()
    count = [0]                     # program will finish when count[0] == n
    lock = threading.Lock()         # lock for the list computing
    storeQueue = Queue.Queue()      # queue of data to be stored into database
    computing = []                  # pSets that are currently out to a client and are being computed.
                                    # A pSet is removed from this list once client returns data, and
                                    # the pSet remains on the list if client disconnects/crashes/times
                                    # out before returning data.

    # separate thread for handling database
    t = storingThread(storeQueue,computing, parameterList, dbName,lock,count)
    t.setDaemon(True)
    t.start()
    
    try:
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
                    print "new client connected"

                    # send client a point to compute when it connects
                    msg = createComputeMsg(parameterSets, computing, lock, count, n)
                    new_socket.sendall(msg)
                    print "sent:", msg
                else:
                    start_time = time.time()
                    i.settimeout(2)     # will time out after 2 seconds
                    try:
                        # wait for data
                        data = ""
                        while data[len(data)-1] != "\n":
                            data = i.recv(1024)
                        print "received:",data
                        tokens = data.split(" ")

                        # send next point to be computed
                        msg = createComputeMsg(parameterSets, computing, lock, count, n)
                        i.sendall(msg)
                        print "sent:", msg

                        # data put on queue to be stored
                        if tokens[0] == "store":
                            storeQueue.put(tokens)
                        elif tokens[0] == "resend":
                            pass
                        else:
                            print "check message"
                        
                    except:
                        # disconnect the client if it times out
                        open_sockets.remove(i)
                        print "Connection closed"
##                    print "Time:",time.time()-start_time
    finally:
        for s in open_sockets:
            s.close()
        listening_socket.close()
        raise

if __name__=="__main__":
    a = 1
    if a == 0:
        queue = Queue.Queue()
        for i in range(1000):
            queue.put([i,float(2*i)])
        server(queue, ["b"], "test.db")
    elif a ==1:
        #files
        fileListName = "trainNames.csv"
        allFileNames = []
        with open(fileListName, 'r') as fh:
            for line in fh:
                allFileNames.append(line.strip())
        allFileNames = [int(x.split('_')[1]) for x in allFileNames]

        print "Creating parameter sets."
        dropTimes = np.arange(10,18.5,.5)#[13.6]#17
        dropPercents = np.arange(.35,.67,.02)#[.590000003]#17
        topAverageRates = [.92,.94,.96,.97,.98,.985,.99,.994,.996]#9
        eventLengthMins = np.arange(.8,2.2,.2)#[.80000000004]#7
        eventLengthMaxs = np.arange(30.0,80,10)#[30]#5
        
##        dropTimes = np.arange(10,18,2)#[13.6]#17
##        dropPercents = np.arange(.35,.65,.1)#[.590000003]#17
##        topAverageRates = [.98,.985,.99]#9
##        eventLengthMins = np.arange(.8,2.8,1)#[.80000000004]#7
##        eventLengthMaxs = np.arange(30.0,90,20)#[30]#5
        parameterSets = [[fn, dt, dp, ar, ml, mL] for fn in allFileNames for 
                         dt in dropTimes for dp in dropPercents for ar in topAverageRates
                         for ml in eventLengthMins for mL in eventLengthMaxs]
        queue = Queue.Queue()
        for pSet in parameterSets:
            queue.put(pSet)
        
        parameterList = ['dropTime', 'dropPercent', 'Avgrate', 'minLen', 'maxLen']
        print "Server Started"
        server(queue, parameterList, "db/parameterSearch.db")

