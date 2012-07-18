import numpy
import pylab as plt
from RespData.RespData import RespData

points = [#(15,1),(16,1),(21,0),(21,1),(22,0),(22,1),(23,0),
          (23,1),(24,0),(24,1),(29,1),(31,0),(46,1),(55,1),
          (57,1),(63,0),(64,1),(69,1),(77,0),(108,0),(110,1)]
k=0
for point in points:
    i, j = point
    fname = "RespFiles/Alg_ ("+str(i)+").resp"
    a = RespData(fname)
    plt.figure(str(i)+": "+str(j))
    plt.plot(a.rawData[:,j])
    plt.plot([0 for x in a.rawData[:,j]])
##    plt.figure(str(i)+": "+str(2))
##    plt.plot(a.rawData[:,1])
##    plt.plot([0 for x in a.rawData[:,1]])
    k+=1
    print k
    if not k %7 :
        plt.show()
