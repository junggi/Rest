from RespDataMedfilt import RespData as rdm
from RespDataRCFilt import RespData as rdr
from RespDataFiltfilt import RespData as rdf
import pylab as plt
import os

##files = ['B23 Night 1', 'ALG_ (26)','ALG_ (1)','Carson 2012-04-17','DulcieTankTest']
files = ['B23 Night 1']
datadir = ''
for fname in files:
    print fname
    print 1
    a = rdm(os.path.join(datadir, "{}.resp".format(fname)), verbose=False)
    print 2
    b = rdr(os.path.join(datadir, "{}.resp".format(fname)), verbose=False)
    print 3
    c = rdf(os.path.join(datadir, "{}.resp".format(fname)), verbose=False)
    for i in range(2):
        plt.figure()
        ax1 = plt.subplot(411)
        ax1.plot(a.data[:,i])
        ax1.plot([0 for x in a.data[:,i]])
        ax2 = plt.subplot(412, sharex=ax1)
        ax2.plot(b.data[:,i])
        ax3 = plt.subplot(413, sharex=ax1)
        ax3.plot(c.data[:,i])
        ax4 = plt.subplot(414, sharex=ax1)
        ax4.plot(a.rawData[:,i])
plt.show()
