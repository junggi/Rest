import sqlite3
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import operator
import matplotlib.cm as cm
import matplotlib.mlab as mlab


parameterList = ['dropTime', 'dropPercent', 'Avgrate', 'minLen', 'maxLen']
mildCutoff = 3
moderateCutoff = 15
severeCutoff = 25
minScore = 0           # R2 scoring
exp = 3
# Rounds to 10^exp-th decimal. Currently set to 3.
# Change if parameters or output need more precision
def roundTo(r):
    return int(r*(10**exp))/(10.0**exp)

def closestToAHI(output, ahi):
    return abs(ahi-output)

def R2(x,y, plot = False):
    return R2in3regions(x,y,findRSquared, plot)

def spearman(x,y,plot = False):
    return R2in3regions(x,y,findSpearman, plot)

def pearson(x,y,plot = False):
    return R2in3regions(x,y,findPearson, plot)

def hasApnea(a):
    if a>5:
        return 1
    else:
        return 0

def findPatients(patientsList):
    seen = []
    for patient in patientsList:
        if patient not in seen:
            seen += [patient]
    return seen


def findRSquared(x,y):
    coeffs = np.polyfit(x,y,1)

    p = np.poly1d(coeffs)
    yhat = [p(z) for z in x]
    ybar = float(sum(y))/len(y)
    ssreg = sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = sum([ (yi - ybar)**2 for yi in y ])
    rs = ssreg/sstot
    return rs

def findSpearman(x,y):
    xbar = float(sum(x))/len(x)
    ybar = float(sum(y))/len(y)
    p = (sum( [ (x[i]-xbar)*(y[i]-ybar) for i in range(len(x))] ) ) / ( (sum( [ (xi-xbar)**2 for xi in x] )*(sum( [ (yi-ybar)**2 for yi in y] ) ) )**(1/2) )
    return abs(p)

def findPearson(x,y):
    return abs(np.cov(x,y)[0][1] / (np.std(x) * np.std(y)))

# f = findRSquared, findSpearman, findPearson
def R2in3regions(x,y,f, plot = False):
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    for i in range(len(x)):
        if y[i]<mildCutoff:
            x1 += [x[i]]
            y1 += [y[i]]
        elif y[i]<moderateCutoff:
            x2 += [x[i]]
            y2 += [y[i]]
        else:
            x3 += [x[i]]
            y3 += [y[i]]
    score1 = 0
    score2 = 0
    score3 = 0
    if len(x1)>1:
        score1 = f(x1,y1)
    if len(x2)>1:
        score2 = f(x2,y2)
    if len(x3)>1:
        score3 = f(x3,y3)
    if plot:
        plt.figure()
        if len(x1)>0:
            fit1 = np.polyfit(x1, y1,1)
            line1 = np.poly1d(fit1)
            plt.plot(x1,y1,'bo')
            plt.plot(x1, line1(np.array(x1)), '--c')
            plt.plot(x1, x1, '-k')
        if len(x2)>0:
            fit2 = np.polyfit(x2, y2,1)
            line2 = np.poly1d(fit2)
            plt.plot(x2,y2,'bo')
            plt.plot(x2, line2(np.array(x2)), '--c')
            plt.plot(x2, x2, '-k')
        if len(x3)>0:
            fit3 = np.polyfit(x3, y3,1)
            line3 = np.poly1d(fit3)
            plt.plot(x3,y3,'bo')
            plt.plot(x3, line3(np.array(x3)), '--c')
            plt.plot(x3, x3, '-k')
        plt.title("Output v AHI")
    if score1>=minScore and score3>=minScore:
        return score2
    else:
        return 0
    
def findBest(data, m):
    d = {}
    for i in range(data.shape[1]):
        parameterSet = (data[:,i][0],data[:,i][1],data[:,i][2],data[:,i][3],data[:,i][4])
        score = data[:,i][5]
        if parameterSet not in d:
            d[parameterSet] = 0
        d[parameterSet] = d[parameterSet]+score
    if m :
        best = min(d.iteritems(), key = operator.itemgetter(1))[0]
    else:
        best = max(d.iteritems(), key = operator.itemgetter(1))[0]
    return best, d[best]

# scoringType = 0 is like abs(output-ahi): only one point needed to score
# scoringType = 1 is like RSquared value: all patients needed to score the parameter set
def extractData(filename, scoringFunction, scoringType):
    conn = sqlite3.connect(filename)
    c = conn.cursor()
    try:
        i = 0
        scoredData = [[],[],[],[],[],[]]            # p1 p2 p3 p4 p5 score
        data = [[],[],[],[],[],[],[]]               # p1 p2 p3 p4 p5 output ahi
        patients = []
        x = []                                      # list with output of each patient for one parameter set
        y = []                                      # list with ahi of each patient for one parameter set
        tempScore = 0
        currentParameterSet = None
        print c.execute('''SELECT COUNT(id) FROM results''').fetchone()
        for row in c.execute('''SELECT results.dropTime, results.dropPercent, results.Avgrate, results.minLen, results.maxLen,
                                    results.output, ahi.ahi, ahi.patient FROM results JOIN ahi ON results.patient=ahi.patient
                                    ORDER BY results.dropTime , results.dropPercent, results.Avgrate,
                                    results.minLen, results.maxLen,ahi.patient'''):
            
            outliers = [26,14,61,63,83,27]
            outliers = ["Alg_ ("+str(num)+")" for num in outliers]
            if row[7] not in outliers:
                for j in range(7):
                    data[j] += [row[j]]
                patients += [row[7]]
                if not currentParameterSet:
                    currentParameterSet = (row[0], row[1], row[2], row[3], row[4])
                    tempScore = 0
                if currentParameterSet != (row[0], row[1], row[2], row[3], row[4]):
                    for j in range(5):
                        scoredData[j] += [currentParameterSet[j]]
                    if not scoringType:
                        scoredData[5] += [tempScore]
                        tempScore = 0
                    else:
                        scoredData[5] += [scoringFunction(x,y)]
                        x = []
                        y = []
                    currentParameterSet = (row[0], row[1], row[2], row[3], row[4])
                    
                if not scoringType:
                    tempScore += scoringFunction(row[5], row[6])
                else:
                    x += [row[5]]
                    y += [row[6]]
                if i % 100000 == 0 :
                    print i
                i+=1

        # last iteration
        currentParameterSet = (row[0], row[1], row[2], row[3], row[4])
        for j in range(5):
            scoredData[j] += [currentParameterSet[j]]
        scoredData[5] += [tempScore]
        tempScore = 0

        
        scoredData = np.array(scoredData)
        data = np.array(data)
        print scoredData.shape
        c.close()
        conn.close()
    except:
        c.close()
        conn.close()
        raise
    return scoredData, data, np.array(patients)

def plot6Subplots(scoredData, data, bestParameterSet, patients):
    plt.figure()
    for j in range(5):
        x = []
        y = []
        print scoredData.shape[1]
        for i in range(scoredData.shape[1]):
            if abs((scoredData[:,i][(j+1)%5])-(bestParameterSet[(j+1)%5]))<.00001 and abs((scoredData[:,i][(j+2)%5]) -(bestParameterSet[(j+2)%5]))<.00001 and \
               abs((scoredData[:,i][(j+3)%5]) - (bestParameterSet[(j+3)%5]))<.00001 and abs((scoredData[:,i][(j+4)%5]) - (bestParameterSet[(j+4)%5]))<.00001:
                x += [ scoredData[:,i][j] ]
                y += [ scoredData[:,i][5] ]
                if not i:
                    print x
                    print y
        plt.subplot(3,2,j+1)
        plt.plot(x,y,'bo')
        plt.axvline(x = bestParameterSet[j])
        plt.title(parameterList[j])
        print len(x)
        print x

    x = []
    y = []
    labels = []
    for i in range(data.shape[1]):
        if abs((data[:,i][0]) - (bestParameterSet[0]))<.00001 and abs((data[:,i][2]) - (bestParameterSet[2]))<.00001 \
           and abs((data[:,i][1]) - (bestParameterSet[1]))<.00001 and abs((data[:,i][3]) - (bestParameterSet[3]))<.00001 \
           and abs((data[:,i][4]) - (bestParameterSet[4]))<.00001:
            x += [data[:,i][5]]
            y += [data[:,i][6]]
            labels += [patients[i]]
        
    x = np.array(x)
    y = np.array(y)
    plt.subplot(3,2,6)
    fit = np.polyfit(x,y,1)
    line = np.poly1d(fit)
    plt.plot(x,y,'bo')
    plt.plot(x,line(x), '--c')
    plt.plot(x,x,'-k')
    plt.title("Output v AHI")
    for label, xi, yi in zip(labels, x, y):
        plt.annotate(label, xy = (xi, yi), textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

def plotScoringFunction(data, bestParameterSet, patients, scoringFunction):
    x = []
    y = []
    labels = []
    for i in range(data.shape[1]):
        if abs((data[:,i][0]) - (bestParameterSet[0]))<.00001 and abs((data[:,i][2]) - (bestParameterSet[2]))<.00001 \
           and abs((data[:,i][1]) - (bestParameterSet[1]))<.00001 and abs((data[:,i][3]) - (bestParameterSet[3]))<.00001 \
           and abs((data[:,i][4]) - (bestParameterSet[4]))<.00001:
            x += [data[:,i][5]]
            y += [data[:,i][6]]
            labels += [patients[i]]
            
    scoringFunction(x,y, plot = True)
    for label, xi, yi in zip(labels, x, y):
        plt.annotate(label, xy = (xi, yi), textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

def plot3D(scoredData, bestParameterSet):
    avgRate = [.92,.94,.96,.97,.98,.985,.99,.994,.996]
    fig = plt.figure()
    for j in range(9):
        x = []
        y = []
        z = []
        for i in range(scoredData.shape[1]):
            if abs((scoredData[:,i][2]) - (avgRate[j]))<.00001 and abs((scoredData[:,i][3]) - (bestParameterSet[3]))<.00001 and \
                   abs((scoredData[:,i][4]) - (bestParameterSet[4]))<.00001:
                x += [scoredData[:,i][0]]
                y += [scoredData[:,i][1]]
                z += [scoredData[:,i][5]]
        ax = fig.add_subplot(3,3,j+1, projection = '3d')
        ax.scatter(x,y,z,label = parameterList[2]+" = "+str(avgRate[j]))
        ax.set_xlabel(parameterList[0])
        ax.set_ylabel(parameterList[1])
        ax.set_zlabel("Score"+str(j))
        
def plotHeatmap(scoredData, bestParameterSet):
    avgRate = [.92,.94,.96,.97,.98,.985,.99,.994,.996]
    fig = plt.figure()
    for j in range(9):
        x = np.arange(10,18.5,.5)
        y = np.arange(.35,.67,.02)
        z = np.zeros((len(x),len(y)))
        k = 0
        for i in range(scoredData.shape[1]):
            if abs((scoredData[:,i][2]) - (avgRate[j]))<.00001 and abs((scoredData[:,i][3]) - (bestParameterSet[3]))<.00001 and \
                   abs((scoredData[:,i][4]) - (bestParameterSet[4]))<.00001:
                z[k/len(y)][k%len(y)] = scoredData[:,i][5]
                k += 1
        ax = fig.add_subplot(3,3,j+1)
        CS = ax.contour(x,y,z,label = parameterList[2]+" = "+str(avgRate[j]))
        ax.set_xlabel(parameterList[0])
        ax.set_ylabel(parameterList[1])
        plt.clabel(CS, inline=1, fontsize=10)
def plotResults(filename,scoringFunction = R2, scoringType = 1, m = False):
    
    scoredData ,  data, patients= extractData(filename,scoringFunction, scoringType)
    
    bestParameterSet , bestScore = findBest(scoredData, m)
    print bestParameterSet, bestScore

    plot6Subplots(scoredData, data, bestParameterSet, patients)
    
    if scoringType == 1:
        plotScoringFunction(data, bestParameterSet, patients, scoringFunction)
    
##    plot3D(scoredData, bestParameterSet)

    plotHeatmap(scoredData, bestParameterSet)
    
    return bestParameterSet, data, scoredData


if __name__ == "__main__":
    names = ["merged"]#["carson", "lab", "thomas", "merged"]
    for name in names:
        print name
        filename = "db before server-client/AHI data/full parameter search results - "+name+".db"
        
##        scoredData ,  data, patients= extractData(filename,scoringFunction = R2, scoringType = 1)
##        
##        x = []
##        y = []
##        labels = []
##        outliers = [26,14,61,63,83,27]
##        outliers = ["Alg_ ("+str(num)+")" for num in outliers]
##        for i in range(data.shape[1]):
##            if abs((data[:,i][0]) - (18))<.00001 and abs((data[:,i][2]) - (.98))<.00001 \
##               and abs((data[:,i][1])- (.55))<.00001 and abs((data[:,i][3]) - (2))<.00001 \
##               and abs((data[:,i][4]) - (50))<.00001:
##                if patients[i] not in outliers:
##                    x += [data[:,i][5]]
##                    y += [data[:,i][6]]
##                    labels += [patients[i]]
##                
##        score = R2(x,y, plot = True)
##        print score
##        for label, xi, yi in zip(labels, x, y):
##            plt.annotate(label, xy = (xi, yi), textcoords = 'offset points', ha = 'right', va = 'bottom',
##            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        
        plotResults(filename, m = False)
##        print "done", 1
##        plotResults(filename, scoringFunction = spearman, m  = False)
##        print "done", 2
##        plotResults(filename, scoringFunction = pearson, m = False)
##        print "done", 3
##        plotResults(filename, scoringFunction = closestToAHI, scoringType = 0)
    plt.show()
