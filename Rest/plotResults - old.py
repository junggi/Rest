import sqlite3
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import mlpy

parameterList = ['dropTime', 'dropPercent', 'Avgrate', 'minLen', 'maxLen']

conn = sqlite3.connect("full parameter search results.db")
c = conn.cursor()
# metrics = distance to AHI, distance to neighborhood
metric = 2
neighborhood = 5
cumulative = False
_3D = False
apneaThreshold = 5

def scoring(ahi,output,scoringType):
    if scoringType == 0:
        return ahi-output
    elif scoringType == 1:
        if ahi-neighborhood< output and output < ahi+neighborhood:
            return 0
        else:
            return min(abs(output-ahi-neighborhood),abs(output-ahi+neighborhood))
    elif scoringType == 2:
        return output
    else:
        print "invalid metric"
        raise

def hasApnea(ahi):
    if ahi<apneaThreshold:
        return "apnea"
    else:
        return "no apnea"
try:
    if cumulative:
        parameters = [[],[],[],[],[]]
        tempParameters = [{},{},{},{},{}]
        outputs = [[],[],[],[],[]]
        i=0
        currentPatient = None
        currentDataPoint = 0
        for row in c.execute('''SELECT results.dropTime, results.dropPercent, results.Avgrate, results.minLen, results.maxLen,
                                results.output, ahi.ahi, ahi.patient FROM results JOIN ahi ON results.patient=ahi.patient
                                ORDER BY ahi.patient'''):
            if row[7] != currentPatient:
                if currentPatient!=None:
                    for j in range(5):
                        for data in tempParameters[j]:
                            parameters[j] += [data]
                            outputs[j] += [tempParameters[j][data]]
                currentPatient = row[7]
            for j in range(5):
                if row[j] not in tempParameters[j]:
                    tempParameters[j][row[j]] = 0
                tempParameters[j][row[j]] = tempParameters[j][row[j]]+abs(scoring(row[6],row[5],metric))
            if i % 10000 == 0 :
                print i
            i+=1
                
        for i in range(5):
            plt.figure()
            plt.plot(parameters[i],outputs[i],'bo')
            plt.title(parameterList[i])
        plt.show()
            
##        # 3D
##        if _3D:
##            for i in range(5):
##                for j in range(5):
##                    if i >j :
##                        fig = plt.figure()
##                        ax = fig.add_subplot(111,projection='3d')
##                        ax.scatter(parameters[i].keys(),parameters[j].keys(), , label = parameterList[i] + ", "+ parameterList[j])
##                        ax.set_xlabel(parameterList[i])
##                        ax.set_ylabel(parameterList[j])
##                        ax.set_zlabel("Score")
##            plt.show()
            
    else:
        parameters = [[],[],[],[],[]]
        patients = {}
        outputs = []
        ahi = []
        i=0
        for row in c.execute('''SELECT results.dropTime, results.dropPercent, results.Avgrate, results.minLen, results.maxLen,
                                results.output, ahi.ahi, ahi.patient FROM results JOIN ahi ON results.patient=ahi.patient'''):
            if i % 1000 == 0 :
                for j in range(5):
                    parameters[j] += [row[j]]
                outputs += [scoring(row[6],row[5],metric)]
                ahi += [row[6]]

            if i % 10000 == 0 :
                print i
            i+=1
        # 3D
        if metric == 2:
            for j in range(5):
                seen = []
                for d in parameters[j]:
                    if d in seen:
                        break
                    seen += [d]
                    x = np.array([outputs[i] for i in range(len(parameters[j])) if parameters[j][i] == d])
                    y = [hasApnea(ahi[i]) for i in range(len(parameters[j])) if parameters[j][i] == d]
        if _3D:
            for i in range(5):
                for j in range(5):
                    if i >j :
                        fig = plt.figure()
                        ax = fig.add_subplot(111,projection='3d')
                        ax.scatter(parameters[i],parameters[j], outputs, label = parameterList[i] + ", "+ parameterList[j])
                        ax.set_xlabel(parameterList[i])
                        ax.set_ylabel(parameterList[j])
                        ax.set_zlabel("Score")
            plt.show()
        #2D
##        else:
##            for i in range(5):
##                plt.figure()
##                plt.plot(parameters[i], outputs, 'bo')
##                plt.title(parameterList[i])
##            plt.show()
    c.close()
    conn.close()
    
except:
    c.close()
    conn.close()
    raise
