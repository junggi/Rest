from RespData.RespData import RespData
from ApneaDetector.ApneaDetector import detectApneas
##from getREI3 import getREI
import numpy as np
import pylab as plt
import csv
import sqlite3

#parameters
sampleRate = 5
channel = 1
##dropTimes = np.arange(13.5, 15.1 , 0.1)
##dropPercents = np.arange(0.45, 0.65, 0.01)
dropTimes = np.arange(10,18.5,.5)#[13.6]#17
dropPercents = np.arange(.35,.67,.02)#[.590000003]#17
topAverageRates = [.92,.94,.96,.97,.98,.985,.99,.994,.996]#9
eventLengthMins = np.arange(.8,2.2,.2)#[.80000000004]#7
eventLengthMaxs = np.arange(30.0,80,10)#[30]#5
stagings = ['NONE']#['NONE','LAB']

# index of the processor.
index = 0

respFolder = "RespFiles\\"
stageFolder = "Staging\\"
logFolder = "Logs\\"

#files
fileListName = "trainTestNames.csv"
allFileNames = []
with open(fileListName, 'r') as fh:
    for line in fh:
        allFileNames.append(line.strip().split(','))
allFileNames = [x[0].split('_')[0]+'_ ('+x[0].split('_')[1]+')' for x in allFileNames]
fileNames = allFileNames[2*index:2*(index+1)]
if index<=3:
    fileNames += [allFileNames[index+42]]
print fileNames
##fileNames = [filename[0]+".resp" for filename in fileList]

conn = sqlite3.connect("full parameter search results.db")
try:
    c = conn.cursor()
    try:
        c.execute('''CREATE TABLE IF NOT EXISTS results
                (id INTEGER PRIMARY KEY, patient text, staging text, dropTime num, dropPercent num,
                Avgrate num, minLen num, maxLen num, output num)''')
        conn.commit()
        c.close()
    except:
        c.close()

    for fileName in fileNames:

        print "current file is: " + fileName

        #load data
        respFileName = respFolder+fileName+".resp"
        data = RespData(respFileName)

        #Get matching peak data
        temp = data.getFeature('upperPeakTimes')
        upperPeakTimes = np.zeros(data.data.shape[0])           #pair peaks down
        upperPeakTimes[temp] = 1                                #cut down to length
        upperPeakTimes = np.nonzero(upperPeakTimes)[0]          #get index
        
        data.loadLog(logFolder+fileName+".txt")

        # Full parameter search. Save results into database.
        for dropTime in dropTimes:
            for dropPercent in dropPercents:
                for topAverageRate in topAverageRates:
                    for eventLengthMin in eventLengthMins:
                        for eventLengthMax in eventLengthMaxs:
                            for staging in stagings:
                                options = {
                                    'dropTime':         dropTime,
                                    'dropPercent':      dropPercent,
                                    'topAverageRate':   topAverageRate,
                                    'eventLengthMin':   eventLengthMin,
                                    'eventLengthMax':   eventLengthMax,
                                    'staging':          staging,
                                }
                                
                                apneaIndex = detectApneas(data, options)[0]
                                while True:
                                    c = conn.cursor()
                                    try:
                                        t=(fileName,staging, dropTime,dropPercent,topAverageRate,eventLengthMin,
                                           eventLengthMax,apneaIndex,)
                                        print t
                                        c.execute('''INSERT INTO results (patient, staging, dropTime, dropPercent,
                                                     Avgrate, minLen, maxLen, output) VALUES (?,?,?,?,?,?,?,?)''',t)
                                        conn.commit()
                                        c.close()
                                        break
                                    except:
                                        c.close()
    conn.close()
except:
    conn.close()
    raise

raw_input("Done! Press any key to exit.")
