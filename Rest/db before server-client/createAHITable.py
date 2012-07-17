import sqlite3

def createAHITable(filename, copyfilename):
    conn = sqlite3.connect(filename)
    c = conn.cursor()
    try:
        c.execute('''CREATE TABLE IF NOT EXISTS results
                (id INTEGER PRIMARY KEY, patient text, staging text, dropTime num, dropPercent num,
                Avgrate num, minLen num, maxLen num, output num)''')
        
        c.execute('''ATTACH ? as toMerge''',(copyfilename,))
        c.execute('''INSERT INTO results SELECT NULL, patient, staging, dropTime, dropPercent,
                                Avgrate, minLen, maxLen, output FROM toMerge.results ''')
        
        c.execute('''DETACH DATABASE toMerge''')

        c.execute('''CREATE TABLE IF NOT EXISTS ahi
                    (a_id INTEGER PRIMARY KEY, patient text, ahi num)''')

        fileListName = "trainNames.csv"
    ##    fileListName = "sampleNames.csv"
        allFileNames = []
        with open(fileListName, 'r') as fh:
            for line in fh:
                allFileNames.append(line.strip())
        allFileNames = [x.split('_')[0]+'_ ('+x.split('_')[1]+')' for x in allFileNames]

##        allFileNames = allFileNames[0:2]+[allFileNames[4]]+[allFileNames[6]]+allFileNames[8:11]+allFileNames[12:18]+allFileNames[30:42]
##        print allFileNames
##        print len(allFileNames)

        fileNameAHI = "trainAHI.csv"
    ##    fileNameAHI = "sampleAHI.csv"
        allAHI = []
        with open(fileNameAHI, 'r') as fh:
            for line in fh:
                allAHI.append(float(line.strip()))
##        allAHI = allAHI[0:2]+[allAHI[4]]+[allAHI[6]]+allAHI[8:11]+allAHI[12:18]+allAHI[30:42]
##        print allAHI
##        print len(allAHI)

        for i in range(len(allFileNames)):
            t = (allFileNames[i],allAHI[i])
            c.execute('''INSERT INTO ahi (patient, ahi) VALUES (?,?)''',t)
        conn.commit()
        c.close()
        conn.close()
    except:
        c.close()
        conn.close()
        raise

if __name__=="__main__":
    names = ["merged","pablo"]
    for name in names:
        print name
        filename = "AHI data/full parameter search results - "+name+".db"
        copyfilename = 'rawData/full parameter search results - '+name+'.db'
        createAHITable(filename,copyfilename)
        print "done"
