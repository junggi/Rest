import sqlite3

def checkFullDataSet(dbName, n):
##    dbName = "thomas" #"carson", "lab", "pablo", "thomas"
    filename = "rawData/full parameter search results"
    fileListName = "trainNames.csv"
    allFileNames = []
    with open(fileListName, 'r') as fh:
        for line in fh:
            allFileNames.append(line.strip())
    allFileNames = [x.split('_')[0]+'_ ('+x.split('_')[1]+')' for x in allFileNames]
    if dbName == "carson":
        patients = allFileNames[0:12]+allFileNames[42:46]
    elif dbName == "lab":
        patients = allFileNames[12:18]
    elif dbName == "pablo":
        patients = allFileNames[18:30]
    elif dbName == "thomas":
        patients = allFileNames[30:42]
    else:
        patients = allFileNames
    print patients
    conn = sqlite3.connect(filename + " - " + dbName+".db")
    c = conn.cursor()
    try:

    ##    patients = []
    ##    for row in c.execute('''SELECT patient FROM results'''):
    ##        if row[0] not in patients:
    ##            print row
    ##            patients += [row[0]]
        for patient in patients:
            t= c.execute('''SELECT COUNT(id) FROM results WHERE patient = (?)''', (patient,)).fetchone()
            if t[0] == 0 :
                print "Data file "+patient + "is missing."
            elif t[0] < n:
                print "Data file "+patient +" is incomplete with " +str(t[0])+" points. Deleting..."
                c.execute('''DELETE FROM results WHERE patient=(?)''',(patient,))
                conn.commit()
            elif t[0] > n:
                print "Data file "+patient +" has more than "+str(n)+"points. Check file."
            else:
                print patient + ": "+str(t[0])
        print "all tests done"
        c.close()
        conn.close()
    except:
        c.close()
        conn.close()

if __name__ == "__main__":
    dbNames = ["merged"]#["carson", "lab", "pablo", "thomas","lab2","pablo2"]
    n = 91035
    for dbName in dbNames:
        checkFullDataSet(dbName,91035)
