import sqlite3

##dbNames = ["carson", "lab", "pablo", "thomas","lab2","pablo2"]
##for dbName in dbNames:
##    print dbName
##    filename = "rawData/full parameter search results"
##    conn = sqlite3.connect(filename + " - " + dbName+".db")
##    c = conn.cursor()
##    t = c.execute("SELECT COUNT(id) FROM results WHERE patient='Alg_ (10)'").fetchone()
##    print "Alg_ (10): "+str(t[0])
##    t = c.execute("SELECT COUNT(id) FROM results WHERE patient='Alg_ (11)'").fetchone()
##    print "Alg_ (11): "+str(t[0])
##    t = c.execute("SELECT COUNT(id) FROM results WHERE patient='Alg_ (121)'").fetchone()
##    print "Alg_ (121): "+str(t[0])
##
dbName = "pablo"
filename = "rawData/full parameter search results"
conn = sqlite3.connect(filename + " - " + dbName+".db")
c = conn.cursor()

patients = ["Alg_ (10)","Alg_ (11)", "Alg_ (121)"]
##for patient in patients:
##    c.execute('''DELETE FROM results WHERE patient=(?)''',(patient,))
##    conn.commit()

t = c.execute("SELECT COUNT(id) FROM results WHERE patient='Alg_ (10)'").fetchone()
print "Alg_ (10): "+str(t[0])
t = c.execute("SELECT COUNT(id) FROM results WHERE patient='Alg_ (11)'").fetchone()
print "Alg_ (11): "+str(t[0])
t = c.execute("SELECT COUNT(id) FROM results WHERE patient='Alg_ (121)'").fetchone()
print "Alg_ (121): "+str(t[0])
