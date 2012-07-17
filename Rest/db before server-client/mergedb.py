import sqlite3

def mergeDB(file):
    conn = sqlite3.connect("rawData/full parameter search results - merged.db")
    c = conn.cursor()
    try:
        c.execute('''CREATE TABLE IF NOT EXISTS results
                (id INTEGER PRIMARY KEY, patient text, staging text, dropTime num, dropPercent num,
                Avgrate num, minLen num, maxLen num, output num)''')
        
        c.execute('''ATTACH ? as toMerge''',(file,))
        c.execute('''INSERT INTO results SELECT NULL, patient, staging, dropTime, dropPercent,
                                Avgrate, minLen, maxLen, output FROM toMerge.results ''')
        
        c.execute('''DETACH DATABASE toMerge''')
        conn.commit()
        c.close()
        conn.close()
        print "Merge complete"
    except:
        c.close()
        conn.close()
        raise

if __name__ == "__main__":
    names = ["carson", "lab", "thomas","pablo","lab2","pablo2"]
    for name in names:
        print name
        mergeDB('./rawData/full parameter search results - '+name+'.db')
