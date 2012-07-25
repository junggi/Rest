import sys
import numpy as np
from datetime import datetime, timedelta
from Events import *

class LogReader(object):
    def __init__(self, filename):
        self.filename = filename
        self.sampleRate = 5

        # Dictionary of classes to use for each keyword
        self.eventTypes = {'Stage': StageEvent,
                           'Arousal': ArousalEvent,
                           'Respiratory': RespiratoryEvent,
                           'Position': PositionEvent,
                           'Snore': SnoreEvent,
                           'Desaturation': DesaturationEvent,
                           'PLM': PLMEvent,
                           'Bilevel': BilevelEvent,
                           'Montage': MontageEvent,
                           'High': FilterEvent,
                           'Low': FilterEvent,
                           'Sensitivity': SensitivityEvent,
                           'EYES': CalibrationEvent,
                           'RIGHT': CalibrationEvent,
                           'LEFT': CalibrationEvent,
                           'R': CalibrationEvent,
                           'L': CalibrationEvent,
                           'LOOK': CalibrationEvent,
                           'MOUTH': CalibrationEvent,
                           'HOLD': CalibrationEvent,
                           'GRIT': CalibrationEvent,
                           'BLINK': CalibrationEvent,
                           'COUGH': CalibrationEvent,
                           'TIR/TOR': CalibrationEvent,
                           'LIGHTS': CalibrationEvent,
                           'NASAL': CalibrationEvent,
                           'pause': CalibrationEvent
                           }

        self.events = {}

        self.processLog()

    def __repr__(self):
        return "<LogReader: {}>".format(self.filename)
    
    def processLog(self):
        self.startTime = None
        with open(self.filename, 'r') as fh:
            for line in fh:
                self.processLine(line)
            
    def processLine(self, line):
        # Get first word of description
        time, description = line.split('\t')
        keyword = description.split()[0]
      
        # if multiple 'New' external adjustment events are in the line,
        # process each one by recursively calling processLine
        if (keyword == 'New') or (keyword == '"New'): 
            description = description.strip('"')
            n = description[4:].find('New')
            if n > -1:
                newDescription = description[n+4:].strip()
                self.processLine(time+"\t"+newDescription)

            
        # Check that keyword doesn't start with '*' or '"New'
        if (keyword == "*") or (keyword == "New") or (keyword == '"New'):
            keyword = description.split()[1]
            description = " ".join(description.split()[1:])
              
        # Parse time to samples ellapsed since start
        time = datetime.strptime(time.strip(), '%H:%M:%S.%f')
        if self.startTime is None:
            # Store starting time of study
            self.startTime = time
        if time < self.startTime:
            # Check if we moved past midnight
            time += timedelta(days=1)
        time = time-self.startTime
        timestamp =  int(time.total_seconds()*self.sampleRate)

        event = self.eventTypes.get(keyword, Event)(description.strip(), timestamp)                    
        if event.eType in self.events:
            self.events[event.eType].append(event)
        else:
            self.events[event.eType] = [event]

    # Get list of types of events
    def getEventTypes(self):
        return self.events.keys()

    # Return time-series array of events at sampleRate
    def getTimeSeries(self, eventType, outLength, filt=None):
        out = np.zeros(outLength)
        try:
            eventList = self.events[eventType]
        except KeyError:
            return out

        if eventType in ['Stage', 'Position']:
            # These are stateful (stay the same until changed)
            for i in range(len(eventList)-1):
                start = eventList[i].time
                end = eventList[i+1].time
                out[start:end] = eventList[i].value
            out[eventList[-1].time:] = eventList[-1].value
            
        else:
            # These are isolated events (only duration matters)
            for event in eventList:
                if (filt is None) or (event.type in filt):
                    out[event.time:event.time+event.duration*self.sampleRate] = 1

        return out

if __name__ == "__main__":
    import pylab as plt

    #a = LogReader('../Data/Logs/AT_Apr12.parsed.txt')
    #a = LogReader('Data\\AT_Apr12.txt')
    a = LogReader('C:/Users/Lukulele/workspace/restdevices/RespData/Data/RL_Apr12.txt')
    b = LogReader('C:/Users/Lukulele/workspace/restdevices/RespData/Data/SL_Mar12.txt')
    c = LogReader('C:/Users/Lukulele/workspace/restdevices/RespData/Data/SP_Jan12.txt')
    d = LogReader('C:/Users/Lukulele/workspace/restdevices/RespData/Data/TF_Mar12.txt')
    #a.processLog()

    #t = 'Respiratory Event'
    #t = 'Snore'
    # z = a.getTimeSeries(t, 150000)
    # z1 = a.getTimeSeries(t, 150000, filt=['Obstructive Apnea'])
    # z2 = a.getTimeSeries(t, 150000, filt=['Central Apnea'])
    # z4 = a.getTimeSeries(t, 150000, filt=['Hypopnea'])
    # z3 = a.getTimeSeries(t, 150000, filt=['Mixed Apnea'])
    # z5 = a.getTimeSeries(t, 150000, filt=['RERA'])
    
    # plt.plot(z1, 'r', linewidth=15, alpha=0.5)
    # plt.plot(z2, 'g', linewidth=15, alpha=0.5)
    # plt.plot(z3, 'b', linewidth=15, alpha=0.5)
    # plt.plot(z4, 'm', linewidth=15, alpha=0.5)
    # plt.plot(z5, 'y', linewidth=15, alpha=0.5)
    # plt.plot(z, 'black')
    # plt.show()
    
