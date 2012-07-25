class Event(object):
    eType = "Event"
    def __init__(self, description, time):
        self.fullDescription = description
        self.time = time
        self.duration = 1
        self.parse()

    def __repr__(self):
        return "<{:6} - {}: {}>".format(self.time, self.eType, self.description)

    # Read description and extract useful parameters
    def parse(self):
        self.description = self.fullDescription


class StageEvent(Event):
    eType = "Stage"
    sValues = {'None': 0,
               'N3': 1,
               'N2': 2,
               'N1': 3,
               'R': 4,
               'W': 5
               }
    def parse(self):
        stage = self.fullDescription.split()[-1]
        if stage == "Stage":
            stage = "None"
        if stage == "Mvt":
            stage = "None"
        self.value = self.sValues[stage]
        self.description = "{} {}".format(self.value, stage)


class ArousalEvent(Event):
    eType = "Arousal"
    def parse(self):
        try:
            self.duration = float(self.fullDescription.split()[3])
            self.description = "{:3.1f}".format(self.duration)
        except IndexError:
            self.description = "{} (Data Format Error)".format(self.fullDescription.split()[-1])


class RespiratoryEvent(Event):
    eType = "Respiratory Event"
    def parse(self):
        desc = map(lambda x:x.strip(), self.fullDescription.split('-'))
        try:
            self.duration = float(desc[1].split()[1])
            self.type = desc[2]
            self.desat = float(desc[3].split()[1])
            self.description = "{} - Dur: {:3.2f}, Desat: {:3.2f}".format(self.type, self.duration, self.desat)
        except IndexError:
            self.description = "{} (Data Format Error)".format(desc[-1])

class PositionEvent(Event):
    eType = "Position"

    pValues = {'Supine': 0,
               'Right': 1,
               'Left': 2,
               'Prone': 3,
               'Disconnect': 4,
               'Sitting': 5,
               }
    
    def parse(self):
        try:
            self.position = self.fullDescription.split()[2].strip()
            self.value = self.pValues[self.position]
            self.description = "{} {}".format(self.value, self.position)
        except IndexError:
            self.description = "{} (Data Format Error)".format(self.fullDescription.split()[-1])

class CalibrationEvent(Event):
    eType = "Calibration"
    def parse(self):
        self.description = self.fullDescription

class SnoreEvent(Event):
    eType = "Snore"
    def parse(self):
        if self.fullDescription.split()[1] == "Episode":
            self.duration = 1 # Arbitrary
            self.description = "Episode"
        else:
            self.duration = float(self.fullDescription.split()[3])
            self.description = "{:3.1f}".format(self.duration)

class DesaturationEvent(Event):
    eType = "Desaturation"
    def parse(self):
        desc = map(lambda x:x.strip(), self.fullDescription.split('-'))
        try:
            self.duration = float(desc[1].split()[1])
            self.min = float(desc[2].split()[1])
            self.drop = float(desc[3].split()[1])

            self.description = "Dur: {:3.1f}, Min: {:3.1f}, Drop: {:3.1f}".format(self.duration, self.min, self.drop)
        except IndexError:
            self.description = "{} (Data Format Error)".format(desc[-1])
            
class PLMEvent(Event):
    eType = "PLM"
    def parse(self):
        desc = self.fullDescription.split()
        try:
            self.duration = float(desc[3])
            self.description = "{:2.1f} Type:{}".format(self.duration, desc[-1])
        except IndexError:
            self.description = "{} (Data Format Error)".format(desc[-1])
##External Events:

class BilevelEvent(Event):
    eType = "Bilevel"
    def parse(self):
        desc = self.fullDescription.split()
        self.description = "IPAP: {} EPAP: {}".format(desc[5], desc[9])
        
class MontageEvent(Event):
    eType = "Montage"
    def parse(self):
        self.description = "{}".format(self.fullDescription.split()[3])
    
class SensitivityEvent(Event):
    eType = "Sensitivity"
    def parse(self):
        try:
            desc = self.fullDescription.split()
            self.description = "Value: {} Loc: {}".format(desc[1], desc[3])
        except IndexError:
            desc = self.fullDescription
            self.description = "Location(s): {}".format(desc.split()[desc.find('Sensitivity')+12:])
        
class FilterEvent(Event):
    eType = "Filter"
    def parse(self):
        desc = self.fullDescription.split()
        try:
            self.description = "Freq: {} Type: {}".format(desc[2],desc[4])
        except IndexError:
            self.description = "Freq: {} Type: {}".format(desc[2],desc[3])
