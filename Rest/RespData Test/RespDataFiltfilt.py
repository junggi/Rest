import numpy as np
import scipy.signal as sig
import csv
import cStringIO as StringIO

from LogReader import LogReader

class RespData(object):
    def __init__(self, data, verbose=True, isString=False):
        self.verbose = verbose
        
        # Constant declarations
        self.epochLength = 30
        self.sampleRate = 5
        self.windowLength = self.epochLength * self.sampleRate
        
        # Create filter chain
        self.filterChain = [self.zeroWithFiltFilt,
                            self.normalize,
                            self.roundData]

        # Create storage for features
        self.features = {}
        self.featureTable = {'zeroCrossings': self.zeroCrossings,

                             'upperPeaks': self.peaks,
                             'lowerPeaks': self.peaks,
                             'upperPeakTimes': self.peaks,
                             'lowerPeakTimes': self.peaks,

                             'Ti': self.TiTe,
                             'Te': self.TiTe,
                             'Hi': self.TiTe,
                             'He': self.TiTe,
                             }

        self.stages = None
        self.log = None

        if isString:
            # Create IO object
            fh = StringIO.StringIO(data)
            self.rawData, self.descriptors = self.readFile(fh)
        else:
            # Read file
            with open(data, 'r') as fh:
                self.rawData, self.descriptors = self.readFile(fh)

        # Filter data
        self.data = self.rawData.copy()
        for filt in self.filterChain:
            self.data = filt(self.data)

        # Combine channels into a single channel
        self.singleChannel = self.getSingleChannel()[0]

        
    """#########################################################################
    # Getters
    #########################################################################"""
    
    def getData(self, channel=None, raw=False):
        if channel is not None:
            if raw:
                return self.rawData[:,channel]
            else:
                return self.data[:,channel]
        else:
            return self.singleChannel


    """#########################################################################
    # File Read/Write
    #########################################################################"""
    
    # Read in a .resp file and return data and descriptors
    def readFile(self, fh):
        if self.verbose: print "Reading file"
        data = []
        descriptors = []

        
        # Check version
        if fh.readline().strip() != "RESP100":
            raise Exception("Incompatible filetype")
        
        # Get descriptors
        descriptors.append(fh.readline().strip().strip('#'))
        descriptors.append(fh.readline().strip().strip('#'))

        # Read each line
        for line in fh:
            # Remove whitespace
            line = line.strip()
            
            # Ignore comments and empty lines
            if line == "" or line[0] == '#':
                continue
            
            # Split columns, convert to integer, and add to data
            line = map(int, line.split(','))
            data.append(line)

        # Return data and descriptors
        return (np.array(data), descriptors)


    # Load staging file
    def loadStaging(self, filename):
        if self.verbose: print "Loading stages"
        stages = []

        # Open file and read in stages
        with open(filename, 'r')as fh:
            for line in fh:
                stages.append(int(line.strip()[-1]))

        # Check length of data is compatible with stages
        if len(stages) > self.data.shape[0] / self.windowLength:
            raise Exception('Too many stages for data file')

        # Pad stages to length of data
        padding = (self.data.shape[0] / self.windowLength) - len(stages)
        if padding > 0:
            if self.verbose: print "Padding stage list by {}".format(padding)
            stages += [0] * padding

        self.stages = np.array(stages)


    # Write file to ARFF format with correct features
    def writeARFF(self, filename, featureList=None, channel=0):
        # If no feature list specified, use all features
        if featureList == None:
            featureList = ['Ti', 'Te', 'Hi', 'He', 'varVt']

        # Check that staging is already loaded
        if self.stages is None:
            raise Exception("Staging must be loaded first")

        # Get all features from feature list
        features = []
        for feature in featureList:
            features.append(self.getFeature(channel, feature))

        # Build large matrix of all features
        data = np.vstack(features + [self.stages]).transpose()

        # Filter data to remove unstaged sections
        data = np.delete(data, np.nonzero(data[:,-1] == 0), 0)

        # Convert to list for wake/sleep
        c = {1:'s', 2:'s', 3:'s', 4:'s', 5:'w'}
        data = data.tolist()
        for i in range(len(data)):
            data[i][-1] = c[data[i][-1]]

        # Write ARFF file
        with open(filename, 'w') as fh:
            fh.write("@RELATION {}\n\n".format('name')) ## TODO: FIGURE OUT WHAT NAME DOES
            for feature in featureList:
                fh.write("@ATTRIBUTE {} NUMERIC\n".format(feature))
            fh.write("@ATTRIBUTE stage {s,w}\n")

            fh.write("@DATA\n")

            writer = csv.writer(fh)
            writer.writerows(data)
            
            
    """#########################################################################
    # Stage Extraction
    #########################################################################"""

    def getStageData(self, stageList, data=None):
        # Check that stages are already loaded
        if self.stages is None:
            raise Exception("Stages must be loaded first.")

        # If data is not specified, use self.data
        if data is None:
            data = self.data

        # Check that data is the correct length
        if data.shape[0] != self.data.shape[0]:
            raise Exception("Data is not comparable to loaded data, check size")

        # Create array that indicates which points should be included
        validPoints = np.zeros(data.shape[0], dtype="bool")

        # For each stage, if it is to be included, set validPoints
        for i in range(self.stages.shape[0]):
            if self.stages[i] in stageList:
                validPoints[i*self.windowLength:(i+1)*self.windowLength] = 1

        return data[validPoints]
    

    """#########################################################################
    # Filters
    #########################################################################"""

    # Zero-center data
    def zeroWithFiltFilt(self, data):
        if self.verbose: print "Zeroing data"
        N = 3
        Wn = 0.05
        b, a = sig.butter(N, Wn)
        numCols = data.shape[1]
        newData = np.zeros(data.shape)
        for i in range(numCols):
            newData[:,i] = data[:,i] - sig.filtfilt(b, a, data[:,i])
        return newData


    # Normalize data using median filter
    def normalize(self, data):
        if self.verbose: print 'Normalizing data'
        # Parameters
        filterWidth = 151
        scale = 30
##        nearZero = np.std(data) * 0.05
        nearZero = 10
        numCols = data.shape[1]
        maxValue = 400

        newData = np.zeros(data.shape)
        for i in range(numCols):
            d = data[:,i].copy()
            
            # Find envelope
            if self.verbose: print " - {}: Finding envelope".format(i)
            upperPeaks = np.zeros(d.shape)
            lowerPeaks = np.zeros(d.shape)
            currentUpper = 0.0
            currentLower = 0.0
            for j in range(1, len(d)-1):
                if (d[j-1] < d[j] > d[j+1]) and d[j] > 0:
                    currentUpper = d[j]
                if (d[j-1] > d[j] < d[j+1]) and d[j] < 0:
                    currentLower = d[j]

                upperPeaks[j] = currentUpper
                lowerPeaks[j] = currentLower

            # Decimate peak list
            dec = 10
            decUpperPeaks = upperPeaks[np.arange(1,len(upperPeaks), dec)]
            decLowerPeaks = lowerPeaks[np.arange(1,len(lowerPeaks), dec)]
        
            # Filter shortened peak list
            if self.verbose: print " - {}: Median filter".format(i)
            decUpperPeaks = sig.medfilt(decUpperPeaks, filterWidth)
            decLowerPeaks = sig.medfilt(decLowerPeaks, filterWidth)

            # Un-decimate peak list
            for x in range(len(upperPeaks)):
                upperPeaks[x] = decUpperPeaks[x/dec]
                lowerPeaks[x] = decLowerPeaks[x/dec]

            # Normalize using width of envelope
            width = (upperPeaks - lowerPeaks) / 2.0
            width[width < nearZero] = np.inf
            d = (d / width) * scale

            # Limit peask
            d[d > maxValue] = maxValue
            d[d < -maxValue] = -maxValue

            newData[:,i] = d
            
        return newData
            

    # Round data array to integer
    def roundData(self, data):
        if self.verbose: print 'Rounding Data'
        newData = np.zeros(data.shape, int)
        data.round(0, newData)
        return newData

    """#########################################################################
    # Feature Extraction
    #########################################################################"""
    
    # Get requested feature from table, calculate if not available
    def getFeature(self, featureName):
        # Check if feature needs to be calculated
        if featureName not in self.features:
            
            if featureName in self.featureTable:
                data = self.featureTable[featureName]()
                self.features.update(data)

        # Return feature
        return self.features[featureName]
    

    # Find peaks between zero crossings
    def peaks(self):
        # Need zero crossings
        crossings = self.getFeature('zeroCrossings')

        data = self.getData()

        # Storage
        uPeaks = []
        uPeakTimes = []
        lPeaks = []
        lPeakTimes = []

        for i in range(0, len(crossings)-2, 2):
            bump = data[crossings[i]:crossings[i+2]]
            uPeakTimes.append(crossings[i] + np.argmax(bump))
            lPeakTimes.append(crossings[i] + np.argmin(bump))

        uPeaks = data[uPeakTimes]
        lPeaks = data[lPeakTimes]
        uPeakTimes = np.array(uPeakTimes)
        lPeakTimes = np.array(lPeakTimes)

        return {'upperPeaks': uPeaks, 'upperPeakTimes': uPeakTimes, 'lowerPeaks': lPeaks, 'lowerPeakTimes': lPeakTimes}
        

    # Find locations of zero crossings
    def zeroCrossings(self):
        data = self.getData()
            
        dataRoll = np.roll(data, 1)
        
        # Effectively: if the next point's sign is different than the curren point, flag as crossing
        crossings = np.nonzero( (data>0) != (dataRoll>0) )[0]
        
        return {'zeroCrossings': crossings}


    def TiTe(self):
        # Requires peaks
        uPeaks = self.getFeature('upperPeaks')
        lPeaks = self.getFeature('lowerPeaks')
        uPeakTimes = self.getFeature('upperPeakTimes')
        lPeakTimes = self.getFeature('lowerPeakTimes')

        # Make arrays same length
        numBreaths = min(len(uPeaks), len(lPeaks))
        uPeaks = uPeaks[:numBreaths]
        lPeaks = lPeaks[:numBreaths]
        uPeakTimes = uPeakTimes[:numBreaths]
        lPeakTimes = lPeakTimes[:numBreaths]
        
        # Find Ti/Te values
        if uPeakTimes[0] > lPeakTimes[0]:
            # Inhale happened first
            tiValues = uPeakTimes - lPeakTimes
            teValues = (np.roll(lPeakTimes, -1) - uPeakTimes)[:-1]

            hiValues = uPeaks - lPeaks
            heValues = (np.roll(lPeaks, -1) - uPeaks)[:-1]
        else:
            # Exhale happened first
            teValues = lPeakTimes - uPeakTimes
            tiValues = (np.roll(uPeakTimes, -1) - lPeakTimes)[:-1]

            heValues = lPeaks - uPeaks
            hiValues = (np.roll(uPeaks, -1) - lPeaks)[:-1]


        # Fill in long array with most recent value
        ti = np.zeros(len(self.data))
        te = np.zeros(len(self.data))
        hi = np.zeros(len(self.data))
        he = np.zeros(len(self.data))

        for i in range(len(uPeakTimes)-1):
            ti[uPeakTimes[i]:uPeakTimes[i+1]] = tiValues[i]
            hi[uPeakTimes[i]:uPeakTimes[i+1]] = hiValues[i]
        ti[uPeakTimes[-1]:] = tiValues[-1]
        hi[uPeakTimes[-1]:] = hiValues[-1]

        for i in range(len(lPeakTimes)-1):
            te[lPeakTimes[i]:lPeakTimes[i+1]] = teValues[i]
            he[lPeakTimes[i]:lPeakTimes[i+1]] = heValues[i]
        te[lPeakTimes[-1]:] = teValues[-1]
        he[lPeakTimes[-1]:] = heValues[-1]

        return {'Ti': ti, 'Te': te, 'Hi': hi, 'He': he}


    """#########################################################################
    # Channel Selection
    #########################################################################"""

    def getSingleChannel(self):
        # Parameters
        maxValue = 2**24
        extremeMargin = 0.10
        extremeTop = maxValue * (1-extremeMargin)
        extremeBottom = maxValue * extremeMargin

        # Get datasets in windows
        ch1 = self.rawData[:, 0]
        ch2 = self.rawData[:, 1]
        remainder = self.windowLength - (ch1.shape[0] % self.windowLength)
        if remainder != 0:
            ch1 = np.hstack([ch1, np.zeros(remainder)])
            ch2 = np.hstack([ch2, np.zeros(remainder)])
        ch1 = ch1.reshape((-1, self.windowLength))
        ch2 = ch2.reshape((-1, self.windowLength))

        # Build list of comparator functions that will be applied in order, lower value will be chosen
        functionList = [lambda i,ch: np.count_nonzero((ch[i]==maxValue) + (ch[i]==0)),
                        lambda i,ch: np.count_nonzero((ch[i]<extremeBottom)+(ch[i]>extremeTop)),
                        lambda i,ch: -1 * np.std(ch[i]),
                        ]

        # For each window, decide which channel to use
        output = np.zeros(ch1.shape[0])
        for i in range(ch1.shape[0]):
            for func in functionList:
                # Get score
                v1 = func(i,ch1)
                v2 = func(i,ch2)

                # See if we can make a decision on these values
                if v1 < v2:
                    output[i] = 0
                    break
                elif v2 < v1:
                    output[i] = 1
                    break
                else:
                    continue


        # Create final dataset
        data = np.zeros(self.data.shape[0])
        for i in range(output.shape[0]):
            r1 = i*self.windowLength
            r2 = min((i+1)*self.windowLength, self.data.shape[0])
            data[r1:r2] = self.data[r1:r2, output[i]]
        
        return data, output


    """#########################################################################
    # Sleep/Wake Based on Actigraphy
    #########################################################################"""

    def getWake(self, channel=None, moveWindow=25, sleepWindow=150, moveThresh=3.0, wakeThresh=0.40):
        # Get view of data
        data = self.getData(channel)

        # Create standard deviation
        stdChart = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            stdChart[i] = np.std(data[max(0, i-moveWindow/2) : min(data.shape[0], i+moveWindow/2)])

        # Create movement chart
        moveChart = stdChart > moveThresh * np.mean(stdChart)

        # Pad to fill each epoch
        remainder = moveChart.shape[0] % sleepWindow
        if remainder:
            moveChart = np.hstack([moveChart, [0]*(sleepWindow-remainder)])

        # For each epoch, if enough movement, mark wake
        wakeChart = moveChart.reshape(-1, sleepWindow)
        wakeChart = np.sum(wakeChart, 1) / float(sleepWindow)
        wakeChart = (wakeChart > wakeThresh) * 1

        return wakeChart, moveChart

    """#########################################################################
    # Utilities
    #########################################################################"""

    # Convert from a list of values per epoch to a list of time-values
    def fillEpochs(self, data):
        return np.ravel(np.tile(data, self.windowLength).reshape(-1, data.shape[0]).transpose())[:self.data.shape[0]]

    # Convert from time-series to a value per epoch array
    def getEpoch(self, data):
        return data.reshape(-1, self.windowLength)[:,0].copy()

    """#########################################################################
    # Sleep Lab Log Files
    #########################################################################"""

    def loadLog(self, filename):
        self.log = LogReader(filename)

    def getEvent(self, eventType, filt=None):
        if self.log is None:
            raise Exception('Error: Must load log first')
        
        return self.log.getTimeSeries(eventType, self.data.shape[0], filt=filt)

    def getEventTypes(self):
        if self.log is None:
            raise Exception('Error: Must load log first')
        
        return self.log.getEventTypes()

if __name__ == "__main__":
    import pylab as plt
    import os

    datadir = 'C:/Users/cdarling/Documents/Workspace/RespData/Data'
    fname = 'CD_Feb11'
    a = RespData(os.path.join(datadir, "{}.resp".format(fname)), verbose=True)
    #a.loadStaging("Data\\{}_staging.txt".format(fname))
    a.loadLog(os.path.join(datadir, "{}_log.parsed.txt".format(fname)))

    

    a.getEvent('Snore')
















    
