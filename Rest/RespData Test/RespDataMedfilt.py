import numpy as np
import csv
import cStringIO as StringIO
from LogReader import LogReader
import pylab as plt

class RespData(object):
    def __init__(self, data, verbose=True, isString=False, plotAll = False):
        self.verbose = verbose
        # Constant declarations
        self.epochLength = 30
        self.sampleRate = 5
        self.windowLength = self.epochLength * self.sampleRate
        
        # Create filter chain
        self.filterChain = [self.zeroWithLowPeaks,
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

        # if plotAll is True, plots raw data vs. zeroed data vs. normalized data.
        if plotAll:
            i = 1
            numPlots = len(self.filterChain)
            plt.figure(1)
            sp1 = plt.subplot(numPlots,1,i)
            sp1.plot(self.data[:,0])
            plt.figure(2)
            sp2 = plt.subplot(numPlots,1,i)
            sp2.plot(self.data[:,1])
        for filt in self.filterChain:
            self.data = filt(self.data)
            if plotAll and (not filt == self.roundData):
                i+=1
                plt.figure(1)
                sp = plt.subplot(numPlots,1,i,sharex=sp1)
                sp.plot(self.data[:,0])
                sp.plot([0 for x in self.data[:,0]])
                plt.figure(2)
                sp = plt.subplot(numPlots,1,i,sharex=sp2)
                sp.plot(self.data[:,1])
                sp.plot([0 for x in self.data[:,1]])
                
            # stores zeroed Data
            if filt == self.zeroWithLowPeaks:
                self.zeroData = self.data.copy()
        if plotAll:
            plt.show()
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

    # simple median filter. 
    def medfilt(self,data,width):
        w = width
        newData = np.zeros(len(data))
        for i in range(len(data)):
            if i-w/2<0:
                newData[i] = np.median(np.hstack((np.zeros(w/2-i),data[0:i+w/2+1])))
            elif i+w/2+1>len(data):
                newData[i] = np.median(np.hstack((data[i-w/2:len(data)],np.zeros(i+w/2+1-len(data)))))
            else:
                newData[i] = np.median(data[i-w/2:i+w/2+1])
        return newData

    # Returns lower peaks, upper peaks, average line.
    def peaks(self, data = None):
        if data == None:
            data = self.getData()
            
        lowerPeakLine = np.zeros(data.shape)
        upperPeakLine = np.zeros(data.shape)
        avgs = np.zeros(data.shape)
        # parameter. Window length for running mean used for zero crossings. Window is about a breath long. 
        runningMeanWidth = 25
        lowerPeaks =[]
        upperPeaks = []
        lowerPeakTimes = []
        upperPeakTimes = []
        d = data.copy()
        dead = np.zeros(d.shape)        # Array to indicate where signal stays constant
        currentBreath = []              # Values of the current breath (one period of signal)
        currentBreathi = []             # Indices corresponding to above
        previous = 0                    # previous data point
        stable = 0                      # count of how long a signal is constant
        lastLowerPeak = None            # (index of the last peak, value of last peak)
        lastUpperPeak = None
        # intialize positive, a boolean which keeps track of whether signal is above or below the average. 
        if d[0]>0:
            positive = True
        else:
            positive = False

        for j in range(len(d)):
            # find mean line
            avg = np.mean(d[max(0,j-runningMeanWidth/2):min(len(d),j+runningMeanWidth/2)])
            avgs[j] = avg

            # signal above mean line. Add values to current breath. 
            if d[j] > avg:
                positive = True
                currentBreath += [d[j]]
                currentBreathi += [j]

            # signal below mean line
            else:
                # full period completed. Process peaks.     
                if positive and currentBreath:
                    m = np.argmin(currentBreath-avg)        # index of the new peak. currentBreath[m] = currentPeak
                    M = np.argmax(currentBreath-avg)
                    
                    lowerPeakTimes += [currentBreathi[m]]
                    upperPeakTimes += [currentBreathi[M]]
                    
                    # detect first peak. peak line set constant at first peak, until first peak.
                    if lastLowerPeak == None:
                        for k in range(m):
                            lowerPeakLine[k] = currentBreath[m]
                        lastLowerPeak = (currentBreathi[m], currentBreath[m])
                    if lastUpperPeak == None:
                        for k in range(M):
                            upperPeakLine[k] = currentBreath[M]
                        lastUpperPeak = (currentBreathi[M], currentBreath[M])
                    else:
                        # line fit from the last peak (lastLowerPeak=(a,b)) to the current peak (m,currentBreath[m])
                        x1 = lastLowerPeak[0]
                        y1 = lastLowerPeak[1]
                        x2 = currentBreathi[m]
                        y2 = currentBreath[m]
                        if x2-x1:
                            for k in range(x1,x2):
                                # only store peaks if the signal is not dead
                                if not dead[k]:
                                    lowerPeakLine[ k ] = float(y2-y1)/float(x2-x1)*(k-x1)+y1
                            lastLowerPeak = (currentBreathi[m], currentBreath[m])
                        else:
                            # very rare case where last peak is the same as new peak (x2-x1 = 0)
                            lowerPeakLine[currentBreathi[m]] = lastLowerPeak[1]
                            lastLowerPeak = (currentBreathi[m], currentBreath[m])
                            
                        # line fit from lastUpperPeak=(a,b) to (M,currentBreath[M])
                        x1 = lastUpperPeak[0]
                        y1 = lastUpperPeak[1]
                        x2 = currentBreathi[M]
                        y2 = currentBreath[M]
                        if x2-x1:
                            for k in range(x1,x2):
                                # only store peaks if the signal is not dead
                                if not dead[k]:
                                    upperPeakLine[ k ] = float(y2-y1)/float(x2-x1)*(k-x1)+y1
                            lastUpperPeak = (currentBreathi[M], currentBreath[M])
                        else:
                            # very rare case where last peak is the same as new peak (x2-x1 = 0)
                            upperPeakLine[currentBreathi[M]] = lastUpperPeak[1]
                            lastUpperPeak = (currentBreathi[M], currentBreath[M])
                    # Reset breath
                    currentBreath = []
                    currentBreathi = []
                    
                # add data to current breath
                currentBreath += [d[j]]
                currentBreathi += [j]
                positive = False

            # if constant, increment stable. 
            if abs(d[j] - previous) < 5:
                stable += 1
            # if changing, reset stable. 
            else:
                stable = 0
            # if constant for long enough, process peaks. When constant, every point is counted as a peak. 
            if stable>5:
                
                # connect previous peak to new "peak"
                x1 = lastLowerPeak[0]
                y1 = lastLowerPeak[1]
                x2 = j
                y2 = d[j]
                if x2-x1:
                    for k in range(x1,x2):
                        if not dead[k]:
                            lowerPeakLine[ k ] = float(y2-y1)/float(x2-x1)*(k-x1)+y1
                x1 = lastUpperPeak[0]
                y1 = lastUpperPeak[1]
                if x2-x1:
                    for k in range(x1,x2):
                        if not dead[k]:
                            upperPeakLine[ k ] = float(y2-y1)/float(x2-x1)*(k-x1)+y1
                
                lowerPeakTimes += [j]
                upperPeakTimes += [j]
                
                # points where stable was incrementing is also dead. 
                for k in range(j-4,j+1):
                    dead[j] = 1
                    lowerPeakLine[j] = d[j]
                    upperPeakLine[j] = d[j]
                lastLowerPeak = (j,d[j])
                lastUpperPeak = (j,d[j])
                currentBreath = []
                currentBreathi = []
                
            previous = d[j]
        # final peak to the end is a flat line
        for j in range(lastLowerPeak[0],len(d)):
            lowerPeakLine[j] = lastLowerPeak[1]
        for j in range(lastUpperPeak[0],len(d)):
            upperPeakLine[j] = lastUpperPeak[1]
        
        upperPeaks = data[upperPeakTimes]
        lowerPeaks = data[lowerPeakTimes]
        upperPeakTimes = np.array(upperPeakTimes)
        lowerPeakTimes = np.array(lowerPeakTimes)
        
        return {'avgs':avgs,'upperPeaks': upperPeaks, 'upperPeakTimes': upperPeakTimes, 'lowerPeaks': lowerPeaks,
                'lowerPeakTimes': lowerPeakTimes,'upperPeakLine':upperPeakLine, 'lowerPeakLine':lowerPeakLine}

    def TiTe(self, peaks = None):
        if peaks == None:
            uPeaks = self.getFeature('upperPeaks')
            lPeaks = self.getFeature('lowerPeaks')
            uPeakTimes = self.getFeature('upperPeakTimes')
            lPeakTimes = self.getFeature('lowerPeakTimes')
        else:
            uPeaks = peaks[0]
            uPeakTimes = peaks[1]
            lPeaks = peaks[2]
            lPeakTimes = peaks[3]
        
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
    
    # Zero-center data    
    def zeroWithLowPeaks(self, data):
        if self.verbose: print "Zeroing data"
        numCols = data.shape[1]
        newData = np.zeros(data.shape)
        for i in range(numCols):
            d = self.peaks(data=data[:,i])
            lowerPeakLine = d['lowerPeakLine']
            avgs = d['avgs']
            newData[:,i]=data[:,i]-lowerPeakLine
            plt.figure()
            plt.plot(data[:,i])
            plt.plot(lowerPeakLine)
            plt.plot(avgs)
            plt.plot(newData[:,i])
            plt.plot([0 for x in newData[:,i]])
        return newData
        
    # Normalize data using median filter
    def normalize(self, data):
        if self.verbose: print 'Normalizing data'
        # Parameters
        medFilterWidth = 151
        scale = 30
        numCols = data.shape[1]
        maxValue = 400

        newData = np.zeros(data.shape)
        for i in range(numCols):
            d = self.peaks(data=data[:,i])
            upperPeakLine = d['upperPeakLine']
            avgs = d['avgs']
            plt.figure()
            plt.plot(data[:,i])
            plt.plot(upperPeakLine)
            plt.plot(avgs)
                
            # Decimate peak list
            dec = 10
            decUpperPeaks = upperPeakLine[np.arange(1,len(upperPeakLine), dec)]
            
            # Filter shortened peak list
            if self.verbose: print " - {}: Median filter".format(i)
            decUpperPeaks = self.medfilt(decUpperPeaks, medFilterWidth)
            
            
            # Un-decimate peak list
            for j in range(len(decUpperPeaks)):
                if not j:
                    for k in range(1):
                        upperPeakLine[k] = decUpperPeaks[j]
                else:
                    for k in range(dec):
                        upperPeakLine[dec*(j-1)+k+1] = float(decUpperPeaks[j] - decUpperPeaks[j-1] )/dec * k + decUpperPeaks[j-1]
            for k in range(len(decUpperPeaks)*dec+1, len(upperPeakLine)):
                upperPeakLine[k] = decUpperPeaks[len(decUpperPeaks)-1]
                
            plt.plot(upperPeakLine)
                
            # Normalize using width of envelope
            width = (upperPeakLine) / 2.0
            width[width == 0] = np.inf
            data[:,i] = (data[:,i] / width) * scale

            # Limit peaks
            data[:,i][data[:,i] > maxValue] = maxValue
            data[:,i][data[:,i] < -maxValue] = -maxValue

            newData[:,i] = data[:,i]
            
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
    

##    # Find peaks between zero crossings
##    def peaks(self):
##        # Need zero crossings
##        crossings = self.getFeature('zeroCrossings')
##
##        data = self.getData()
##
##        # Storage
##        uPeaks = []
##        uPeakTimes = []
##        lPeaks = []
##        lPeakTimes = []
##
##        for i in range(0, len(crossings)-2, 2):
##            bump = data[crossings[i]:crossings[i+2]]
##            uPeakTimes.append(crossings[i] + np.argmax(bump))
##            lPeakTimes.append(crossings[i] + np.argmin(bump))
##
##        uPeaks = data[uPeakTimes]
##        lPeaks = data[lPeakTimes]
##        uPeakTimes = np.array(uPeakTimes)
##        lPeakTimes = np.array(lPeakTimes)
##
##        return {'upperPeaks': uPeaks, 'upperPeakTimes': uPeakTimes, 'lowerPeaks': lPeaks, 'lowerPeakTimes': lPeakTimes}
        

    # Find locations of zero crossings
    def zeroCrossings(self):
        data = self.getData()
            
        dataRoll = np.roll(data, 1)
        
        # Effectively: if the next point's sign is different than the curren point, flag as crossing
        crossings = np.nonzero( (data>0) != (dataRoll>0) )[0]
        
        return {'zeroCrossings': crossings}


##    def TiTe(self):
##        # Requires peaks
##        uPeaks = self.getFeature('upperPeaks')
##        lPeaks = self.getFeature('lowerPeaks')
##        uPeakTimes = self.getFeature('upperPeakTimes')
##        lPeakTimes = self.getFeature('lowerPeakTimes')
##
##        # Make arrays same length
##        numBreaths = min(len(uPeaks), len(lPeaks))
##        uPeaks = uPeaks[:numBreaths]
##        lPeaks = lPeaks[:numBreaths]
##        uPeakTimes = uPeakTimes[:numBreaths]
##        lPeakTimes = lPeakTimes[:numBreaths]
##        
##        # Find Ti/Te values
##        if uPeakTimes[0] > lPeakTimes[0]:
##            # Inhale happened first
##            tiValues = uPeakTimes - lPeakTimes
##            teValues = (np.roll(lPeakTimes, -1) - uPeakTimes)[:-1]
##
##            hiValues = uPeaks - lPeaks
##            heValues = (np.roll(lPeaks, -1) - uPeaks)[:-1]
##        else:
##            # Exhale happened first
##            teValues = lPeakTimes - uPeakTimes
##            tiValues = (np.roll(uPeakTimes, -1) - lPeakTimes)[:-1]
##
##            heValues = lPeaks - uPeaks
##            hiValues = (np.roll(uPeaks, -1) - lPeaks)[:-1]
##
##
##        # Fill in long array with most recent value
##        ti = np.zeros(len(self.data))
##        te = np.zeros(len(self.data))
##        hi = np.zeros(len(self.data))
##        he = np.zeros(len(self.data))
##
##        for i in range(len(uPeakTimes)-1):
##            ti[uPeakTimes[i]:uPeakTimes[i+1]] = tiValues[i]
##            hi[uPeakTimes[i]:uPeakTimes[i+1]] = hiValues[i]
##        ti[uPeakTimes[-1]:] = tiValues[-1]
##        hi[uPeakTimes[-1]:] = hiValues[-1]
##
##        for i in range(len(lPeakTimes)-1):
##            te[lPeakTimes[i]:lPeakTimes[i+1]] = teValues[i]
##            he[lPeakTimes[i]:lPeakTimes[i+1]] = heValues[i]
##        te[lPeakTimes[-1]:] = teValues[-1]
##        he[lPeakTimes[-1]:] = heValues[-1]
##
##        return {'Ti': ti, 'Te': te, 'Hi': hi, 'He': he}


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

    datadir = ''
##    files = ['B23 Night 1', 'ALG_ (26)','DulcieTankTest']
    files = ['B23 Night 1']
    
##    for fname in files:
##        a = RespData(os.path.join(datadir, "{}.resp".format(fname)), verbose=False, plotAll = True)
        
    # plot rawData vs normalized data
    for fname in files:
        a = RespData(os.path.join(datadir, "{}.resp".format(fname)), verbose=True)
    plt.show()
##        plt.figure()
##        ax1 = plt.subplot(211)
####        ax1.plot(a.rawData[:,0])
##        ax1.plot(a.rawData[:,1])
##        ax2 = plt.subplot(212, sharex=ax1)
##        ax2.plot(a.data[:,1])
##        ax2.plot([0 for x in a.data[:,1]])
####        ax2.plot([0 for x in a.data[:,0]])
##    plt.show()
    
    #a.loadStaging("Data\\{}_staging.txt".format(fname))
##    a.loadLog(os.path.join(datadir, "{}_log.parsed.txt".format(fname)))
##
##    a.getEvent('Snore')
