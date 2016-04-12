import sys
import numpy as np
import pandas as pd

class Datasets:

    """ List of Class variables"""
    __author__ = 'Jose A Alvarado'

    """ List of class methods """
   
    #Reading the data file """
    # Read, parse and save in a list named colHeaders the first line of the file containing the name of the columns of the dataset if fileHeader is set to true in the class constructor.
    # Starting at the second line (if fileHeader is set to true) to the end of the data file it read, parse and append the line to the features list except the index of the column 
    # specified in the labelIndex variable of the class constructor which represent the labels so that column is save in the labels list and remove from the row list before is appended to the features list.

    def readDataFile(self,labelIndex):
        if labelIndex == -1:
            self.fileLabelIndex = labelIndex
        else:
            self.fileLabelIndex = labelIndex - 1
        try:
            data = open(self.dataFile,'r')
        except IOError as e:
            raise
            
        firstline = self.fileHeader
        for line in data:
            if firstline:
                self.colHeaders = line.strip().split(self.fileDelimiter)
                self.colHeaders.pop(self.fileLabelIndex)
                firstline = False
            else:
                row = line.strip().split(self.fileDelimiter)
                self.labels.append(int(row[-1]))
                row.pop(self.fileLabelIndex)
                self.features.append(row)
        data.close()
        self.nrows = len(self.features) # number of rows in datafile
        self.ncols = len(self.features[0]) # number of columns in the datafile

    
    # Given a categorical vector it creates a dictionary where the keys are the vector unique values
    # and the values of the dictionary is a list containing the frequency(integer) and 
    # relative frequency (string exemple: 10.4%)
    # parameters:
        #self: to get access to instance variables,
        #x: categorical vector to compute the frequencies
    def getFrequency(self,x):
        result={}
        for value in x:
            if value in result.keys():
                result[value][0]+=1
            else:
                result[value]=[1]
        for key in result.keys():
            result[key].append('{:.1f}%'.format((float(result[key][0])/len(x))*100))
            result[key][0]=result[key][0]
        return result

        # Determine which colums are categorical and which are numeric
        # parameters:
            # numIndex: List containing the indeces of the columns of the data that are numeric
    def setFeatureType(self,numIndex):
        for col in range(self.ncols):
            if len(numIndex)==0 or col in numIndex:
                for row in range(self.nrows):
                    try:
                        self.features[row][col] = float(self.features[row][col])
                        if row+1==self.nrows:
                            self.numericFeatures.append(col)
                    except:
                        self.categoricalFeatures.append(col)
                        break
            else:
                self.categoricalFeatures.append(col)

    # Method to create binary columns for each value of the categorical features (factorize)
    # parameters: 
        # self: to access the instance variables of the class
    def stringsToFactors(self):
        categoricalFeaturesFactorizeNames = [] # list to save the variable names of the new features
        categoricalFeaturesFactorize = []
        numericFeatures = []
        # For each categorical feature gets the frequency of each value
        for col in self.categoricalFeatures:
            self.categoricalFeaturesDistributions[col]=self.getFrequency([self.features[row][col] for row in range(self.nrows)])
            for catCol in self.categoricalFeaturesDistributions[col].keys():
                categoricalFeaturesFactorizeNames.append(self.colHeaders[col] + '_' + catCol)
        dim = sum([len(self.categoricalFeaturesDistributions[key].keys()) for key in self.categoricalFeatures])
        
        for row in range(self.nrows):
            catFeatureRow = [0] * dim
            for key in self.categoricalFeatures:
                value = self.colHeaders[key] + '_' + self.features[row][key]
                valueIndex = next(i for i in range(len(categoricalFeaturesFactorizeNames)) if categoricalFeaturesFactorizeNames[i]==value)
                catFeatureRow[valueIndex] = 1
            categoricalFeaturesFactorize.append(catFeatureRow)
            numericFeatures.append([self.features[row][numCol] for numCol in self.numericFeatures])
        self.featuresFactorized = np.hstack([np.array(numericFeatures),np.array(categoricalFeaturesFactorize)])
        self.featuresFactorizedHeader = [self.colHeaders[name] for name in self.numericFeatures] + categoricalFeaturesFactorizeNames
    
    # Method to normalize the features.
    # It creates two Nunpy 2D array one with only the numeric features normailzed and the rest stay the same
    # and the other 2D array have all features normalized
    def normalizeFeatures(self):
        normalizedFeatures = self.featuresFactorized.copy()
        for col in range(self.featuresFactorized.shape[1]):
            normalizedFeatures[:,col] = (normalizedFeatures[:,col] - normalizedFeatures[:,col].mean())/normalizedFeatures[:,col].std()
            if col == len(self.numericFeatures) - 1:
                print 'setting normalizeNumeric'
                self.featuresNormalizeNumeric = normalizedFeatures.copy()
        self.featuresNormalizeAll = normalizedFeatures
        
    """ Class constructor"""
    # Use to instantiate the class instance
    # parameters:
        # self: same as above
        # dataFile: physical location of the file to read
        # sep: file data delimiter
        # header: does the file contains the column header as the first row?
        # label_index: what is the column index where the labels are located in the file. If not specified it uses the last column on the file
        # numericIndeces: a list specifying the indeces of the numeric columns columns. If not specified it takes all the columns that can be 
            #  converted to float without errors as numeric. The indices in this list are excluded from converting the categorical indeces to factors.
    def __init__(self,dataFile,sep=',',header=True,labelIndex=-1,numericIndeces=[]):
        """ Instance variables """
        self.dataFile = dataFile # contains the file location as passed to the class constructor
        self.fileDelimiter = sep # contains the file delimiter passed to the class constructor
        self.fileHeader = header # Boolean indicating whether the file contains a header
        self.colHeaders = [] # list of features names
        self.features = [] # Contains the data file loaded to a list of rows of a list of columns
        self.featuresFactorized = None # Numpy 2D array containing the numeric and factorized features of the original data
        self.featuresFactorizedHeader = None # List containing the features names of the featuresFactorized matrix
        self.labels = [] # list of labels
        self.numericFeatures = [] # list of numerical features names
        self.featuresNormalizeNumeric = None # Numpy 2D array containing the data with only the numeric features normalized
        self.featuresNormalizeAll = None # Numpy 2D array containing the data with all features normalized
        self.featuresDataFrame = None # Pandas data frame containing the original data of the features withut any transformation
        self.categoricalFeatures = [] # dictionary of frequencies for the cartegorical features
        self.categoricalFeaturesDistributions = {}
        self.ncols = None
        self.nrows = None
        self.fileLabelIndex = None  #Index of the file column containing the labels

        try:
            self.readDataFile(labelIndex)
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            return None
        self.setFeatureType(numericIndeces)
        self.stringsToFactors()
        self.normalizeFeatures()
        self.featuresDataFrame = pd.DataFrame(self.features,columns=self.colHeaders)