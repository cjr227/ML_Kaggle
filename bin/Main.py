import Datasets as ds
import sys

try:
    #fileLocation = sys.argv[1]
    fileLocation = "/Users/josealvarado/Documents/DataScience/Classes/MachineLearning/Project/data.csv"
    #fileLocation='C:\Users\jaa2220\Documents\DataScience\Machine_Learning\Project\data.csv'
    data = ds.Datasets(fileLocation,numericIndeces=[47,48])
except IndexError:
    print('Please enter the location of the file to load')