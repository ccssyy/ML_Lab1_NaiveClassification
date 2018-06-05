import csv
import random
import math
import numpy as np

#filename = 'wine.csv'

def loadCsv(filename):
    lines = csv.reader(open(filename))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[0] not in separated):
            separated[vector[0]] = []
        separated[vector[0]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attibute), stdev(attibute)) for attibute in zip(*dataset)]
    del summaries[0]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def calculatePreProbability(dataset):
    classProbability = {}
    for classValue, instances in separateByClass(dataset).items():
        #print(len(instances),len(dataset))
        classProbability[classValue] = float(len(instances)/len(dataset))
    #print(classProbability)
    return classProbability

def calculateClassProbabilities(classProbability, summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i + 1]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
        probabilities[classValue] *= classProbability[classValue]
    return probabilities


def predict(classProbability, summaries, inputVector):
    probabilities = calculateClassProbabilities(classProbability, summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(classProbability, summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(classProbability, summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][0] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

if __name__ == '__main__':
    filename = 'wine.csv'
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset,splitRatio)
    print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset),len(trainingSet),len(testSet)))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    classProbability = calculatePreProbability(trainingSet)
    # test model
    #test = np.array(testSet)[:,0]
    predictions = getPredictions(classProbability, summaries,testSet)
    #print(test)
    #print(np.array(predictions))
    accuracy = getAccuracy(testSet,predictions)
    print('Accuracy: {0}%'.format(accuracy))