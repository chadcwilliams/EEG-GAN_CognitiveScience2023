#Modules
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import signal
from scipy.interpolate import interp1d
import scipy
import random as rnd
from IPython.display import clear_output
import time

#User Inputs
#addSyntheticData = 0 #Determine whether to add augmented EEG
avgOrTrialData = 'avg' #avg vs trial
crop = False
smallSearch = False
filteredGAN = True
features = False
validationOrTest = 'test'
runs = 5
dataEpochs = '8000'
dataSampleSizes = ['005','010','015','020','030','060','100']
syntheticDataOptions = [1]
proportionSynData = .5 #0.1 = 50 syn people, 0.2 = 100 syn people, 1.0 = 500 syn people
augFilename = 'augmentedPredictions_LR_SynP050_Runs8000_Filtered_TestClassification.csv'
empFilename = 'empiricalPredictions_LR_Runs8000_TestClassification.csv'
####
print('Average or Trial: ' + avgOrTrialData)
print('Crop: ' + str(crop))
print('Small Search: ' + str(smallSearch))
print('Runs: ' + str(runs))
print('Data Epochs: ' + dataEpochs)
print('data Sample Sizes: ' )
print(dataSampleSizes)
print('Synthetic Data: ' + str(syntheticDataOptions[0]))
print('Proportion Synthetic: ' + str(proportionSynData))
print('Classification Data: ' + validationOrTest)
print('Augmented Filename: ' + augFilename)
print('Empirical Filename: ' + empFilename) 


#Functions 

#Define Filter Function
def filterSyntheticEEG(EEG):
    #Bandpass
    w = [x / 100 for x in [0.1, 30]]
    b, a = signal.butter(4, w, 'band')
    
    #Notch
    b_notch, a_notch = signal.iirnotch(60, 30, 500)

    #Process
    tempFilteredEEG = [signal.filtfilt(b, a, EEG[trial,:]) for trial in range(len(EEG))]
    filteredEEG = [signal.filtfilt(b_notch, a_notch, tempFilteredEEG[trial]) for trial in range(len(EEG))]
    
    return filteredEEG

#Define Baseline Function
def baselineCorrect(EEG):
    #Baseline
    baselineRange = [0, 20]

    #process
    baselinedEEG = [(EEG[trial] - (np.mean(EEG[trial][baselineRange[0]:baselineRange[1]]))) for trial in range(len(EEG))]

    return baselinedEEG

#Define Resample Function
def resampleEEG(EEG):
    #Define function
    def interpolateArray(EEGTrial):
        interpModel = interp1d(np.arange(1,101), EEGTrial, fill_value='extrapolate')
        resampledEEG = interpModel(np.arange(1,101,100/600))
        return resampledEEG

    #Process
    resampleEEG = [interpolateArray(EEG[trial]) for trial in range(len(EEG))]

    return resampleEEG

def extractERP(EEG):
    #Time of interest (ms)
    startTime = 264
    endTime = 356
    
    #Convert to datapoints
    startPoint = round((startTime/12)+16)
    endPoint = round((endTime/12)+17) #Add an extra datapoint so last datapoint is inclusive
    
    #Process
    extractedERP = [np.mean(EEG[trial,startPoint:endPoint]) for trial in range(len(EEG))]
    #extractedERP = [EEG[trial,startPoint:endPoint] for trial in range(len(EEG))]

    return np.array(extractedERP)

def extractFFT(EEG):
    
    def runFFT(EEG):
        numberDataPoints = len(EEG) #Determine how many datapoints will be transformed per trial and channel
        SR = 83.3333
        frequencyResolution = SR/numberDataPoints #Determine frequency resolution
        fftFrequencies = np.arange(frequencyResolution,(SR/2),frequencyResolution) #Determine array of frequencies

        deltaRange = [frequencyResolution,3]
        thetaRange = [4,8]
    
        deltaIndex = [np.where(deltaRange[0]<=fftFrequencies)[0][0], np.where(deltaRange[1]<=fftFrequencies)[0][0]]
        thetaIndex = [np.where(thetaRange[0]<=fftFrequencies)[0][0], np.where(thetaRange[1]<=fftFrequencies)[0][0]]

        fftOutput = None #Empty variable
        fftOutput = scipy.fft.fft(EEG) #Compute the Fourier transform
        fftOutput = fftOutput/numberDataPoints #Normalize output
        fftOutput = np.abs(fftOutput) #Absolute transformation
        fftOutput = fftOutput[range(int(numberDataPoints/2))] #Extract the one-sided spectrum
        fftOutput = fftOutput*2 #Double values to account for lost values         
        fftOutput = fftOutput**2 #Convert to power
        fftOutput = fftOutput[1:] #Remove DC Offset
        extractedFFT = [np.mean(fftOutput[deltaIndex[0]:deltaIndex[1]]),np.mean(fftOutput[thetaIndex[0]:thetaIndex[1]])]
        
        return extractedFFT
    
    fftFeatures = [runFFT(EEG[trial,:]) for trial in range(len(EEG))]
    return np.array(fftFeatures)

def extractFeatures(EEG):
    erpFeatures = extractERP(EEG)
    fftFeatures = extractFFT(EEG)
    eegFeatures = np.transpose(np.vstack((erpFeatures,np.transpose(fftFeatures))))
    return eegFeatures

def averageSynthetic(synData):
    samplesToAverage = 50

    lossSynData = synData[synData[:,0]==0,:]
    winSynData = synData[synData[:,0]==1,:]

    lossTimeIndices = np.arange(0,lossSynData.shape[0],samplesToAverage)
    winTimeIndices = np.arange(0,winSynData.shape[0],samplesToAverage)
    
    if avgOrTrialData == 'avg':
        newLossSynData = [np.insert(np.mean(lossSynData[int(trialIndex):int(trialIndex)+samplesToAverage,1:],axis=0),0,0) for trialIndex in lossTimeIndices]
        newWinSynData = [np.insert(np.mean(winSynData[int(trialIndex):int(trialIndex)+samplesToAverage,1:],axis=0),0,1) for trialIndex in winTimeIndices]

    avgSynData = np.vstack((np.asarray(newLossSynData),np.asarray(newWinSynData)))
    
    return avgSynData

def trainTestSplit(predictors, outcomes, proportion = .20):
    splitPercentage = proportion
    
    predictorsCon0 = predictors[outcomes==0,:]
    predictorsCon1 = predictors[outcomes==1,:]

    testSize = round(len(predictorsCon0)*splitPercentage)
    testIndex = rnd.sample(range(len(predictorsCon0)),testSize)
    
    testCon0 = predictorsCon0[np.r_[testIndex],:]
    testCon1 = predictorsCon1[np.r_[testIndex],:]
    testPredictors = np.vstack((testCon0,testCon1))
    testOutcomes = np.concatenate((np.zeros(len(testCon0)),np.ones(len(testCon1))))
    
    trainCon0 = np.delete(predictorsCon0,testIndex,0)
    trainCon1 = np.delete(predictorsCon1,testIndex,0)
    trainPredictors = np.vstack((trainCon0,trainCon1))
    trainOutcomes = np.concatenate((np.zeros(len(trainCon0)),np.ones(len(trainCon1))))
    
    return (trainPredictors, testPredictors, trainOutcomes, testOutcomes)

def reduceData(EEG):
    #Remove baseline
    extractedEEG = [EEG[trial,100:] for trial in range(len(EEG))]
    
    #Define function
    def downsampleArray(EEGTrial):
        interpModel = interp1d(np.arange(0,500), EEGTrial)
        resampledEEG = interpModel(np.arange(0,500,5))
        return resampledEEG
    
    reducedEEG = [downsampleArray(extractedEEG[trial]) for trial in range(len(extractedEEG))]

    return np.array(reducedEEG)

def cropData(EEG):
    startDatapoint = 41 #200ms
    endDatapoint = 61 #400ms
    croppedEEG = [EEG[trial,startDatapoint:endDatapoint] for trial in range(len(EEG))]
    croppedEEG = [EEG[trial,18:] for trial in range(len(EEG))]
    return np.array(croppedEEG)

def averageEEG(EEG):
    participants = np.unique(EEG[:,0])
    averagedEEG = []
    for participant in participants:
        for condition in range(2):
            averagedEEG.append(np.mean(EEG[(EEG[:,0]==participant)&(EEG[:,1]==condition),:], axis=0))
    return np.array(averagedEEG)

def cutData(synData):
    keepProportion = proportionSynData       
    keepIndex = round((synData.shape[0]*keepProportion)/2)
    
    lossSynData = synData[synData[:,0]==0,:]
    winSynData = synData[synData[:,0]==1,:]

    synData = np.vstack((lossSynData[0:keepIndex,:],winSynData[0:keepIndex,:]))

    return synData

#Load test data to be used for all analyses
if validationOrTest == 'validation':
    EEGDataTest = np.genfromtxt('data/training_data/gansTrialERP_len100_ValidationData.csv', delimiter=',', skip_header=1)
else:
    EEGDataTest = np.genfromtxt('data/training_data/gansTrialERP_len100_TestData.csv', delimiter=',', skip_header=1)

if avgOrTrialData == 'avg':
    EEGDataTest = averageEEG(EEGDataTest)[:,1:]
else:
    EEGDataTest = EEGDataTest[:,1:]
    
y_test = EEGDataTest[:,0]

if features:
    x_features = np.array(extractFeatures(EEGDataTest[:,2:]))
    x_test = x_features
    x_test = scale(x_test, axis=0)
else:
    x_features = EEGDataTest[:,2:]
    x_test = x_features
    x_test = scale(x_test,axis = 1)

if crop:
    x_test = cropData(x_test)

#Load Synthetic Data 
for dataSampleSize in dataSampleSizes:
    for addSyntheticData in syntheticDataOptions:
        pScores = []
        empiricalSamples = int(dataSampleSize)
        for run in range(runs):
            if addSyntheticData:
            #sampleSizes = np.arange(5,450, 5)

                #Load Synthetic Data
                print('Loading Synthetic Data...')
                if filteredGAN:
                    synFilename = 'generated_samples/filtered_checkpoint_SS' + dataSampleSize + '_Run' + str(run).zfill(2) + '_nepochs'+dataEpochs+'.csv'
                else:
                    synFilename = 'generated_samples/checkpoint_SS' + dataSampleSize + '_Run' + str(run).zfill(2) + '_nepochs'+dataEpochs+'.csv'
                synData = np.genfromtxt(synFilename, delimiter=',', skip_header=1)
                print('Synthetic Data Loaded!')
                synData = cutData(synData)
                synOutcomes = synData[:,0]

                #Process synthetic data
                print('Filtering Synthetic Data...')
                tempSynData = filterSyntheticEEG(synData[:,1:]) 
                print('Baseline Correcting Synthetic Data...')
                tempSynData = baselineCorrect(tempSynData)
                print('Synthetic Data Processed!')
                
                del synData #### TODO: This is rough, bud.

                #Create new array for processed synthetic data
                
                processedSynData = np.zeros((len(tempSynData),len(tempSynData[0])+1))
                processedSynData[:,0] = synOutcomes
                processedSynData[:,1:] = np.array(tempSynData)

                del tempSynData #### TODO: This is rough, bud.

                #Average data across trials
                if avgOrTrialData == 'avg':
                    print('Averaging Synthetic Data...')
                    processedSynData = averageSynthetic(processedSynData)
                    print('Done Averaging Synthetic Data!')
                
                #Add to EEG Data
                synOutomes = processedSynData[:,0]
                if features:
                    synFeatures = np.array(extractFeatures(processedSynData[:,1:]))
                    synPredictors = synFeatures
                    synPredictors = scale(synPredictors, axis=0)
                else:
                    synFeatures = processedSynData[:,1:]
                    synPredictors = synFeatures
                    synPredictors = scale(synPredictors, axis=1)

                if crop:
                    synPredictors = cropData(synPredictors)

                del processedSynData #### TODO: This is rough, bud.

            #Load Empirical Data
            tempFilename = 'data/training_data/Runs/gansTrialERP_len100_SS'+dataSampleSize+ '_Run' + str(run).zfill(2) +'.csv'
            EEGData = np.genfromtxt(tempFilename, delimiter=',', skip_header=1)#[:,1:]
            if avgOrTrialData == 'avg':
                EEGData = averageEEG(EEGData)[:,1:]
            else:
                EEGData = EEGData[:,1:]
                
            Y_train = EEGData[:,0]
            if features:
                xFeatures = np.array(extractFeatures(EEGData[:,2:]))
                X_train = xFeatures
                X_train = scale(X_train, axis=0)
            else:
                X_train = EEGData[:,2:]
                X_train = scale(X_train, axis=1)

            if crop:
                X_train = cropData(X_train)
            trainShuffle = rnd.sample(range(len(X_train)),len(X_train))
            Y_train = Y_train[trainShuffle]
            X_train = X_train[trainShuffle,:]
            
            if addSyntheticData:
                Y_train = np.concatenate((Y_train,synOutomes))
                X_train = np.concatenate((X_train,synPredictors))
                #I am re-scalign the X_train now as it was scaled independently from the synthetic data before
                #X_train = scale(X_train)
                
                #Shuffle the dataset (so the samples are not dependent - i.e., the first n being empirical and the last n being synthetic)
                trainShuffle = rnd.sample(range(len(X_train)),len(X_train))
                X_train = X_train[trainShuffle,:]
                Y_train = Y_train[trainShuffle]
                            

            if addSyntheticData:
                f = open(augFilename, 'a')
            else:
                f = open(empFilename, 'a')

            startTime = time.time()
            clear_output(wait=True)
            print('Sample Size: ' + str(empiricalSamples))
            print('Run: ' + str(run))
            print(pScores)

        #Define search space            
            param_grid = [
                {'penalty' : ['l1', 'l2'],
                'C' : np.logspace(-4, 4, 20),
                'solver' : ['liblinear'],
                'max_iter' : [5000, 10000]}]

            #Search over search space
            optimal_params = GridSearchCV(
                LogisticRegression(), 
                param_grid, 
                verbose = False,
                n_jobs = -1)

            optimal_params.fit(X_train, Y_train);

            #Run Neural Network
            logRegOutput = LogisticRegression(C=optimal_params.best_params_['C'], 
                                        penalty=optimal_params.best_params_['penalty'],
                                        solver = optimal_params.best_params_['solver'],
                                        max_iter = optimal_params.best_params_['max_iter'])
            
            logRegOutput.fit(X_train, Y_train)

            #Determine predictability
            predictResults = round(logRegOutput.score(x_test,y_test)*100)
            pScores.append(predictResults)
            
            #print('Prediction: ' + str(round(predictResults['macro avg']['f1-score']*100)) + '%')
            
            if addSyntheticData:
                toWrite = [str(dataSampleSize),str(run),str(dataEpochs),str(round(predictResults)),str(time.time()-startTime),optimal_params.best_params_]
            else:
                toWrite = [str(dataSampleSize),str(run),'0',str(round(predictResults)),str(time.time()-startTime),optimal_params.best_params_]

            for currentWrite in toWrite:
                f.write(str(currentWrite))
                if not currentWrite==toWrite[-1]:
                    f.write(',')
            f.write('\n')
            f.flush()

    f.close()