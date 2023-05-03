###################################################################
####Modules
###################################################################

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from scipy import signal
import scipy
import random as rnd

###################################################################
####User Inputs
###################################################################

features = False
validationOrTest = 'test'
evaluationApproach = 'TRTR' #TSTR, [TRTS], TRTR, [TSTS]
saveFilename = 'TESTevaluationPredictions_GridSearch_TestClassification.csv'

print('Validation or Test: ' + validationOrTest)
print('Evaluation Approach: ' + evaluationApproach)
print('Save Filename: ' + saveFilename)

###################################################################
####Functions 
###################################################################

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

#Define ERP Extraction Function
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

#Define FFT Extraction Function
def extractFFT(EEG):
    
    def runFFT(EEG):
        numberDataPoints = len(EEG) #Determine how many datapoints will be transformed per trial and channel
        SR = 83.3333 #This is because the original data was 600 datapoints at 500Hz and we downsampled to 100 datapoints
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

#Define a feature extraction function
def extractFeatures(EEG):
    erpFeatures = extractERP(EEG)
    fftFeatures = extractFFT(EEG)
    eegFeatures = np.transpose(np.vstack((erpFeatures,np.transpose(fftFeatures))))
    return eegFeatures

#Define the average data function for synthetic data
def averageSynthetic(synData):
    samplesToAverage = 50

    lossSynData = synData[synData[:,0]==0,:]
    winSynData = synData[synData[:,0]==1,:]

    lossTimeIndices = np.arange(0,lossSynData.shape[0],samplesToAverage)
    winTimeIndices = np.arange(0,winSynData.shape[0],samplesToAverage)
    
    newLossSynData = [np.insert(np.mean(lossSynData[int(trialIndex):int(trialIndex)+samplesToAverage,1:],axis=0),0,0) for trialIndex in lossTimeIndices]
    newWinSynData = [np.insert(np.mean(winSynData[int(trialIndex):int(trialIndex)+samplesToAverage,1:],axis=0),0,1) for trialIndex in winTimeIndices]

    avgSynData = np.vstack((np.asarray(newLossSynData),np.asarray(newWinSynData)))
    
    return avgSynData

#Define the average data function for empirical data
def averageEEG(EEG):
    participants = np.unique(EEG[:,0])
    averagedEEG = []
    for participant in participants:
        for condition in range(2):
            averagedEEG.append(np.mean(EEG[(EEG[:,0]==participant)&(EEG[:,1]==condition),:], axis=0))
    return np.array(averagedEEG)

###################################################################
####Load Train Data
###################################################################

#Train Real:
if (evaluationApproach == 'TRTS') | (evaluationApproach == 'TRTR'):
    EEGData = np.genfromtxt('../Data/Training Datasets/gansTrialERP_len100_SS100_Run00.csv', delimiter=',', skip_header=1)
    EEGData = averageEEG(EEGData)[:,1:]
    Y_train = EEGData[:,0]
    if features:
        X_train = np.array(extractFeatures(EEGData[:,2:]))
        X_train = scale(X_train, axis=0)
    else:
        X_train = EEGData[:,2:]
        X_train = scale(X_train, axis=1)

#Train Synthetic
elif (evaluationApproach == 'TSTR') | (evaluationApproach == 'TSTS'):
    EEGData = np.genfromtxt('../GANs/GAN Generated Data/filtered_checkpoint_SS100_Run00_nepochs8000.csv', delimiter=',', skip_header=1)
    tempEEGData = filterSyntheticEEG(EEGData[:,1:]) 
    tempEEGData = baselineCorrect(tempEEGData)
    
    processedEEGData = np.zeros((len(tempEEGData),len(tempEEGData[0])+1))
    processedEEGData[:,0] = EEGData[:,0]
    processedEEGData[:,1:] = np.array(tempEEGData)
    processedEEGData = averageSynthetic(processedEEGData)

    Y_train = processedEEGData[:,0]
    if features:
        X_train = np.array(extractFeatures(processedEEGData[:,1:]))
        X_train = scale(X_train, axis=0)
    else:
        X_train = processedEEGData[:,1:]
        X_train = scale(X_train, axis=1)

else:
    print('Train set unsupported')

trainShuffle = rnd.sample(range(len(X_train)),len(X_train))
X_train = X_train[trainShuffle,:]
Y_train = Y_train[trainShuffle]

###################################################################
####Load Test Data
###################################################################

#Test Real:
if (evaluationApproach == 'TSTR') | (evaluationApproach == 'TRTR'):
    if validationOrTest == 'validation':
        EEGDataTest = np.genfromtxt('../Data/Validation and Test Datasets/gansTrialERP_len100_ValidationData.csv', delimiter=',', skip_header=1)
    else:
        EEGDataTest = np.genfromtxt('../Data/Validation and Test Datasets/gansTrialERP_len100_TestData.csv', delimiter=',', skip_header=1)
    
    EEGDataTest = averageEEG(EEGDataTest)[:,1:]
    y_test = EEGDataTest[:,0]
    if features:
        x_test = np.array(extractFeatures(EEGDataTest[:,2:]))
        x_test = scale(x_test, axis=0)
    else:
        x_test = EEGDataTest[:,2:]
        x_test = scale(x_test, axis=1)

#Test Synthetic:
elif (evaluationApproach == 'TRTS') | (evaluationApproach == 'TSTS'):
    if validationOrTest == 'validation':
        ## ADD SYNTHETIC TEST/VALIDATION SET HERE
        #EEGDataTest = 
        print('Currently unsupported.')
    else:
        ## ADD SYNTHETIC TEST/VALIDATION SET HERE
        #EEGDataTest = 
        print('Currently unsupported.')

#Error Message
else:
    print('Test set unsupported')

###################################################################
####Run Classifier
###################################################################
for run in range(100):
    print('Run: ' + str(run))

    #Define search space
    param_grid = [
        {'hidden_layer_sizes': [(25,), (50,), (25, 25), (50,50), (50,25,50)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter' : [5000, 10000, 20000, 50000]}]

    #Search over search space
    optimal_params = GridSearchCV(
        MLPClassifier(), 
        param_grid, 
        verbose = True,
        n_jobs = -1)

    optimal_params.fit(X_train, Y_train)

    #Run Neural Network
    neuralNetOutput = MLPClassifier(hidden_layer_sizes=optimal_params.best_params_['hidden_layer_sizes'], 
                                activation=optimal_params.best_params_['activation'],
                                solver = optimal_params.best_params_['solver'], 
                                alpha = optimal_params.best_params_['alpha'], 
                                learning_rate = optimal_params.best_params_['learning_rate'], 
                                max_iter = optimal_params.best_params_['max_iter'])

    neuralNetOutput.fit(X_train, Y_train)

    #Determine predictability
    y_true, y_pred = y_test , neuralNetOutput.predict(x_test)
    predictResults = classification_report(y_true, y_pred, output_dict=True)
    predictScore = round(predictResults['accuracy']*100)

    #Append file with results
    f = open(saveFilename, 'a')
    toWrite = [evaluationApproach,str(run),str(predictScore),optimal_params.best_params_]
    for currentWrite in toWrite:
        f.write(str(currentWrite))
        if not currentWrite==toWrite[-1]:
            f.write(',')
    f.write('\n')
    f.flush()

f.close()