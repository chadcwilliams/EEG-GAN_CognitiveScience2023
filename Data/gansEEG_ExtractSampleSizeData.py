###############################################
## IMPORT MODULES                            ##
###############################################
import numpy as np
import random as rnd

###############################################
## SETUP                                     ##
###############################################

#Set seeds for reproducibility
np.random.seed(1251)
rnd.seed(1251)

###############################################
## LOAD DATA                                 ##
###############################################

#Load EEG data
EEGData = np.genfromtxt('Full Datasets/ganTrialERP_len100.csv', delimiter=',', skip_header=1)
EEGDataHeader = np.genfromtxt('Full Datasets/ganTrialERP_len100.csv', delimiter=',', names=True).dtype.names

###############################################
## EXTRACT VALIDATION/TEST DATASETS          ##
###############################################

#Create test file of 400 participants
testParticipants = np.sort(rnd.sample(range(1,int(np.unique(EEGData[:,0])[-1])+1), 400))
testEEGData = []
for participant in testParticipants: 
    #Add to test dataset
    testEEGData.extend(EEGData[EEGData[:,0]==participant,:])
    
    #Remove from dataset to be used for training
    EEGData = np.delete(EEGData, np.where(EEGData[:,0]==participant), axis=0)

#Save File
saveFilename = 'Validation and Test Datasets/gansTrialERP_len100_TestValidationData.csv'
np.savetxt(saveFilename, testEEGData, delimiter=",", fmt='%f', header=','.join(EEGDataHeader), comments='')
    
###############################################
## EXTRACT TRAINING DATASETS                 ##
###############################################

#Iterate through different sample sizes to save data
for sampleSize in [5, 10, 15, 20, 30, 60, 100]:
    for run in range(5):

        #Determine samples of current sample size
        sampleParticipants = np.sort(rnd.sample(list(np.unique(EEGData[:,0])), sampleSize))
        
        #Create new data frame with determined samples
        reducedData = []
        for participant in sampleParticipants:
            reducedData.extend(EEGData[EEGData[:,0]==participant,:])

        #Save file
        saveFilename = 'Training Datasets/gansTrialERP_len100_SS' + str(sampleSize).zfill(3) + '_Run'+ str(run).zfill(2) + '.csv'
        np.savetxt(saveFilename, reducedData, delimiter=",", fmt='%f', header=','.join(EEGDataHeader), comments='')