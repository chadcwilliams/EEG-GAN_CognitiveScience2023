import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.multitest import multipletests
import pandas as pd
import itertools

def conductANOVA(filenames):
    
    #### LOAD AND MANIPULATE DATA ####
    #Load data subfunction
    def loadPredictions(filename):
        data = []
        with open(filename) as f:
            [data.append(line.split(',')[0:4]) for line in f.readlines()]
        return np.asarray(data).astype(int)

    #Load data and convert to pandas
    data = np.vstack((loadPredictions(filenames[0]),loadPredictions(filenames[1])))
    data = pd.DataFrame(data, columns = ['SampleSize','Run','Type','Prediction'])
    
    #Recode type of analyses
    data['Type'] = data['Type'].replace(0,'Empirical')
    data['Type'] = data['Type'].replace(8000,'Augmented')
    
    #Create a new combined condition variable for tukey hsd
    data['TypexSampleSize'] = 0
    for row in range(len(data)):
        data['TypexSampleSize'][row] = data['Type'][row] + '_' + str(data['SampleSize'][row]).zfill(3)
    
    #### ANOVA ####
    #Conduct between-subject 2x7 ANOVA
    ANOVA = sm.stats.anova_lm(ols('Prediction ~ C(Type) * SampleSize', data).fit(), typ=3)
    
    #### POST-HOC ####
    #Determine all pairwise comparisons
    pairwiseComparisons = list(itertools.combinations(np.unique(data['TypexSampleSize']),2))
    
    #Conduct a t-test for each pairwise comparison
    pairwiseResults = pd.DataFrame(index = range(len(pairwiseComparisons)), columns = ['Group 1', 'Group 2', 'p-value', 'adj-p'])
    for pairwiseIndex, pairwiseComparison in enumerate(pairwiseComparisons):
        group1 = data[data['TypexSampleSize']==pairwiseComparison[0]]['Prediction']
        group2 = data[data['TypexSampleSize']==pairwiseComparison[1]]['Prediction']
        pairwiseResults.iloc[pairwiseIndex] = pairwiseComparison[0], pairwiseComparison[1], ttest_ind(group1, group2)[1], 'NaN'
        
    #Adjust p-values for multiple comparisons 
    pairwiseResults['adj-p'] = multipletests(list(pairwiseResults['p-value']),alpha = .5, method = 'holm-sidak')[1]
    
    #Create main effect of type (empirical, augmented) summary table
    pairwiseTypeTable = pd.DataFrame(index=range(7), columns = ['group1', 'group2', 'p-value', 'p-adj'])
    pairwiseIndex = 0
    for row in np.array(pairwiseResults):
        if str(row[0]).split('_')[1] == str(row[1]).split('_')[1]:
            pairwiseTypeTable.loc[pairwiseIndex] = row
            pairwiseIndex += 1
                    
    #Create main effect of sample size (5 - 100) summary table
    pairwiseSampleSizeTable = pd.DataFrame(index=range(12), columns = ['group1', 'group2', 'p-value', 'p-adj'])
    sampleSizes = np.array(['005','010','015','020','030','060','100','NaN'])
    pairwiseIndex = 0
    for row in np.array(pairwiseResults):
        if (str(row[0]).split('_')[0] == str(row[1]).split('_')[0]) & (str(row[1]).split('_')[1] == sampleSizes[np.where(sampleSizes==str(row[0]).split('_')[1])[0]+1][0]):
            pairwiseSampleSizeTable.loc[pairwiseIndex] = row
            pairwiseIndex += 1
                    
    print('\n******************************')
    print('******************************\n')
    print('ANOVA RESULTS FOR THE DATASETS\n')
    print(filenames[0].split('/')[1])
    print(filenames[1].split('/')[1])
    print('\n')
    print(ANOVA)
    print('\n******************************')
    print('******************************\n')
    print('POST-HOC RESULTS FOR MAIN EFFECT OF GANs AUGMENTATION\n')
    print('\n')
    print(pairwiseTypeTable)
    print('\n******************************')
    print('******************************\n')
    print('POST-HOC RESULTS FOR MAIN EFFECT OF GANs SAMPLE SIZE\n')
    print('\n')
    print(pairwiseSampleSizeTable)
    print('\n******************************')
    print('******************************')

nnFullfilenames = ['Classification Results/empiricalPredictions_NN_Runs8000_TestClassification.csv','Classification Results/augmentedPredictions_NN_SynP050_Runs8000_Filtered_TestClassification.csv']
nnFeaturesfilenames = ['Classification Results/empiricalPredictions_NN_Runs8000_Features_TestClassification.csv','Classification Results/augmentedPredictions_NN_SynP050_Runs8000_Filtered_Features_TestClassification.csv']

svmFullfilenames = ['Classification Results/empiricalPredictions_SVM_Runs8000_TestClassification.csv','Classification Results/augmentedPredictions_SVM_SynP050_Runs8000_Filtered_TestClassification.csv']
svmFeaturesfilenames = ['Classification Results/empiricalPredictions_SVM_Runs8000_Features_TestClassification.csv','Classification Results/augmentedPredictions_SVM_SynP050_Runs8000_Filtered_Features_TestClassification.csv']


lrFullfilenames = ['Classification Results/empiricalPredictions_LR_Runs8000_TestClassification.csv','Classification Results/augmentedPredictions_LR_SynP050_Runs8000_Filtered_TestClassification.csv']
lrFeaturesfilenames = ['Classification Results/empiricalPredictions_LR_Runs8000_Features_TestClassification.csv','Classification Results/augmentedPredictions_LR_SynP050_Runs8000_Filtered_Features_TestClassification.csv']

conductANOVA(nnFullfilenames)
conductANOVA(nnFeaturesfilenames)

conductANOVA(svmFullfilenames)
conductANOVA(svmFeaturesfilenames)

conductANOVA(lrFullfilenames)
conductANOVA(lrFeaturesfilenames)