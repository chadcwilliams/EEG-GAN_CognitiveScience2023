%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written by Chad C. Williams                                             %
% www.chadcwilliams.com                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filenames = dir('*.mat');
parfor participant = 1:length(filenames)
    participant
    EEGdata = load(filenames(participant).name);
    
    for trialIdx = 1:size(EEGdata.EEG.data,3)
        if strmatch(EEGdata.EEG.epoch(trialIdx).eventtype,'S111')
            break
        end
    end

    trialCounter = 1;
    firstCondition = 1;
    thisData = zeros(1,604)
    dataIndex = 1;
    for trial = 1:size(EEGdata.EEG.data,3)
        for electrode = 1:size(EEGdata.EEG.data,1)
            thisData(dataIndex,1) = participant;
            thisData(dataIndex,2) = trial>=trialIdx;
            thisData(dataIndex,3) = trialCounter;
            thisData(dataIndex,4) = electrode;
            thisData(dataIndex,5:end) = EEGdata.EEG.data(electrode,:,trial);
            dataIndex = dataIndex + 1;
        end
        trialCounter = trialCounter + 1;
    
        if trialCounter >= trialIdx & firstCondition
            trialCounter = 1;
            firstCondition = 0;
        end
    end
    thisDataTable = array2table(thisData);
    tableNames = ["ParticipantID", "Condition", "Trial", "Electrode"];
    for timeIndex = 1:size(EEGdata.EEG.ERP.data,2)
        tableNames(end+1) = strcat("Time", num2str(timeIndex));
    end
    thisDataTable.Properties.VariableNames = tableNames;
    if participant < 10
        pNum = ['000',num2str(participant)];
    elseif participant < 100
        pNum = ['00',num2str(participant)];
    else
        pNum = ['0',num2str(participant)];
    end
    
    writetable(thisDataTable,['ganTrialElectrodeERP_', pNum,'.csv'],'Delimiter',',');
end

%Combine data
filenames = dir('ganTrialElectrode*');
allData = zeros(1,604);
for filenameIndex = 1:length(filenames)
    disp(filenameIndex)
    participantEEG = readmatrix(filenames(filenameIndex).name);
    if filenameIndex == 1
        allData = participantEEG;
    else
        allData(end+1:end+size(participantEEG,1),:) = participantEEG;
    end
    if participant < 10
        pNum = ['000',num2str(participant)];
    elseif participant < 100
        pNum = ['00',num2str(participant)];
    else
        pNum = ['0',num2str(participant)];
    end
    delete(['ganTrialElectrodeERP_', pNum,'.csv']);
end

allDataTable = array2table(allData);
tableNames = ["ParticipantID", "Condition", "Trial", "Electrode"];
for timeIndex = 1:size(EEGdata.EEG.ERP.data,2)
    tableNames(end+1) = strcat("Time", num2str(timeIndex));
end
allDataTable.Properties.VariableNames = tableNames;
writetable(allDataTable,'ganTrialElectrodeERP.csv','Delimiter',',');
zip('ganTrialElectrodeERP','ganTrialElectrodeERP.csv')