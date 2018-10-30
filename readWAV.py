#######################################################################################
# @Purpose:  Predict the Characters in The Simposons 
# @Company:  NTUT Speech Communication Lab.
#            https://sites.google.com/site/speechlabx/
# @Author:   TAI-RONG, CHEN <t106368030@ntut.edu.tw>
# @Date:     2018-10-27
# @package:  
# @Input:   
########################################################################################
import os,sys
from scipy.io import wavfile

import numpy as np
import librosa

import matplotlib.pyplot as plt
import librosa.display

images =[]
labels = []
listdir = []
listshape = []
labels_to_keep = ["anger","boredom","disgust","anxiety/fear","happiness","sadness","neutral"]


def plt_data(data,Label):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(data, x_axis='time', y_axis='mel')
    plt.title('Label :' + labels_to_keep [Label])
    plt.colorbar(format='%+02.0f dB')
    plt.show()

def Mel_power_spectrongam(sample_rate,samples):
    S = librosa.feature.melspectrogram(samples.astype(np.float), sr=sample_rate, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S

def MFCC(log_S):
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    return mfcc,delta2_mfcc
    

def read_WAVs_labels(path,i=0):
    for file in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file))   # abs_path =  C:\\XXX\XXX\ + train\XXX\  ||  +(XXX).jpg 
        # print(abs_path)
        if os.path.isdir(abs_path):
            i+=1
            # print(i)                                           # 1- 20
            temp = os.path.split(abs_path)[-1]                 # C:\\XXX\XXX\ + train\XXX\ >> XXX
            listdir.append(temp)                               # stack file path
            read_WAVs_labels(abs_path,i)                     # read_images_labels(C:\\XXX\XXX\ + train\XXX\)
        else:  
            if file.endswith('.wav'):
                sample_rate, samples = wavfile.read(abs_path)
                
                # listshape.append(np.pad(samples, (0,160000), 'constant').shape)
                listshape.append(samples.shape)
                images.append(np.pad(samples, (0,160000), 'constant'))
                labels.append(i-1)
    print(listshape)
    return images ,labels ,listdir

if __name__ == '__main__':
    plt_data()
    Mel_power_spectrongam()
    MFCC()
    read_WAVs_labels()
