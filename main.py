#######################################################################################
# @Purpose:  Predict the Characters in The Simposons 
# @Company:  NTUT Speech Communication Lab.
#            https://sites.google.com/site/speechlabx/
# @Author:   TAI-RONG, CHEN <t106368030@ntut.edu.tw>
# @Date:     2018-10-27
# @package:  
# @Input:    
########################################################################################
import sys
from readWAV import *

def main():
    
    WAVs ,labels ,listdir = read_WAVs_labels("Berlin Database WAV/")
    for x in range(2):
        log_S = Mel_power_spectrongam(16000,WAVs[x])
        mfcc,delta2_mfcc=MFCC(log_S)
        # print(mfcc.shape)
    plt_data(mfcc,labels[1])
    print("WAV amount :",len(WAVs))

if __name__ == '__main__':
    main()