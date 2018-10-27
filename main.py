#######################################################################################
# @Purpose:  Predict the Characters in The Simposons 
# @Company:  NTUT Speech Communication Lab.
#            https://sites.google.com/site/speechlabx/
# @Author:   TAI-RONG, CHEN <t106368030@ntut.edu.tw>
# @Date:     2018-10-27
# @package:  
# @Input:    
########################################################################################
from readWAV import *


def main():
    
    WAVs ,labels ,listdir = read_WAVs_labels("Berlin Database WAV/")
    for x in range(len(WAVs)):
        log_S = Mel_power_spectrongam(16000,WAVs[x])
        print(log_S.shape)
    plt_data(log_S,labels[x])
    print("WAV amount :",len(WAVs))

if __name__ == '__main__':
    main()