
# import librosa
# import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, 'D:/大四上学习文件/声学所/语音增强/变声器/figaro/res/filters')
# from crackle import Crackle
# from echo import Echo
# from pitch import Pitch
# data,sr = librosa.load("D:/录音.mp3")
# #a = Echo()
# b=Echo.start(["0.5","0.5"])
# # b = a.Filter(0.5,0.5)

# # a.Filter.__init__(a,0.5,0.5)
# output_data = b.apply(data)

# import soundfile as sf
# sf.write("output_echo.wav",output_data,samplerate=sr)
import numpy as np
whole_array = []
def test_mosquito(buff):
    global whole_array
    whole_array=np.concatenate((whole_array,buff))
    if len(whole_array)>1000:

        whole_array=[]
    return np.ones(len(whole_array))*2
test_mosquito([1,2,3,4,4])
a=test_mosquito([235,5,2,26])
print(a)


path="D:/大四上学习文件/声学所/语音增强/变声器/figaro/lib/"
model_name="Win_30_Stride_5_2021_10_12_11_51_14-e05accuracy0.9822.hdf5"
from tensorflow.keras.models import load_model
model=load_model(path+model_name,custom_objects={"dropout":0.2})
import numpy as np
import librosa
path = ""
whole_array=[]
pos=0
def test_mosquito(buff):
    global whole_array
    global pos
    whole_array=np.concatenate((whole_array,buff))
    if len(whole_array)>20000:
        mel_spectrogram = librosa.feature.melspectrogram(whole_array, sr=44100, n_fft=2048, hop_length=512, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        log_mel_spectrogram = (log_mel_spectrogram-np.mean(log_mel_spectrogram))/np.std(log_mel_spectrogram)
        # n_slides = np.shape(log_mel_spectrogram)[1]//30
        # for n in range (n_slides):
        y_pred = log_mel_spectrogram[:,0:30].T
        y_pred = y_pred.reshape(1,1,30, 128)
        res=model.predict(y_pred)[0]
        
        whole_array=[]
        pos=res
    return np.ones(len(buff))*pos