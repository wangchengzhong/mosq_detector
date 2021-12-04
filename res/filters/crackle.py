"""A crackling filter for audio"""

import numpy as np
from typing import List, Dict, Any

# from numpy.lib.utils import who

import lib.filters.filter
from lib.utils import parse_perc
# from tensorflow.keras.models import load_model
# import librosa
# path = "D:/大四上学习文件/声学所/语音增强/变声器/figaro/lib/"
# model_name = 'Win_30_Stride_5_2021_10_12_11_51_14-e05accuracy0.9822.hdf5'
# model = load_model(path+model_name,custom_objects={"dropout": 0.2})

class Crackle(lib.filters.filter.Filter):
    class Filter(lib.filters.filter.Filter.Filter):
        """
        Adds a crackling effect to audio.

        Attributes
        ----------
        fac : float
            How much crackling should be applied.

        Methods
        -------
        apply(data: np.ndarray)
            Applies the filter and returns the result.
        """

        def __init__(self, fac: float):
            self.fac: float = fac
            # # self.have_mosquito: bool = True
            # self.whole_array=[]
            # self.all_y_pred = []
            # self.flag = 0

        @classmethod
        def parse_args(cls, args: List[str]) -> List[Any]:
            args = [a.strip() for a in args if a.strip()]
            if not args:
                raise Exception('Missing parameter <factor> ... ')
            return [parse_perc(args[0].strip()),]

        # def sendParameters(self, data:np.array) -> List[str]:
        #     self.whole_array=np.concatenate((self.whole_array,data))
        #     while(len(self.whole_array)>10000):
        #         mel_spectrogram = librosa.feature.melspectrogram(self.whole_array, sr=8000, n_fft=2048, hop_length=512, n_mels=128)
                
                
        #         log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        #         log_mel_spectrogram = (log_mel_spectrogram-np.mean(log_mel_spectrogram))/np.std(log_mel_spectrogram)
        #         n_slides = np.shape(log_mel_spectrogram)[1]//30
        #         for n in range (n_slides):
        #             y_pred = log_mel_spectrogram[:,n*30:(n+1)*30].T
        #             y_pred = y_pred.reshape(1,1,30, 128)
        #             self.all_y_pred.append(model.predict(y_pred)[0])
        #             if self.all_y_pred[0]>0.5:
        #                 self.all_y_pred=[]
        #                 self.whole_array=[]
        #                 self.flag = 1
        #             else:
        #                 self.all_y_pred=[]
        #                 self.whole_array=[]
        #                 self.flag = 0
        #     return self.flag
                
            # return super().sendParameters(args)

        def update(self, *args: List[Any]) -> None:
            self.fac = args[0]

        def apply(self, data: np.ndarray) -> np.ndarray:
            ifac = 1 - .9 * self.fac
            # # ifac = 1 - .9 *  self.fac
            # if self.sendParameters(data):
            #     return np.ones(len(data))
            # else:
            #     return np.zeros(len(data))
            return data.clip(data.min() * ifac, data.max() * ifac) * (.5 / ifac)

        def toJSON(self) -> Dict[str, Any]:
            return dict(name='Crackle', fac=self.fac)

        def __call__(self, data: np.ndarray) -> np.ndarray:
            return self.apply(data)

        def __str__(self) -> str:
            return f'Crackle({self.fac*100:.2f}%)'
        
        

    desc: str = 'Adds a crackling effect to your audio!'


    @classmethod
    def start(cls, args: List[str]) -> "Crackle.Filter":
        return Crackle.Filter(*Crackle.Filter.parse_args(args))

    @classmethod
    def props(cls) -> List[Dict[str, Any]]:
        return [dict(name='fac', min=0, max=1, step=.01),]