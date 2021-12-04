"""Handles requests related to audio/sound updates from websockets"""

import json
import struct
import asyncio
from tensorflow.python.keras.backend import dropout
import websockets, websockets.server
from typing import Dict, Any

from lib.server import sutils
from lib.channel import Channel
path="D:/大四上学习文件/声学所/语音增强/变声器/figaro/lib/"
model_name="Win_30_Stride_5_2021_10_12_11_51_14-e05accuracy0.9822.hdf5"
from tensorflow.keras.models import load_model
model=load_model(path+model_name,custom_objects={"dropout":0.2})
import numpy as np
import librosa
path = ""
whole_array=[]
pos=0
res_array=np.zeros(10)
def test_mosquito(buff):
    global whole_array
    global pos
    global model
    global res_array

    # if len(buff)<2000:
    #     return res_array
    #if count>512*30*6:
    if len(buff)<2:
        return np.array([])
    down_buff = librosa.resample(buff,44100,8000)
    whole_array=np.concatenate((whole_array,down_buff))
    if len(whole_array)>512*30:
        mel_spectrogram = librosa.feature.melspectrogram(whole_array, sr=8000, n_fft=2048, hop_length=512, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        log_mel_spectrogram = (log_mel_spectrogram-np.mean(log_mel_spectrogram))/np.std(log_mel_spectrogram)
        # n_slides = np.shape(log_mel_spectrogram)[1]//30
        # for n in range (n_slides):
        y_pred = log_mel_spectrogram[:,0:30].T
        y_pred = y_pred.reshape(1,1,30, 128)
        pos = 1-model.predict(y_pred)[0][0]
        
        # whole_array=[]
        whole_array=whole_array[512*15:]
        # pos=y_pred[0]
    res_array=np.append(res_array,pos)
    res_array=res_array[1:]
    return res_array
        #all_y_pred.append(model.predict(y_pred)[0]/3)
        

async def send_audio(ws: websockets.server.WebSocketServerProtocol, ch: Channel, scale: float):
    """
    Regularly sends the raw audio data to be displayed.
    """
    while True:
        try:
            buff = test_mosquito(ch.buff * scale)
            await ws.send(struct.pack('f'*len(buff), *buff))
        except websockets.exceptions.ConnectionClosed:
            return
        await asyncio.sleep(0.05)

async def send_sounds(ws: websockets.server.WebSocketServerProtocol, ch: Channel):
    """
    Regularly sends the currently running sounds.
    """
    while True:
        try:
            await ws.send(json.dumps(list(map(lambda s: s.toJSON(), ch.get_sounds()))))
        except websockets.exceptions.ConnectionClosed:
            return
        await asyncio.sleep(0.1)

async def get_audio(ws: websockets.server.WebSocketServerProtocol, key: bytes, req: Dict[str, Any], rid: str, ch: Channel) -> None:
    """
    Handles a request to send periodical audio info coming from a websocket.
    """
    if 'scale' not in req.keys():
        await sutils.error(ws, key, 'Missing parameter `scale`!')
        return
    asyncio.ensure_future(send_audio(ws, ch, req['scale']))

async def get_sounds(ws: websockets.server.WebSocketServerProtocol, key: bytes, req: Dict[str, Any], rid: str, ch: Channel) -> None:
    """
    Handles a request to send periodical sound info coming from a websocket.
    """
    asyncio.ensure_future(send_sounds(ws, ch))
