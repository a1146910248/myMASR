本项目使用的环境：
 - Anaconda 3
 - Python 3.8
 - Pytorch 1.13.1
 - Windows 10 or Ubuntu 18.04

# 声明
本项目基于MASR项目，仅供自己学习

# 快速使用

这里介绍如何使用MASR快速进行语音识别，前提是要安装MASR，文档请看[快速安装](./docs/install.md)。执行过程不需要手动下载模型，全部自动完成。

1. 短语音识别
```python
from masr.predict import MASRPredictor

predictor = MASRPredictor(model_tag='conformer_streaming_fbank_aishell')

wav_path = 'dataset/test.wav'
result = predictor.predict(audio_data=wav_path, use_pun=False)
score, text = result['score'], result['text']
print(f"识别结果: {text}, 得分: {int(score)}")
```

2. 长语音识别
```python
from masr.predict import MASRPredictor

predictor = MASRPredictor(model_tag='conformer_streaming_fbank_aishell')

wav_path = 'dataset/test_long.wav'
result = predictor.predict_long(audio_data=wav_path, use_pun=False)
score, text = result['score'], result['text']
print(f"识别结果: {text}, 得分: {score}")
```

3. 模拟流式识别
```python
import time
import wave

from masr.predict import MASRPredictor

predictor = MASRPredictor(model_tag='conformer_streaming_fbank_aishell')

# 识别间隔时间
interval_time = 0.5
CHUNK = int(16000 * interval_time)
# 读取数据
wav_path = 'dataset/test.wav'
wf = wave.open(wav_path, 'rb')
data = wf.readframes(CHUNK)
# 播放
while data != b'':
    start = time.time()
    d = wf.readframes(CHUNK)
    result = predictor.predict_stream(audio_data=data, use_pun=False, is_end=d == b'')
    data = d
    if result is None: continue
    score, text = result['score'], result['text']
    print(f"【实时结果】：消耗时间：{int((time.time() - start) * 1000)}ms, 识别结果: {text}, 得分: {int(score)}")
# 重置流式识别
predictor.reset_stream()
