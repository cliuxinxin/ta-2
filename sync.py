import aiohttp
import asyncio
import async_timeout
import os
from time import time
from datetime import datetime
import json
import sox
from datetime import datetime
import tensorflow as tf
from datasets import audio
import numpy as np
import random

async def call_coroutine(session, url,i,files):
    with async_timeout.timeout(60):
        async with session.get(url) as response:
            content = await response.text()
            content = json.loads(content)
            files[i] = content['wav_file']
            # print(i)
            return await response.release()
 
 
async def main(loop,sentences,files):
    async with aiohttp.ClientSession(loop=loop) as session:
        tasks = [call_coroutine(session, 'http://127.0.0.1:8099/?sentences=' + sentence,i,files) for i,sentence in enumerate(sentences)]
        await asyncio.gather(*tasks)


def synic(sentences,delimiters):
    start = time()
    files = {}
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main(loop,sentences,files))
    files  = sorted(files.items(),key = lambda item:item[0])
    files = [np.asarray(file[1])  for file in files]
    nowTime = datetime.now()
    output_file = 'wav-' + nowTime.strftime('%Y%m%d%H%M%S') + str(nowTime.microsecond) + '.wav'
    if len(files) > 1:
        new_files = []
        for file, delimiter in zip(files, delimiters):
            break_sound = np.zeros(int(16000 * 1 * 0.480))
            if delimiter == '，':
                break_sound = np.zeros(int(16000 * 1 * 0.240))
            new_files.append(np.concatenate((file,break_sound)))
        wav = np.concatenate((new_files), axis=0)
    else:
        wav = files[0]
    audio.save_wav(wav, os.path.join('wav_out/{}'.format(output_file)), sr=16000)
    stop = time()
    print(stop - start)
    return output_file
 
if __name__ == '__main__':
    files = {}
    start = time()
    sentences = "你好。我叫刘新新。他叫白利兵。我们都是电子科技大学毕业的。现在我们都开始工作了。今天的午饭吃的什么东西。我也不知道。不过昨天的午饭我吃的是红烧牛肉。"
    
    sentences = sentences.split("。")
    while '' in sentences:
        sentences.remove('')
    print(synic(sentences))
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main(loop,sentences))
    # files  = sorted(files.items(),key = lambda item:item[0])
    # files = [file[1]+'.wav'  for file in files]
    # break_wav = ["480ms.wav"] * (len(files) - 1)
    # j = 1
    # for i, ele in enumerate(break_wav):
    #     files.insert(i + j, ele)
    #     j += 1
    # files = ['wav_out/' + filename for filename in files]
    # cbn = sox.Combiner()
    # nowTime = datetime.now().strftime('%Y%m%d%H%M%S')
    # output_file = 'wav-' + nowTime + '.wav'
    # cbn.build(files, 'wav_out/' + output_file, 'concatenate')
    # stop = time()
    # print(stop-start)
