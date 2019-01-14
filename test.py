import os
import re
from time import time

import numpy as np
import pypinyin
import tensorflow as tf
from pypinyin import lazy_pinyin

from datasets import audio
from hparams import hparams
from tacotron.synthesizerp import Synthesizer


def pinyin_sentence(sentences):
    pinyin = lazy_pinyin(sentences, style=pypinyin.TONE3)
    return " ".join(pinyin)


taco_checkpoint = os.path.join('logs-Tacotron', 'taco_pretrained/')
checkpoint_path = tf.train.get_checkpoint_state(taco_checkpoint).model_checkpoint_path
# logs-Tacotron/taco_pretrained/tacotron_model.ckpt-65000
synth = Synthesizer()
synth.load(checkpoint_path, hparams)

test_times = 1
wav_time = []
generate_time = []

# sentences = [
#     '你好',
#     '您现在这个手机对您打过来',
#     '我们套餐从几块钱到几百的都有，您要看您自己一个月',
#     '您一个月能打多少分钟电话用多少兆流量',
#     '有一个大屏神卡，流量通话不限量使用，但达到一定的值，它会限制',
#     '对，在网站上可以那个要套餐要办的话',
#     '您这边是二百九十六的，再往上就到您这个，噢，您的呢',
#     '您要看您自己一个月能打多少分钟电话，用多少兆流量去选套餐',
#     '嗯，那您还要改那个一百九十九元的吗',
#     '嗯，那您考虑好，有需要再办',
#     '您要咨询您这个套餐怎么扣费呢？还是说是一号之间的消费在哪些方面呢',
#     '这个查某一个时段呢，我查不了，我这边只能看到一号之间总的消费情况',
#     '对是不是买了什么游戏道具之类的',
#     '嗯，那这边我们看到的是叫沃商店，我看一下能不能检测出来',
#     '感谢您的耐心等待上上面写的是粒子搜索',
#     '那您这边应该是游戏吧，您是不是购买了什么五百七十的金币',
#     '上面分析的话是叫粒子搜索',
#     '就是您不要去点，不要去订购就不会收费好的，那请问还有其他的问题吗',
#     '这个我看到，嗯，您还有办理过国防科技四块钱，唉，奇艺十五元，那就是其他对',
#     '积分您是说您用积分兑换兑换过话费还是怎么说没听清',
#     '好，您稍等一下，感谢耐心等待，您这张卡当前可用积分是七百零二分',
#     '用了七百分是什么时候消费的',
#     '您之前用七百分是兑换的什么呢兑换的话费是吧',
#     '观察查询，八月五号是有一个兑换记录',
#     '那您自己通过您当时兑换是通过我们这边网厅手厅兑换的吗？还是积分商城',
#     '不客气，那那您先查一下，还有其它的问题吗',
#     '好的，我的宝贝用心，噢，您好，还在吗',
#     '嗯就是不知道这本机号码是吗',
#     '那您提供一下身份证号码和机主姓名',
#     '嗯好的炫铃需要什么呢',
#     '嗯，好的资料核对正确，那您请记录一下电话号码好吧',
#     '月租费是十九块钱一个月',
#     '大概六毛六毛零一点的费用',
#     '您要查本机余额是吗？那您不要挂机，我给您转到自动台，请稍等',
#     '我帮您看了一下，您有话费的呀，您现在这里还有一百八十块一毛五的',
#     '但是我帮您看了一下，您手机卡是开通状态的功能之前，也是正常的',
#     '您在欠费停机状态下，它也能拨打幺零零幺零的，嗯最低',
#     '嗯，那非常抱歉了，嗯，如果之后有出现的话，就及时给我们打电话进来好吧',
#     '您好，电话已接通，请问什么可以帮您',
#     '余额是吧，我帮您查了一下，您的这边余额的话，显示到还有这个二十七块二毛五',
#     '您好，您现在的话就是说没有收到停机提醒那个短信提醒对吗',
#     '您现在话费的话有四十八块零九毛的费用了，您今天缴了五十块钱',
#     '您缴的费用已经到账了呀',
#     '说起吴总，大家都认识'
#     ]

sentences= [
    '你好',
    '您二零一八年十二月消费合计九十九点零零元',
]

dic = list("①①②③④")

for i in range(test_times):
    print("{} time test".format(str(i)))
    for i,sentence in enumerate(sentences):
        start_time = time()
        sentence = pinyin_sentence(sentence)
        sentence = re.sub(r'[a-z]\d', lambda x: x.group(0)[0] + dic[int(x.group(0)[1])], sentence)
        wav = synth.synthesize([sentence])
        output_file = "test-{}.wav".format(str(i))
        audio.save_wav(wav, os.path.join('wav_out/{}'.format(output_file)), sr=hparams.sample_rate)
        stop_time = time()
        if i == 0:
            pass
        else:
            one_wav_time = len(wav)/hparams.sample_rate
            one_genrate_time = stop_time - start_time
            print("the {} wav len is {}, generate time is {}, the ratio is {}".format(str(i),str(one_wav_time), str(one_genrate_time),str(one_wav_time/one_genrate_time)))
            wav_time.append(one_wav_time)
            generate_time.append(one_genrate_time)

wav_time_mean = np.mean(wav_time)
generate_time_mean = np.mean(generate_time)

print("It will comsume average time about : {}".format(str(generate_time_mean)))
print("The wav average length is {}".format(str(wav_time_mean)))
print("The ratio is {}".format(str(wav_time_mean/generate_time_mean)))
