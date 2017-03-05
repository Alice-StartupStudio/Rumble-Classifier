#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from sklearn.decomposition import PCA
import scipy.io.wavfile as wav

(rate,sig) = wav.read("english.wav")
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate)

# pca = PCA(n_components=2)
# pca.fit(fbank_feat)
# print(fbank_feat[1:3,:])
print(fbank_feat.shape)
