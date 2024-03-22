#encoding=utf8
# Copyright 2020 The grace_t Authors. All Rights Reserved.
#  
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  
# http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math

MIN_NORM_VALUE = 1e-06
def kld(p, q):
    """
    计算分布p与分布q的Kullback-Leibler divergence(KLD)
    """
    if 0 in q:
        return 1
    return sum(_p * math.log(_p / _q) for (_p, _q) in zip(p, q) if _p != 0)


def jsd(p, q):
    """
    计算分布p与分布q的Jensen-Shannon Divergence(JSD)
    """
    M = [0.5 * (_p + _q) for _p, _q in zip(p, q)]
    return 0.5 * kld(p, M) + 0.5 * kld(q, M)


def norm(p):
    """
    对topic计数进行归一化，为了计算kld和jsd，归一化后小于1e-06，全置为1e-06
    """
    norm_p = list()
    p_sum = 0
    for count in p:
        p_sum += count
    for count in p:
        if p_sum == 0:
            norm_p.append(MIN_NORM_VALUE)
        elif count * 1.0 / p_sum < MIN_NORM_VALUE:
            norm_p.append(MIN_NORM_VALUE)
        else:
            norm_p.append(count * 1.0 / p_sum)
    return norm_p


def calc_lda_sim(vec1, vec2):
    """calc_lda_sim"""
    vec1 = conv_lda_vec(vec1)
    vec2 = conv_lda_vec(vec2)
    norm_vec1 = norm(vec1)
    norm_vec2 = norm(vec2)
    return jsd(norm_vec1, norm_vec2)

def conv_lda_vec(vec):
    """conv_lda_vec
    vec是一个dict，key是topicid, value是prob   
    """
    r = 1000
    vec_x = [0] * r
    for i in xrange(r):
        if i in vec:
            vec_x[i] = vec[i]
    return vec_x
