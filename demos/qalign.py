## https://github.com/WNQzhu/Q-align
import tensorflow as tf

RGID_MAX_SIZE=1 # 10 too slow

## 这两个是一样的
BATCH_SIZE=tf.shape(uid_embedding)[0]
BATCH_SIZE_G=tf.shape(gid_embedding)[0]

## alignment：正样本对之间的特征相似程度，
## e^{label*(2t-2t*内积)}，内积越大，距离越近，要最小化这个
def alg(src_emb, pos_emb, mt, coef): 
    mt = tf.expand_dims(mt, axis=1)
    # mt = tf.Print(mt, ['debug label:', mt], summarize=1024)
    multi = tf.multiply(src_emb, pos_emb)
    # multi = tf.Print(multi, ['debug multi 1:', multi], summarize=1024)
    multi = tf.reduce_sum(multi, axis=1, keepdims=True)
    # multi = tf.Print(multi, ['debug multi 2:', multi], summarize=1024)
    coe = tf.multiply(coef, (2 - 2*multi))
    # coe = tf.Print(coe, ['debug coe:', coe], summarize=1024)
    exp_coe = tf.exp(tf.multiply(coe, mt))
    # exp_coe = tf.Print(exp_coe, ['debug exp_coe:', exp_coe], summarize=1024)
    return tf.reduce_sum(exp_coe)

## uniformity：特征向量的分布的均匀程度
## w*e^{2t*内积-t}，内积越小，距离越远，要最小化这个
def uif(emb, t=3, wt=256.0): 
    prod = tf.matmul(emb, tf.transpose(emb))
    # prod = tf.Print(prod, ['debug prod:', prod], summarize=1024)
    coef = 2*t*prod - 2 * t
    # coef = tf.Print(coef, ['debug coef:', coef], summarize=1024)
    exp = tf.exp(coef)
    # exp = tf.Print(exp, ['debug exp:', exp], summarize=1024)
    res = tf.reduce_sum(wt * tf.reduce_mean(exp, axis=1))
    # res = tf.Print(res, ['debug res:', res], summarize=1024)
    return res


## [batchsize * rgid_max_size]
label = tf.reshape(tf.tile(tf.expand_dims(label, [1]), [1,RGID_MAX_SIZE]), [BATCH_SIZE*RGID_MAX_SIZE])
## [batchsize*rgid_max_size, emb_dim]
rgid_embedding=tf.reshape(rgid_embedding, [BATCH_SIZE*RGID_MAX_SIZE, EMBEDDING_SIZE])

uid_embedding = tf.math.l2_normalize(uid_embedding, axis=-1)
rgid_embedding = tf.math.l2_normalize(rgid_embedding, axis=-1)
gid_embedding = tf.math.l2_normalize(gid_embedding, axis=-1)

## [batchsize*rgid_max_size, emb_dim]
rgid_embedding=tf.reshape(rgid_embedding, [BATCH_SIZE*RGID_MAX_SIZE, EMBEDDING_SIZE])
## [batchsize*rgid_max_size, emb_dim]
src_emb = tf.reshape(tf.tile(gid_embedding, [1,RGID_MAX_SIZE]), [BATCH_SIZE_G*RGID_MAX_SIZE, EMBEDDING_SIZE])
## [batchsize*rgid_max_size, emb_dim]
rgid_emb = tf.reshape(tf.tile(rgid_embedding, [1,RGID_MAX_SIZE]), [BATCH_SIZE_G*RGID_MAX_SIZE, EMBEDDING_SIZE])

alpha = 0.5
beta = 0.5
coef = tf.ones_like(tf.expand_dims(label, axis=1))
## 1. 如果这个gid是正样本，那gid和rgid们要尽量接近，如果是负样本，则gid和rgid们要尽量远()
## 2. gid和别的gid们要尽量分散
## 3. rgid内部也要尽量分散
loss = alpha * alg(src_emb, rgid_emb, label, coef)+ beta * uif(src_emb) + beta * uif(rgid_emb)

preds = tf.math.sigmoid(tf.reduce_sum(tf.multiply(src_emb, rgid_emb), axis=1))
