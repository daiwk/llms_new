    
import tensorflow as tf
config = 123
fc_dict = {}
g_duplicate_cnts = 444
optimizers = 1
M = 123
# add streaming frequence estimation

# A是这个nid最近出现的step
# B是每隔多少个step这个nid会出现一次
# 1/b就是nid的概率
# t是nid出现的这个step
## B = (1-alpha) * B + alpha * (t - A)
## A = t

assign_optimizer_hit = optimizers.Assign(low=-1.0, high=-1.0)  ##  hit是A数组
MAX_DELTA = 50000.0
assign_optimizer_delta = optimizers.Assign(low=config.MAX_DELTA, high=config.MAX_DELTA)  ## delta是B数组

last_hit_slice = fc_dict[2].feature_slot.add_slice(1, optimizer=assign_optimizer_hit, exclude_norm_clip=True) ## A数组
last_hit = fc_dict[2].get_vector(last_hit_slice)
delta_slice = fc_dict[2].feature_slot.add_slice(1, optimizer=assign_optimizer_delta, exclude_norm_clip=True) ## B数组
delta = fc_dict[2].get_vector(delta_slice)

## 期望
## lr是alpha(学习率)

## delta = (1-lr) * delta + lr * (t - hit)
## hit = t


new_last_hit = tf.ones_like(last_hit)
new_delta = delta
item_weights = tf.ones_like(delta)
@tf.custom_gradient
def update_value_and_gradient(x, y):
    def grad(dy):
        return y, None
    return y, grad

new_last_hit = new_last_hit * tf.cast(M.get_global_step(), tf.float32) ## 当前step t
recent_delta = new_last_hit - last_hit ## t - hit
if config.COUNT_DUPLICATE_HITS:
    recent_delta = recent_delta / g_duplicate_cnts
is_sharp_change = tf.greater(recent_delta, config.SHARP_CHANGE_RATIO * delta)
recent_delta = tf.minimum(config.MAX_DELTA, tf.maximum(recent_delta, config.MIN_DELTA))
# delta * (1-lr) + lr * (t-hit)
new_delta = tf.where(is_sharp_change, recent_delta, delta * (1.0 - config.LEARNING_RATE)+ recent_delta * config.LEARNING_RATE) 
new_delta = tf.minimum(config.MAX_DELTA, tf.maximum(new_delta, config.MIN_DELTA))

## 用t去更新hit
new_last_hit = update_value_and_gradient(last_hit, new_last_hit)
new_delta = update_value_and_gradient(delta, new_delta)
item_weights = tf.log(new_delta / config.MAX_DELTA) * config.EXPONENT