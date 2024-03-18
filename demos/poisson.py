#encoding=utf8

label_for_loss = tf.clip_by_value(label, 0.0, 20)
logit = tf.clip_by_value(logit, lower_power_bound, upper_power_bound)

## 预估发生次数exp(logit)，个人感觉，其实这里也可以不用exp(logit)，只要是一个0到正无穷的数就行, 
pred = tf.clip_by_value(tf.cast(tf.exp(logit), pred.dtype), 0.0, 20)

## poisson分布：
## 预估发生次数x，z是实际发生次数
## https://www.tensorflow.org/api_docs/python/tf/nn/log_poisson_loss
## 函数输入是log(x) = log_input
## loss = -log(exp(-x) * (x^z) / z!) 
## =...
## = math_ops.exp(log_input) - log_input * targets + constant

loss_ins = tf.nn.log_poisson_loss(targets=label_for_loss, log_input=tf.log(pred))
loss = tf.reduce_sum(tf.boolean_mask(loss_ins, loss_mask))

