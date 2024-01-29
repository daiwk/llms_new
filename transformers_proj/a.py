
linear_alpha = 150.0
label_for_loss = tf.clip_by_value(label / 100.0, 0.0, 10000.0) / linear_alpha
label = tf.cast(label_for_loss * linear_alpha, label.dtype)
pred = tf.exp(logit)
pred = tf.cast(pred * linear_alpha, pred.dtype)
weighted_lr_loss = -label_for_loss * logit + (1 + label_for_loss) * (tf.nn.relu(logit) + tf.log(1 + tf.exp(-tf.abs(logit))))
