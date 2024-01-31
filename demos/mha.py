def mha(mh_query, mh_kv, q_len, kv_len, seq_mask, name, nhead=MHA_HEAD, ndim=MHA_HEAD_DIM):
    mh_Q = modules.DenseTower(
        name=name + '_mha_query_dense',
        output_dims=[nhead * ndim],
        initializers=get_init_funcs([mh_query.shape[-1]] + [nhead * ndim]))(
        mh_query)  # self: [bs, q_len, nhead*ndim], target: [bs, nhead*ndim]
    mh_K = modules.DenseTower(
        name=name + '_mha_key_dense',
        output_dims=[nhead * ndim],
        initializers=get_init_funcs([mh_kv.shape[-1]] + [nhead * ndim]))(
        mh_kv)  # [bs, kv_len, head*head_dim]
    mh_V = modules.DenseTower(
        name=name + '_mha_value_dense',
        output_dims=[nhead * ndim],
        initializers=get_init_funcs([mh_kv.shape[-1]] + [nhead * ndim]))(
        mh_kv)  # [bs, kv_len, head*head_dim]
    tf.summary.histogram(name + '_mha/mha_out',
                         tf.reduce_sum(seq_mask, axis=-1))

    mh_Q = tf.reshape(mh_Q, (-1, nhead, q_len, ndim))
    mh_K = tf.transpose(tf.reshape(mh_K, (-1, kv_len, nhead, ndim)),
                        perm=[0, 2, 3, 1])  # [bs, head, head_dim, kv_len]
    mh_scores = tf.matmul(mh_Q, mh_K) / tf.sqrt(tf.cast(
        ndim, tf.float32))  # [bs, head, q_len, kv_len]
    for i in range(10):
        tf.summary.histogram(name + "_mha_att_scores_orig/l_{}".format(str(i)),
                             mh_scores[:, :, :, i])

    mask_score = tf.stop_gradient((tf.constant([1], dtype=tf.float32) - seq_mask) * tf.constant([tf.float32.min]))
    mh_scores = tf.nn.softmax(tf.cast(tf.expand_dims(tf.expand_dims(mask_score, 1), 2), tf.float32) + mh_scores)

    for i in range(10):
        tf.summary.histogram(name + "_mha_att_scores/l_{}".format(str(i)),
                             mh_scores[:, :, :, i])

    mh_V = tf.transpose(tf.reshape(mh_V, (-1, kv_len, nhead, ndim)),
                        perm=[0, 2, 1, 3])  # [bs, head, kv_len, head_dim]
    mh_out = tf.matmul(mh_scores, mh_V)  # [bs, head, q_len, head_dim]
    tf.summary.histogram(name + '_mha/mha_out', mh_out)

    for i in range(nhead):
        tf.summary.histogram(name + '_mha/{}th_head_scores'.format(str(i)),
                             mh_scores[:, i, :, :])
        tf.summary.histogram(name + '_mha/{}th_head_out'.format(str(i)),
                             mh_out[:, i, :, :])
    if q_len == 1:
        mh_out = tf.reshape(mh_out, (-1, nhead * ndim))
    else:
        mh_out = tf.transpose(mh_out, perm=[0, 2, 1, 3])
        mh_out = tf.reshape(mh_out, (-1, q_len, nhead * ndim))
    tf.summary.histogram("{}_feature/out".format(name), mh_out)
    print("{} out: ".format(name), mh_out)
    return mh_out
