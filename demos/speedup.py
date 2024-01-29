def get_din_emb_infer_softmax_old(self, query, key, value, seq_len):
    batch_size = tf.shape(query)[0]
    ## query: [B, 1, D]
    ## key: [B, L, D]
    ## value: [B, L, D]
    query = tf.reshape(query, [batch_size, 1, self.qk_dim]) # [B, 1, D]
    ## [B,1,D] * [B, D, L] = [B,1,L]
    item_att_w = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.qk_dim, tf.float32)) # [B, 1, L]
    paddings = (1 - seq_len) * tf.float16.min # [B, L]
    paddings = tf.reshape(paddings, [batch_size, 1, self.max_len]) # [B, 1, L]
    item_att_w = item_att_w + paddings # [B, 1, L]
    item_att_w = tf.nn.softmax(item_att_w, axis=-1) # [B, 1, L]
    din_emb = tf.matmul(item_att_w, value) # [B, 1, L] * [B, L, D] -> [B, 1, D]
    din_emb = tf.reshape(din_emb, [batch_size, self.v_dim]) # [B, D]
    return din_emb
    
def get_din_emb_infer_softmax(self, query, key, value, seq_len):
    ## [B, L, D] ==> [B*L, D]
    key = tf.reshape(key, [-1, self.qk_dim]) # [BuL, D]
    ## [B, L, D] ==> [B*L, D]
    value = tf.reshape(value, [-1, self.v_dim]) # [BuL, D]
    ## [B,D] * [D, B*L] = [B, B*L] 
    item_att_w = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.qk_dim, tf.float32)) # [Bg, D] * [BuL, D] -> [Bg, BuL]
    paddings = (1 - seq_len) * tf.float16.min # [Bu, L]
    item_att_w = item_att_w + tf.reshape(paddings, [1, -1]) # [Bg, BuL]
    item_att_w = tf.nn.softmax(item_att_w, axis=-1) # [Bg, BuL]
    ## [B, B*L] * [B*L, D] = [B, D]
    din_emb = tf.matmul(item_att_w, value) # [Bg, BuL] * [BuL, D] -> [Bg, D]
    return din_emb
