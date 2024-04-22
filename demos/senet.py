def senet_gating(name, nn_inputs, nn_inputs_slot_dims, nn_dims, senet_inputs, concat=False):
    """
    SE-Net: 学习输入特征slot粒度权重的重要性
    params:
        name: weight_name
        nn_inputs: network input
        nn_inputs_slot_dims: dims of every slot
        nn_dims: dims of lhuc network
        lhuc_inputs: input of lhuc network 
    """
    if concat:
        senet_inputs = tf.concat([tf.stop_gradient(nn_inputs), senet_inputs], axis=1)
    if not concat:
        senet_inputs = tf.stop_gradient(nn_inputs)
    slot_weight = xx.mlp(name='{}_slot_gate'.format(name),
                    output_dims=nn_dims+[len(nn_inputs_slot_dims)],
                    activations=[layers.Relu()] * len(nn_dims) + [layers.Sigmoid()],
                    initializers=initializers.GlorotNormal(mode='fan_in')
                    )(senet_inputs) 
    slot_weight = slot_weight * 2  #[0,2] 同时支持对重要性的放大和缩小 
    nn_inputs_splited = tf.split(nn_inputs, nn_inputs_slot_dims, axis=1)
    weight_splited =  tf.split(slot_weight, [1]*len(nn_inputs_slot_dims), axis=1)
    nn_inputs_concated = tf.concat([embed*w for embed,w in zip(nn_inputs_splited, weight_splited)],axis=1)
    return nn_inputs_concated
