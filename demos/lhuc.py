import tensorflow as tf
def lhuc_net(name, nn_dims, nn_initializers, nn_activations, nn_inputs, lhuc_dims, lhuc_inputs,
            scale_last, lhuc_use_nn_input=False):
    if not isinstance(nn_initializers, list):
        nn_initializers = [nn_initializers] * len(nn_dims)
    if not isinstance(nn_activations, list):
        nn_activations = [nn_activations] * (len(nn_dims) - 1) + [None]
    if lhuc_use_nn_input:
        lhuc_final_input = tf.concat([tf.stop_gradient(nn_inputs), lhuc_inputs], axis=1)
    else:
        lhuc_final_input = lhuc_inputs
    cur_layer = nn_inputs
    for idx, nn_dim in enumerate(nn_dims):
        lhuc_scale = DenseTower(name='{}_lhuc_{}'.format(name, idx),
                        output_dims=lhuc_dims+[int(cur_layer.shape[1])], # 先把lhuc输入的维度映射lhuc_dims, 再映射成输入cur_layer的维度
                        activations=[layers.Relu()] * len(lhuc_dims) + [layers.Sigmoid()],
                        initializers=initializers.GlorotNormal(mode='fan_in'),
                        )(lhuc_final_input) ## luhc_scale和cur_layer的dim是一样的
        cur_layer = DenseTower(name='{}_layer_{}'.format(name, idx),
                        output_dims=[nn_dim],
                        initializers=nn_initializers[idx],
                        activations=[nn_activations[idx]], # fatal
                        )(cur_layer * lhuc_scale * 2.0)
    if scale_last:
        lhuc_scale = DenseTower(name='{}_lhuc_{}'.format(name, len(nn_dims)),
                        output_dims=lhuc_dims+[nn_dims[-1]],
                        activations=[layers.Relu()] * len(lhuc_dims) + [layers.Sigmoid()],
                        initializers=initializers.GlorotNormal(mode='fan_in'),
                        )(lhuc_final_input)
        cur_layer = cur_layer * lhuc_scale * 2.0
    return cur_layer


if __name__ == "__main__":
    NN_DIMS = [256, 64, 32]
    LHUC_DIMS = [128]
    xx_out = lhuc_net('deep_xx_tower',
                    nn_dims=UE_NN_DIMS, 
                    nn_initializers=initializers.GlorotNormal(mode='fan_in'),
                    nn_activations=layers.Relu(),
                    nn_inputs=xx,
                    lhuc_dims=LHUC_DIMS,
                    lhuc_inputs=lhuc_embeds,
                    scale_last=True)
