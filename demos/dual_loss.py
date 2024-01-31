    def dual_loss(self):
        dual_loss = 0
        for i, target_name in enumerate(self.target_names):
            loss_mask = tf.greater(self.label_list[i], 0.5)
            user_emb_stop = tf.stop_gradient(self.user_emb_list[i])
            item_emb_stop = tf.stop_gradient(self.item_emb_list[i])

            loss_u = tf.reduce_sum(tf.square(self.user_emb_dual_list[i] - item_emb_stop), axis=-1)
            loss_g = tf.reduce_sum(tf.square(self.item_emb_dual_list[i] - user_emb_stop), axis=-1)

            loss_u = tf.reduce_sum(tf.boolean_mask(loss_u, loss_mask))
            loss_g = tf.reduce_sum(tf.boolean_mask(loss_g, loss_mask))
            dual_loss = dual_loss + loss_u + loss_g
            tf.summary.scalar('{}/{}'.format(target_name, "dual_loss_u"), loss_u)
            tf.summary.scalar('{}/{}'.format(target_name, "dual_loss_g"), loss_g)

        self.final_loss += dual_loss
