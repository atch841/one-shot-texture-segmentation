import tensorflow as tf
from data_loader import Data_loader
from model import get_model
import time
from tensorflow.contrib import autograph
import numpy as np
from sys import stdout

def weighted_cross_entropy(y_pred, y_gt):
	l = tf.reduce_sum(tf.multiply(y_gt, tf.log(y_pred+10e-100)), axis=1) # +10e-15
	l = tf.reduce_sum(l, axis=1)
	l = tf.reduce_sum(l, axis=1)
	n = tf.reduce_sum(y_gt, axis=1)
	n = tf.reduce_sum(n, axis=1)
	n = tf.reduce_sum(n, axis=1)
	l = -1 * (l / n)
	return l

def total_loss(y_pred, y_gt):
	return tf.reduce_mean(weighted_cross_entropy(y_pred, y_gt) + weighted_cross_entropy((1 - y_pred), (1 - y_gt)))


# @autograph.convert(recursive=True)
def train(data_path, batch_size, max_steps, eval_n_step, init_lr=1e-5):
	is_training = tf.placeholder(tf.bool)
	texture, ref, label, decode_mask = get_model(is_training)
	dl_train = Data_loader(data_path['train'], batch_size)
	dl_val = Data_loader(data_path['val'], batch_size)
	lr = tf.placeholder(tf.float32)
	learning_rate = init_lr
	# opt = tf.train.AdamOptimizer(learning_rate=lr)
	opt = tf.contrib.opt.NadamOptimizer(learning_rate=lr)

	# loss = total_loss(decode_mask, label)
	loss = tf.keras.backend.binary_crossentropy(label, decode_mask)
	loss = tf.reduce_mean(loss)
	print(decode_mask.shape, label.shape, loss.shape)
	# train_op = opt.minimize(loss=loss, global_step=tf.train.get_global_step())
	train_op = opt.minimize(loss)

	saver = tf.train.Saver()
	saver_best = tf.train.Saver()
	init = tf.global_variables_initializer()

	best_loss = np.inf
	loss_log = {'train':[], 'val':[]}
	no_best_cnt = 0
	with tf.Session() as sess:
		sess.run(init)
		# saver.restore(sess, '/fast_data/one_shot_texture_models/best_model0.51105') #55298
		cur_train_loss = 0
		tic = time.time()
		for i in range(max_steps):
			data = dl_train.get_batch_data()	# batch, mask, ref
			_, train_loss = sess.run([train_op, loss], 
				feed_dict={texture: data[0], ref:data[2], label: data[1], lr: learning_rate, is_training: True})
			stdout.write('\r%d, %.5f' % (i, train_loss))
			stdout.flush()
			# print(train_loss)
			cur_train_loss += train_loss

			if i % eval_n_step == 0:
				stdout.write('\r')
				stdout.flush()
				toc = time.time()
				# evaluate validation loss for 10 step
				val_loss = 0
				for _ in range(10):
					test_data = dl_val.get_batch_data()
					val_loss += sess.run(loss, 
						feed_dict={texture: test_data[0], ref:test_data[2], label: test_data[1], is_training: False})
				val_loss /= 10
				if val_loss < best_loss:
					best_loss = val_loss
					print('saving best model (%.5f)' % best_loss)
					saver_best.save(sess, '/fast_data/one_shot_texture_models/best_model%.5f' % best_loss)
					no_best_cnt = 0
				else:
					no_best_cnt += 1

				cur_train_loss /= eval_n_step

				print('%7d/%7d training loss: %.5f, validation loss: %.5f (%d sec)' % (i, max_steps, cur_train_loss, val_loss, toc - tic))
				loss_log['train'].append(cur_train_loss)
				loss_log['val'].append(val_loss)

				cur_train_loss = 0

				saver.save(sess, '/fast_data/one_shot_texture_models/model', global_step=i)

				# if no_best_cnt > 10:
				# 	learning_rate /= 2
				# 	print('setting leaning rate to:', learning_rate)
				# 	no_best_cnt = 0

				tic = time.time()

			if (i + 1) % 4000 == 0:
				learning_rate /= 2
				print('setting leaning rate to:', learning_rate)

	np.save('train_log', loss_log['train'])
	np.save('val_log', loss_log['val'])

if __name__ == '__main__':
	data_path = {
		'train': 'train_texture.npy',
		'val': 'val_texture.npy'
	}
	train(data_path, 8, 2000000, 10)