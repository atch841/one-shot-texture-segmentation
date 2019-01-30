import tensorflow as tf
from tensorflow.contrib import autograph

def get_model(is_training):
	texture = tf.placeholder(tf.float32, shape=(None, 256, 256, 3), name='texture_input')
	ref = tf.placeholder(tf.float32, shape=(None, 64, 64, 3), name='ref_input')
	label = tf.placeholder(tf.float32, shape=(None, 256, 256, 1), name='label')

	with tf.variable_scope('VGG'):
		# texture vgg
		vgg1 = tf.layers.conv2d(texture, 64, 3, padding='SAME', activation=tf.nn.relu, name='conv1_1')
		# vgg1 = tf.layers.batch_normalization(vgg1, training=is_training)
		v = tf.layers.conv2d(vgg1, 64, 3, padding='SAME', activation=tf.nn.relu, name='conv1_2')
		# v = tf.layers.batch_normalization(v, training=is_training)
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg2 = tf.layers.conv2d(v, 128, 3, padding='SAME', activation=tf.nn.relu, name='conv2_1')
		# vgg2 = tf.layers.batch_normalization(vgg2, training=is_training)
		v = tf.layers.conv2d(vgg2, 128, 3, padding='SAME', activation=tf.nn.relu, name='conv2_2')
		# v = tf.layers.batch_normalization(v, training=is_training)
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg3 = tf.layers.conv2d(v, 256, 3, padding='SAME', activation=tf.nn.relu, name='conv3_1')
		# vgg3 = tf.layers.batch_normalization(vgg3, training=is_training)
		v = tf.layers.conv2d(vgg3, 256, 3, padding='SAME', activation=tf.nn.relu, name='conv3_2')
		# v = tf.layers.batch_normalization(v, training=is_training)
		v = tf.layers.conv2d(v, 256, 3, padding='SAME', activation=tf.nn.relu, name='conv3_3')
		# v = tf.layers.batch_normalization(v, training=is_training)
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg4 = tf.layers.conv2d(v, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv4_1')
		# vgg4 = tf.layers.batch_normalization(vgg4, training=is_training)
		v = tf.layers.conv2d(vgg4, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv4_2')
		# v = tf.layers.batch_normalization(v, training=is_training)
		v = tf.layers.conv2d(v, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv4_3')
		# v = tf.layers.batch_normalization(v, training=is_training)
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg5 = tf.layers.conv2d(v, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv5_1')
		# vgg5 = tf.layers.batch_normalization(vgg5, training=is_training)
		print('vgg:', vgg1.shape, vgg2.shape, vgg3.shape, vgg4.shape, vgg5.shape)

		# reference texture vgg
		vgg1_ref = tf.layers.conv2d(ref, 64, 3, padding='SAME', activation=tf.nn.relu, name='conv1_1', reuse=True)
		# vgg1_ref = tf.layers.batch_normalization(vgg1_ref, training=is_training)
		v = tf.layers.conv2d(vgg1_ref, 64, 3, padding='SAME', activation=tf.nn.relu, name='conv1_2', reuse=True)
		# v = tf.layers.batch_normalization(v, training=is_training)
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg2_ref = tf.layers.conv2d(v, 128, 3, padding='SAME', activation=tf.nn.relu, name='conv2_1', reuse=True)
		# vgg2_ref = tf.layers.batch_normalization(vgg2_ref, training=is_training)
		v = tf.layers.conv2d(vgg2_ref, 128, 3, padding='SAME', activation=tf.nn.relu, name='conv2_2', reuse=True)
		# v = tf.layers.batch_normalization(v, training=is_training)
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg3_ref = tf.layers.conv2d(v, 256, 3, padding='SAME', activation=tf.nn.relu, name='conv3_1', reuse=True)
		# vgg3_ref = tf.layers.batch_normalization(vgg3_ref, training=is_training)
		v = tf.layers.conv2d(vgg3_ref, 256, 3, padding='SAME', activation=tf.nn.relu, name='conv3_2', reuse=True)
		# v = tf.layers.batch_normalization(v, training=is_training)
		v = tf.layers.conv2d(v, 256, 3, padding='SAME', activation=tf.nn.relu, name='conv3_3', reuse=True)
		# v = tf.layers.batch_normalization(v, training=is_training)
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg4_ref = tf.layers.conv2d(v, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv4_1', reuse=True)
		# vgg4_ref = tf.layers.batch_normalization(vgg4_ref, training=is_training)
		v = tf.layers.conv2d(vgg4_ref, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv4_2', reuse=True)
		# v = tf.layers.batch_normalization(v, training=is_training)
		v = tf.layers.conv2d(v, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv4_3', reuse=True)
		# v = tf.layers.batch_normalization(v, training=is_training)
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg5_ref = tf.layers.conv2d(v, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv5_1', reuse=True)
		# vgg5_ref = tf.layers.batch_normalization(vgg5_ref, training=is_training)
		print('vgg_ref:', vgg1_ref.shape, vgg2_ref.shape, vgg3_ref.shape, vgg4_ref.shape, vgg5_ref.shape)


	with tf.variable_scope('encoding_network'):
		print('encoding')
		# texture encode
		v = tf.layers.conv2d(vgg5, 512, 1, padding='SAME', name='conv1')
		v = resblock(v, 512, name='res1', is_training=is_training)
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg4], axis=3)
		print(v.shape)

		v = tf.layers.conv2d(v, 512, 1, padding='SAME', name='conv2')
		v = resblock(v, 512, name='res2', is_training=is_training)
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg3], axis=3)
		print(v.shape)

		v = tf.layers.conv2d(v, 256, 1, padding='SAME', name='conv3')
		v = resblock(v, 256, name='res3', is_training=is_training)
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg2], axis=3)
		print(v.shape)

		v = tf.layers.conv2d(v, 128, 1, padding='SAME', name='conv4')
		v = resblock(v, 128, name='res4', is_training=is_training)
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg1], axis=3)
		print(v.shape)

		v = tf.layers.conv2d(v, 128, 1, padding='SAME', name='conv5')
		v = resblock(v, 128, name='res5', is_training=is_training)

		encode_texture = tf.layers.conv2d(v, 64, 1, padding='SAME', name='conv_out')
		print('encode_texture:', encode_texture.shape)


		# reference texture encode
		v = tf.layers.conv2d(vgg5_ref, 512, 1, padding='SAME', name='conv1', reuse=True)
		v = resblock(v, 512, name='res1', reuse=True, is_training=is_training)
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg4_ref], axis=3)
		print(v.shape)

		v = tf.layers.conv2d(v, 512, 1, padding='SAME', name='conv2', reuse=True)
		v = resblock(v, 512, name='res2', reuse=True, is_training=is_training)
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg3_ref], axis=3)
		print(v.shape)

		v = tf.layers.conv2d(v, 256, 1, padding='SAME', name='conv3', reuse=True)
		v = resblock(v, 256, name='res3', reuse=True, is_training=is_training)
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg2_ref], axis=3)
		print(v.shape)

		v = tf.layers.conv2d(v, 128, 1, padding='SAME', name='conv4', reuse=True)
		v = resblock(v, 128, name='res4', reuse=True, is_training=is_training)
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg1_ref], axis=3)
		print(v.shape)

		v = tf.layers.conv2d(v, 128, 1, padding='SAME', name='conv5', reuse=True)
		v = resblock(v, 128, name='res5', reuse=True, is_training=is_training)

		encode_ref = tf.layers.conv2d(v, 64, 1, padding='SAME', name='conv_out', reuse=True)
		print('encode_ref:', encode_ref.shape)

	cor = correlation(encode_texture, encode_ref)
	print('cor:', cor.shape)

	with tf.variable_scope('decoding_network'):
		print('decoding')

		d_cor = tf.image.resize_images(cor, [16, 16])
		d_texture = tf.image.resize_images(encode_texture, [16, 16])
		d_v = tf.layers.conv2d(vgg5, 64, 1, padding='SAME', name='conv1_1')
		v = tf.concat([d_v, d_cor, d_texture], axis=3)
		print(v.shape)

		v = tf.layers.conv2d(v, 64, 1, padding='SAME', name='conv2_1')
		v = resblock(v, 64, name='res2', is_training=is_training)
		v = tf.keras.layers.UpSampling2D()(v)
		d_cor = tf.image.resize_images(cor, [32, 32])
		d_v = tf.layers.conv2d(vgg4, 64, 1, padding='SAME', name='conv2_2')
		v = tf.concat([d_v, d_cor, v], axis=3)
		print(v.shape)

		v = tf.layers.conv2d(v, 64, 1, padding='SAME', name='conv3_1')
		v = resblock(v, 64, name='res3', is_training=is_training)
		v = tf.keras.layers.UpSampling2D()(v)
		d_cor = tf.image.resize_images(cor, [64, 64])
		d_v = tf.layers.conv2d(vgg3, 64, 1, padding='SAME', name='conv3_2')
		v = tf.concat([d_v, d_cor, v], axis=3)
		print(v.shape)

		v = tf.layers.conv2d(v, 64, 1, padding='SAME', name='conv4_1')
		v = resblock(v, 64, name='res4', is_training=is_training)
		v = tf.keras.layers.UpSampling2D()(v)
		d_cor = tf.image.resize_images(cor, [128, 128])
		d_v = tf.layers.conv2d(vgg2, 64, 1, padding='SAME', name='conv4_2')
		v = tf.concat([d_v, d_cor, v], axis=3)
		print(v.shape)

		v = tf.layers.conv2d(v, 64, 1, padding='SAME', name='conv5_1')
		v = resblock(v, 64, name='res5', is_training=is_training)
		v = tf.keras.layers.UpSampling2D()(v)
		# d_cor = tf.image.resize_images(cor, [256, 256])
		d_v = tf.layers.conv2d(vgg1, 64, 1, padding='SAME', name='conv5_2')
		v = tf.concat([d_v, cor, v], axis=3)
		print(v.shape)

		v = tf.layers.conv2d(v, 64, 1, padding='SAME', name='conv6_1')
		v = resblock(v, 64, name='res6', is_training=is_training)

		decode_mask = tf.layers.conv2d(v, 1, 1, padding='SAME', activation=tf.sigmoid, name='conv_out')
		print('decode_out:', decode_mask.shape)

	return texture, ref, label, decode_mask


def correlation(texture, ref):
	# texture = norm_to_one(texture)
	# ref = norm_to_one(ref)
	texture = tf.nn.l2_normalize(texture, 3)
	ref = tf.nn.l2_normalize(ref, 3)
	cor = tf.nn.conv2d(texture, ref, [1, 1, 1, 1], padding='SAME', name='correlation')
	return cor

@autograph.convert()
def norm_to_one(t):
	s = tf.reduce_sum(t, axis=3, keepdims=True)
	return t / s

def resblock(input, filters, name, reuse=None, is_training=True):
	with tf.variable_scope(name):
		r = tf.layers.conv2d(input, filters, 3, padding='SAME', activation=tf.nn.relu, name='conv1', reuse=reuse)
		# r = tf.layers.batch_normalization(r, training=is_training, reuse=reuse)
		r = tf.layers.conv2d(r, filters, 3, padding='SAME', activation=tf.nn.relu, name='conv2', reuse=reuse)
		# r = tf.layers.batch_normalization(r, training=is_training, reuse=reuse)
		r = tf.layers.conv2d(r, filters, 3, padding='SAME', name='conv3', reuse=reuse)
		# r = tf.layers.batch_normalization(r, training=is_training, reuse=reuse)
		r = tf.add(input, r)
		return r