import tensorflow as tf
from tensorflow.contrib import autograph

def get_model():
	texture = tf.placeholder(tf.float32, shape=(None, 256, 256, 3), name='texture_input')
	ref = tf.placeholder(tf.float32, shape=(None, 64, 64, 3), name='ref_input')
	label = tf.placeholder(tf.float32, shape=(None, 256, 256, 1), name='label')

	with tf.variable_scope('VGG'):
		# texture vgg
		vgg1 = tf.layers.conv2d(texture, 64, 3, padding='SAME', activation=tf.nn.relu, name='conv1_1')
		v = tf.layers.conv2d(vgg1, 64, 3, padding='SAME', activation=tf.nn.relu, name='conv1_2')
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg2 = tf.layers.conv2d(v, 128, 3, padding='SAME', activation=tf.nn.relu, name='conv2_1')
		v = tf.layers.conv2d(vgg2, 128, 3, padding='SAME', activation=tf.nn.relu, name='conv2_2')
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg3 = tf.layers.conv2d(v, 256, 3, padding='SAME', activation=tf.nn.relu, name='conv3_1')
		v = tf.layers.conv2d(vgg2, 256, 3, padding='SAME', activation=tf.nn.relu, name='conv3_2')
		v = tf.layers.conv2d(v, 256, 3, padding='SAME', activation=tf.nn.relu, name='conv3_3')
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg4 = tf.layers.conv2d(v, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv4_1')
		v = tf.layers.conv2d(vgg4, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv4_2')
		v = tf.layers.conv2d(v, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv4_3')
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg5 = tf.layers.conv2d(v, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv5_1')

		# reference texture vgg
		vgg1_ref = tf.layers.conv2d(ref, 64, 3, padding='SAME', activation=tf.nn.relu, name='conv1_1', reuse=True)
		v = tf.layers.conv2d(vgg1, 64, 3, padding='SAME', activation=tf.nn.relu, name='conv1_2', reuse=True)
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg2_ref = tf.layers.conv2d(v, 128, 3, padding='SAME', activation=tf.nn.relu, name='conv2_1', reuse=True)
		v = tf.layers.conv2d(vgg2, 128, 3, padding='SAME', activation=tf.nn.relu, name='conv2_2', reuse=True)
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg3_ref = tf.layers.conv2d(v, 256, 3, padding='SAME', activation=tf.nn.relu, name='conv3_1', reuse=True)
		v = tf.layers.conv2d(vgg2, 256, 3, padding='SAME', activation=tf.nn.relu, name='conv3_2', reuse=True)
		v = tf.layers.conv2d(v, 256, 3, padding='SAME', activation=tf.nn.relu, name='conv3_3', reuse=True)
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg4_ref = tf.layers.conv2d(v, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv4_1', reuse=True)
		v = tf.layers.conv2d(vgg4, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv4_2', reuse=True)
		v = tf.layers.conv2d(v, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv4_3', reuse=True)
		v = tf.layers.max_pooling2d(v, 2, 2, padding='SAME')

		vgg5_ref = tf.layers.conv2d(v, 512, 3, padding='SAME', activation=tf.nn.relu, name='conv5_1', reuse=True)


	with tf.variable_scope('encoding_network'):
		# texture encode
		v = tf.conv2d(vgg5, 512, 1, padding='SAME', name='conv1')
		v = resblock(v, 512, name='res1')
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg4], axis=3)

		v = tf.conv2d(v, 512, 1, padding='SAME', name='conv2')
		v = resblock(v, 512, name='res2')
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg3], axis=3)

		v = tf.conv2d(v, 256, 1, padding='SAME', name='conv3')
		v = resblock(v, 256, name='res3')
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg2], axis=3)

		v = tf.conv2d(v, 128, 1, padding='SAME', name='conv4')
		v = resblock(v, 128, name='res4')
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg1], axis=3)

		v = tf.conv2d(v, 128, 1, padding='SAME', name='conv5')
		v = resblock(v, 128, name='res5')

		encode_texture = tf.conv2d(v, 64, 1, padding='SAME', name='conv_out')


		# reference texture encode
		v = tf.conv2d(vgg5_ref, 512, 1, padding='SAME', name='conv1', reuse=True)
		v = resblock(v, 512, name='res1', reuse=True)
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg4_ref], axis=3)

		v = tf.conv2d(v, 512, 1, padding='SAME', name='conv2', reuse=True)
		v = resblock(v, 512, name='res2', reuse=True)
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg3_ref], axis=3)

		v = tf.conv2d(v, 256, 1, padding='SAME', name='conv3', reuse=True)
		v = resblock(v, 256, name='res3', reuse=True)
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg2_ref], axis=3)

		v = tf.conv2d(v, 128, 1, padding='SAME', name='conv4', reuse=True)
		v = resblock(v, 128, name='res4', reuse=True)
		v = tf.keras.layers.UpSampling2D()(v)
		v = tf.concat([v, vgg1_ref], axis=3)

		v = tf.conv2d(v, 128, 1, padding='SAME', name='conv5', reuse=True)
		v = resblock(v, 128, name='res5', reuse=True)

		encode_ref = tf.conv2d(v, 64, 1, padding='SAME', name='conv_out', reuse=True)

	cor = correlation(encode_texture, encode_ref)

	with tf.variable_scope('decoding_network'):
		d_cor = tf.image.resize_images(cor, [16, 16])
		d_texture = tf.image.resize_image(encode_texture, [16, 16])
		d_v = tf.conv2d(vgg5, 64, 1, padding='SAME', name='conv1_1')
		v = tf.concat([d_v, d_cor, d_texture], axis=3)

		v = tf.conv2d(v, 64, 1, padding='SAME', name='conv2_1')
		v = resblock(v, 64, name='res2')
		v = tf.keras.layers.UpSampling2D()(v)
		d_cor = tf.image.resize_image(cor, [32, 32])
		d_v = tf.conv2d(vgg4, 64, 1, padding='SAME', name='conv2_2')
		v = tf.concat([d_v, d_cor, v])

		v = tf.conv2d(v, 64, 1, padding='SAME', name='conv3_1')
		v = resblock(v, 64, name='res3')
		v = tf.keras.layers.UpSampling2D()(v)
		d_cor = tf.image.resize_image(cor, [32, 32])
		d_v = tf.conv2d(vgg3, 64, 1, padding='SAME', name='conv3_2')
		v = tf.concat([d_v, d_cor, v])

		v = tf.conv2d(v, 64, 1, padding='SAME', name='conv4_1')
		v = resblock(v, 64, name='res4')
		v = tf.keras.layers.UpSampling2D()(v)
		d_cor = tf.image.resize_image(cor, [32, 32])
		d_v = tf.conv2d(vgg2, 64, 1, padding='SAME', name='conv4_2')
		v = tf.concat([d_v, d_cor, v])

		v = tf.conv2d(v, 64, 1, padding='SAME', name='conv5_1')
		v = resblock(v, 64, name='res5')
		v = tf.keras.layers.UpSampling2D()(v)
		d_cor = tf.image.resize_image(cor, [32, 32])
		d_v = tf.conv2d(vgg1, 64, 1, padding='SAME', name='conv5_2')
		v = tf.concat([d_v, d_cor, v])

		v = tf.conv2d(v, 64, 1, padding='SAME', name='conv6_1')
		v = resblock(v, 64, name='res6')

		decode_mask = tf.conv2d(v, 1, 1, padding='SAME', name='conv_out')

	return texture, ref, label, decode_mask

@autograph.convert()
def correlation(texture, ref):
	texture = tf.nn.softmax(texture)
	ref = tf.nn.softmax(ref)
	texture = tf.image.resize_image_with_crop_or_pad(texture, 320, 320)
	cor = []
	autograph.set_element_type(cor, tf.float32)
	for i in range(256):
		for j in range(256):
			texture_patch = tf.image.crop_to_bounding_box(texture, i, j, 64, 64)
			m = tf.multiply(tf.nn.l2_normalize(texture_patch, 1), tf.nn.l2_normalize(ref, 1))
			l = tf.reduce_sum(m, axis=1)
			l = tf.reduce_sum(l, axis=1)
			l = tf.reduce_sum(l, axis=1)
			cor.append(l)
	cor = autograph.stack(cor)
	cor = tf.reshape(cor, (256, 256, 5, 1))
	cor = tf.transpose(cor, [2, 0, 1, 3])
	return cor

def resblock(input, filters, name, reuse=None):
	with tf.variable_scope(name):
		r = tf.layers.conv2d(input, filters, 3, padding='SAME', activation=tf.nn.relu, name='conv1', reuse=reuse)
		r = tf.layers.conv2d(r, filters, 3, padding='SAME', activation=tf.nn.relu, name='conv2', reuse=reuse)
		r = tf.layers.conv2d(r, filters, 3, padding='SAME', name='conv3', reuse=reuse)
		r = tf.add(input, r)
		return r