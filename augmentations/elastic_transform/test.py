# path = "pochita.png"
# image_bytes = tf.io.read_file(path)
# image = tf.image.decode_image(image_bytes, channels=3)
# image = tf.cast(image, tf.float32) / 255.0
# image = tf.image.resize(image, (512, 512))
#
# # elastic_transform(image, alpha=991.0, sigma=8.0)
#
# with tf.device('/gpu:2'):
#     plt.imshow(elastic_transform(image, alpha=10.0, sigma=8.0))