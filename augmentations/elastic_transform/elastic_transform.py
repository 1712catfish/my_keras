import math as m
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A numpy array of shape shape with the generated noise.
    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = tf.meshgrid(
        tf.keras.backend.arange(0, res[0], delta[0]),
        tf.keras.backend.arange(0, res[1], delta[1])
    )
    grid = tf.transpose(grid, perm=(1, 2, 0)) % 1

    # Gradients
    angles = 2 * tf.constant(m.pi) * tf.random.uniform((res[0] + 1, res[1] + 1))
    gradients = tf.stack((tf.math.cos(angles), tf.math.sin(angles)), axis=-1)
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = tf.repeat(gradients, repeats=d[0], axis=0)
    gradients = tf.repeat(gradients, repeats=d[1], axis=1)

    # gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]
    # Ramps
    n00 = tf.math.reduce_sum(tf.stack((grid[:, :, 0], grid[:, :, 1]), axis=-1) * g00, axis=2)
    n10 = tf.math.reduce_sum(tf.stack((grid[:, :, 0] - 1, grid[:, :, 1]), axis=-1) * g10, axis=2)
    n01 = tf.math.reduce_sum(tf.stack((grid[:, :, 0], grid[:, :, 1] - 1), axis=-1) * g01, axis=2)
    n11 = tf.math.reduce_sum(tf.stack((grid[:, :, 0] - 1, grid[:, :, 1] - 1), axis=-1) * g11, axis=2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return tf.math.sqrt(2.0) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(
        shape, res, octaves=1, persistence=0.5,
        lacunarity=2, tileable=(False, False),
        interpolant=interpolant
):
    """Generate a 2D numpy array of fractal noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A numpy array of fractal noise and of shape shape generated by
        combining several octaves of perlin noise.
    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """
    noise = tf.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency * res[0], frequency * res[1]), tileable, interpolant
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise


def elastic_transform(image, alpha=1.0, sigma=1.0, random_state=None):
    shape = image.shape

    x = tf.range(shape[0], dtype="float32")  # x = 0 0 0 0 1 1 1 1 ....
    y = tf.range(shape[1], dtype="float32")  # y = 0 1 2 3 0 1 2 3 ....
    x, y = tf.meshgrid(x, y)

    x = x + alpha * tf.cast(generate_fractal_noise_2d(shape[:2], (8, 8), 5), 'float32')
    y = y + alpha * tf.cast(generate_fractal_noise_2d(shape[:2], (8, 8), 5), 'float32')

    x = tf.reshape(x, shape[0] * shape[1])
    x = K.clip(x, 0, shape[0] - 1)

    y = tf.reshape(y, shape[0] * shape[1])
    y = K.clip(y, 0, shape[1] - 1)

    indices = tf.stack([y, x], axis=1)
    indices = tf.cast(indices, dtype='int32')

    d = tf.gather_nd(image, indices)

    return tf.reshape(d, shape)


