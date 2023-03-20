import tensorflow as tf
from deprecation import deprecated


@deprecated
def cosine_decay_restarts(
        lr=1e-2, first_decay_steps=None,
        total_steps=1000,
        cycles=3, m_mul=1.0,
        t_mul=1.0,
        min_lr=1e-12,
):
    return tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=first_decay_steps if first_decay_steps else total_steps // ((1-t_mul**cycles)/(1-t_mul+1e-12)),
        t_mul=t_mul,
        m_mul=m_mul,
        alpha=min_lr / lr,
    )
