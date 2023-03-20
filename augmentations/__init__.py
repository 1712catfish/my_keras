from affine import rand_affine_transform
from cutmix_and_mixup import cutmix, mixup, cut_mix_and_mix_up
from elastic_transform import elastic_transform


def get_augment(augment_name, **kwargs):
    if isinstance(augment_name, (list, tuple)):
        return [get_augment(name, **kwargs) for name in augment_name]

    if augment_name == 'rand_affine_transform':
        return rand_affine_transform
    elif augment_name == 'cutmix':
        return cutmix
    elif augment_name == 'mixup':
        return mixup
    elif augment_name == 'cut_mix_and_mix_up':
        return cut_mix_and_mix_up
    elif augment_name == 'elastic_transform':
        return elastic_transform
    else:
        raise ValueError(f'Unknown augment name: {augment_name}')
