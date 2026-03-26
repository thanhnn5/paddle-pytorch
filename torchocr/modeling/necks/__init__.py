__all__ = ['build_neck']


def build_neck(config):
    from .db_fpn import DBFPN, RSEFPN, LKPAN
    from .rnn import SequenceEncoder
    from .hybrid_encoder_neck import HybridEncoderNeck
    support_dict = [
        'SequenceEncoder', 'DBFPN', 'RSEFPN', 'LKPAN', 'HybridEncoderNeck'
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('neck only support {}'.format(
        support_dict))

    module_class = eval(module_name)(**config)
    return module_class
