from src.swin_transformer.data.build import build_loader as _build_loader
from src.swin_transformer.data.data_simmim_ft import build_loader_finetune
from src.swin_transformer.data.data_simmim_pt import build_loader_simmim


def build_loader(config, args, simmim=False, is_pretrain=False):
    if not simmim:
        return _build_loader(config, args)
    if is_pretrain:
        return build_loader_simmim(config)
    else:
        return build_loader_finetune(config)
