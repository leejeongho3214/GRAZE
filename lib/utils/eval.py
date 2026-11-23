import torch

from lib.utils.frame import get_valid_mask


def get_valid_mask_bunch(is_lhand, is_rhand, max_nframes, duration):
    valid_mask_bunch_lhand = []
    valid_mask_bunch_rhand = []
    valid_mask_bunch_obj = []
    for idx in range(len(duration)):
        valid_mask_lhand, valid_mask_rhand, valid_mask_obj \
            = get_valid_mask(is_lhand[idx], is_rhand[idx], max_nframes, duration[idx])
        valid_mask_lhand = torch.BoolTensor(valid_mask_lhand).unsqueeze(0).cuda()
        valid_mask_rhand = torch.BoolTensor(valid_mask_rhand).unsqueeze(0).cuda()
        valid_mask_obj = torch.BoolTensor(valid_mask_obj).unsqueeze(0).cuda()
        valid_mask_bunch_lhand.append(valid_mask_lhand)
        valid_mask_bunch_rhand.append(valid_mask_rhand)
        valid_mask_bunch_obj.append(valid_mask_obj)
    valid_mask_bunch_lhand = torch.cat(valid_mask_bunch_lhand)
    valid_mask_bunch_rhand = torch.cat(valid_mask_bunch_rhand)
    valid_mask_bunch_obj = torch.cat(valid_mask_bunch_obj)
    return valid_mask_bunch_lhand, valid_mask_bunch_rhand, valid_mask_bunch_obj
