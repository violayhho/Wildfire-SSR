import os
import time

from models.VisionLanguageSSR import VisionLanguageSSR


BACKBONE_MAP = {
    'Small': 'dinov3_vits16',
    'Base': 'dinov3_vitb16',
    'Large': 'dinov3_vitl16',
    '7B': 'dinov3_vit7b16',
}


def build_write_logfile(model_tag, resume):
    if resume is not None:
        stamp = resume.split('/')[-2].split('_')[-1]
    else:
        stamp = str(time.time())

    def write_logfile(message_to_print):
        print(message_to_print)
        os.makedirs('log_files', exist_ok=True)
        log_file = f'log_files/{model_tag}_output_{stamp}.txt'
        with open(log_file, 'a') as of:
            of.write(message_to_print + '\n')
    return write_logfile, stamp


def build_model_tag(args):
    return f"VisionLanguageSSR_{args.geo_embed_type}{'_NoSat' if args.ablated_vision else ''}_{args.model_size}"


def get_model(args, freeze_backbone=True):
    if args.model_size not in BACKBONE_MAP:
        raise NotImplementedError(f"model_size '{args.model_size}' is not supported.")
    backbone_name = BACKBONE_MAP[args.model_size]
    geo_embed_type = None if args.geo_embed_type == 'none' else args.geo_embed_type

    print(f"Initializing VisionLanguageSSR with backbone={backbone_name}, "
          f"geo_embed_type={geo_embed_type}, ablated_vision={args.ablated_vision}")
    return VisionLanguageSSR(
        output_building=args.num_loc_classes,
        output_damage=args.num_classes,
        backbone_name=backbone_name,
        backbone_weights=args.backbone_weights,
        geo_embed_type=geo_embed_type,
        ablated_vision=args.ablated_vision,
        freeze_backbone=freeze_backbone,
    )
