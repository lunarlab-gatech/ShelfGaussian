"""Generate the Talk2DINO text embeddings of the 18 Occ3D classes.

Each class is described by several prompts. The prompts are encoded by CLIP
ViT-B/16, projected into the DINOv2 ViT-B/14 space by Talk2DINO
(https://github.com/lorebianchi98/Talk2DINO, Apache-2.0 License), averaged and
L2-normalized. The output is a (768, 18) tensor, used as `text_protos` in the
configs.

Run it in an environment where Talk2DINO is installed:
    git clone https://github.com/lorebianchi98/Talk2DINO.git
    # download weights/vitb_mlp_infonce.pth following the Talk2DINO README
    python tools/data_generation/generate_text_embeddings.py \
        --talk2dino-root /path/to/Talk2DINO \
        --out ckpts/text_proto_embeds_talk2dino_customized.pth
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

# Ordered as the Occ3D-nuScenes classes (0-17).
CLASS_PROMPTS = [
    # 0: others
    [
        'an animal', 'a dog', 'a cat', 'a person using a wheelchair',
        'a stroller', 'a scooter', 'road debris', 'a pushable cart',
        'a bicycle rack', 'an ambulance', 'a police car', 'noise',
        'background clutter', 'ego vehicle'
    ],
    # 1: barrier
    ['a road barrier', 'a traffic barrier', 'a guard rail',
     'a concrete barrier'],
    # 2: bicycle
    ['a bicycle', 'a bike', 'a mountain bike', 'a road bike'],
    # 3: bus
    ['a bus', 'a city bus', 'a bendy bus', 'a rigid bus', 'a school bus'],
    # 4: car
    ['a car', 'a sedan', 'a sports car', 'an SUV', 'a vehicle on the road'],
    # 5: construction_vehicle
    [
        'a construction vehicle', 'a bulldozer', 'an excavator',
        'a crane truck', 'a dump truck'
    ],
    # 6: motorcycle
    ['a motorcycle', 'a motorbike', 'a scooter', 'a two-wheeled vehicle'],
    # 7: pedestrian
    [
        'a pedestrian', 'a person walking', 'a man crossing the street',
        'a woman on the sidewalk', 'a child walking', 'a construction worker',
        'a police officer on foot'
    ],
    # 8: traffic_cone
    ['a traffic cone', 'a road cone', 'an orange cone', 'a safety cone'],
    # 9: trailer
    ['a trailer', 'a cargo trailer', 'a container trailer',
     'a truck trailer'],
    # 10: truck
    [
        'a truck', 'a pickup truck', 'a delivery truck',
        'a heavy goods vehicle', 'a lorry'
    ],
    # 11: driveable_surface
    ['a road', 'an asphalt surface', 'a paved street', 'a driveable road'],
    # 12: other_flat
    [
        'a parking lot', 'a courtyard ground', 'a flat open surface',
        'a paved plaza'
    ],
    # 13: sidewalk
    ['a sidewalk', 'a pavement', 'a pedestrian walkway', 'a footpath'],
    # 14: terrain
    ['grass', 'dirt ground', 'muddy terrain', 'natural soil surface'],
    # 15: manmade
    ['a building', 'a wall', 'a fence', 'a house', 'an artificial structure'],
    # 16: vegetation
    ['a tree', 'a bush', 'green plants', 'a forest', 'vegetation'],
    # 17: free
    ['the sky', 'open air', 'background', 'empty space'],
]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--talk2dino-root', required=True)
    parser.add_argument(
        '--out', default='ckpts/text_proto_embeds_talk2dino_customized.pth')
    parser.add_argument('--proj-name', default='vitb_mlp_infonce')
    args = parser.parse_args()

    sys.path.insert(0, args.talk2dino_root)
    import clip
    from src.model import ProjectionLayer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    talk2dino = ProjectionLayer.from_config(
        os.path.join(args.talk2dino_root, 'configs', f'{args.proj_name}.yaml'))
    talk2dino.load_state_dict(
        torch.load(
            os.path.join(args.talk2dino_root, 'weights',
                         f'{args.proj_name}.pth'),
            map_location=device))
    talk2dino.to(device).eval()
    clip_model, _ = clip.load('ViT-B/16', device=device, jit=False)
    clip_model.eval()

    embeds = []
    for prompts in CLASS_PROMPTS:
        tokens = clip.tokenize(prompts).to(device)
        text_feats = clip_model.encode_text(tokens)  # (num_prompts, 512)
        projected = talk2dino.project_clip_txt(text_feats)  # (num_prompts, 768)
        embeds.append(F.normalize(projected.mean(dim=0), dim=-1))
    embeds = torch.stack(embeds, dim=0).mT.cpu()  # (768, 18)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(embeds, args.out)
    print(f'Saved {tuple(embeds.shape)} embeddings to {args.out}')


if __name__ == '__main__':
    main()
