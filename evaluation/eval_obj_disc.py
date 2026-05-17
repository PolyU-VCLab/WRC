"""
Evaluate Object Discovery Performance with WRC. Code adapted from LOST: https://github.com/valeoai/LOST and TokenCut: https://github.com/YangtaoWANG95/TokenCut

Sample Usage:
python evaluation/eval_obj_disc.py --dataroot /YOUR/PATH/TO/DATAROOT/ --dataset VOC07 --set trainval --model_type dino_vits16 --wrc_path pretrained/wrc_dinov2_vits14.pth --imsize 224 --tau 0.2
python evaluation/eval_obj_disc.py --dataroot /YOUR/PATH/TO/DATAROOT/ --dataset VOC12 --set trainval --model_type dino_vits16 --wrc_path pretrained/wrc_dinov2_vits14.pth --imsize 224 --tau 0.2
python evaluation/eval_obj_disc.py --dataroot /YOUR/PATH/TO/DATAROOT/ --dataset COCO20k --set train --model_type dino_vits16 --wrc_path pretrained/wrc_dinov2_vits14.pth --imsize 224 --tau 0.3
"""
import os
import argparse
import random
import datetime
from datetime import datetime as dt
import time
import sys

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as pth_transforms

from wrc_extractor import ViTWRCExtractor, infer_settings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'TokenCut'))
from datasets import ImageDataset, Dataset, bbox_iou
from visualizations import visualize_img, visualize_eigvec, visualize_predictions, visualize_predictions_gt 
from object_discovery import ncut 


def main(args):
    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path, args.resize)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard, resize=args.imsize, dataroot=args.dataroot)

    # -------------------------------------------------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTWRCExtractor(args.model_type, args.wrc_path, args.channel, args.patch, args.stride, args.layer, args.facet, device=device)
    extractor.eval()

    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    exp_name = f"WRC-{args.model_type}-patch{args.patch}_{args.which_features}"

    print(f"Running WRC object discovery on the dataset {dataset.name} (exp: {exp_name})")

    # Visualization 
    if args.visualize:
        vis_folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)
        
    if args.save_feat_dir is not None : 
        os.mkdir(args.save_feat_dir)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))
    
    start_time = time.time() 
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):
        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]
        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])
        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # # Move to gpu
        img = img.cuda(non_blocking=True)
        # Size for transformers
        
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        num_patches = 1 + (args.imsize - args.patch_size) // args.stride
        num_patches = 2 * num_patches
        rescale_factor = args.imsize / num_patches

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue
                if args.dataset in ["VOC07", "VOC12"]:
                    orig_h = int(inp[1]['annotation']['size']['height'])
                    orig_w = int(inp[1]['annotation']['size']['width'])
                else:
                    orig_h = inp[2][1]
                    orig_w = inp[2][0]
                new_h = img.shape[1]
                new_w = img.shape[2]
                gt_bbxs = gt_bbxs * np.array([new_w/orig_w, new_h/orig_h, new_w/orig_w, new_h/orig_h])
                gt_bbxs = gt_bbxs.astype(np.int32)

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            if "vit"  in args.arch:
                # Forward pass in the model
                feats = extractor.extract_descriptors(img.unsqueeze(0))
                scales = [rescale_factor,rescale_factor]
            else:
                raise ValueError("Unknown model.")

        # ------------ Apply TokenCut graph cut on WRC features -------------------
        pred, objects, foreground, seed, bins, eigenvector = ncut(feats, [num_patches, num_patches], scales, init_image_size, args.tau, args.eps, im_name=im_name, no_binary_graph=args.no_binary_graph, has_cls=False)

        if args.visualize == "pred" and args.no_evaluation:
            image = dataset.load_image(im_name)
            visualize_predictions(image, pred, vis_folder, im_name)
        if args.visualize == "attn" and args.no_evaluation:
            visualize_eigvec(eigenvector, vis_folder, im_name, [w_featmap, h_featmap], scales)
        if args.visualize == "all" and args.no_evaluation:
            image = dataset.load_image(im_name)
            visualize_predictions(image, pred, vis_folder, im_name)
            visualize_eigvec(eigenvector, vis_folder, im_name, [w_featmap, h_featmap], scales)
                        
        # ------------ Visualizations -------------------------------------------
        # Save the prediction
        preds_dict[im_name] = pred

        # Evaluation
        if args.no_evaluation:
            continue

        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))
        
        if torch.any(ious >= 0.5):
            corloc[im_id] = 1
        vis_folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)
        image = dataset.load_image(im_name)
        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")

    end_time = time.time()
    print(f'Time cost: {str(datetime.timedelta(milliseconds=int((end_time - start_time)*1000)))}')

    # Evaluate
    if not args.no_evaluation:
        # print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
        final_corloc = 100 * np.sum(corloc) / cnt
        print(f"corloc: {final_corloc:.2f} ({int(np.sum(corloc))}/{cnt})")
        
        ### ADDED LOGGING CODE ###
        log_dir = os.path.join(args.output_dir, "obj_log_imsize")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"results_log_{timestamp}.txt")
        
        # Determine upsampler name to log (handle None case)
        u_name = 'wrc'
        
        with open(log_file, "a") as f:
            f.write(f"imsize: {args.imsize}, upsampler: {u_name}, corloc: {final_corloc:.2f}\n")


def parse_args():
    parser = argparse.ArgumentParser('Unsupervised Object Discovery Evaluation with TokenCut')
    ### BACKBONE ###
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                        Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                        vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                        options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--channel', default=None, type=int, help='backbone output channels (default: inferred from --model_type)')
    parser.add_argument('--patch', default=None, type=int, help='backbone patch size (default: inferred from --model_type)')
    parser.add_argument('--stride', default=None, type=int, help='stride of first convolution layer. small stride -> higher resolution. (default: equal to --patch)')
    parser.add_argument('--layer', default=None, type=int, help='layer to create descriptors from. (default: last layer)')
    ### WRC ###
    parser.add_argument('--wrc_path', required=True, type=str, help='path of pretrained WRC model to use')
    ### TASK ###
    parser.add_argument('--dataroot', default='path/to/dataroot')
    parser.add_argument('--imsize', default=224, type=int, help='image resize size')
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "moco_vit_small",
            "moco_vit_base",
            "mae_vit_base",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--save-feat-dir",
        type=str,
        default=None,
        help="if save-feat-dir is not None, only computing features and save it into save-feat-dir",
    )
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )
    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )
    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")
    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["attn", "pred", "all", None],
        default=None,
        help="Select the different type of visualizations.",
    )
    # TokenCut parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v"],
        help="Which features to use",
    )
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )
    parser.add_argument("--resize", type=int, default=None, help="Resize input image to fix size")
    parser.add_argument("--tau", type=float, default=0.2, help="Tau for seperating the Graph.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Eps for defining the Graph.")
    parser.add_argument("--no-binary-graph", action="store_true", default=False, help="Generate a binary graph where edge of the Graph will binary. Or using similarity score as edge weight.")
    ###
    args = parser.parse_args()
    infer_settings(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
