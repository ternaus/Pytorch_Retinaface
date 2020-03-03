"""
This script detects faces in images, crop them and save to disk.
Input:
<file_path_1>.jpg, <file_path_2>.jpg
Result:
images
    <file_name>
        <file_name>_<crop_id>.jpg
labels
    <file_name>.json
"""
import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from iglovikov_helper_functions.utils.img_tools import load_rgb
from tqdm import tqdm

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.general import load_model
from utils.nms.py_cpu_nms import py_cpu_nms
from jpeg4py import JPEGRuntimeError


def split_array(array_length: int, num_splits: int, split_id: int) -> Tuple[int, int]:
    """Split array into parts.
    Args:
        array_length:
        num_splits:
        split_id:
    Returns: start and end indices of the
    """
    if not 0 <= split_id < num_splits:
        raise ValueError(f"gpu_id should be 0 <= {split_id} < {num_splits}")
    if array_length % num_splits == 0:
        step = int(array_length / num_splits)
    else:
        step = int(array_length / num_splits) + 1

    return split_id * step, min((split_id + 1) * step, array_length)


def get_args():
    parser = argparse.ArgumentParser(description="Retinaface")
    arg = parser.add_argument
    arg("-g", "--gpu_id", type=int, help="GPU_id")
    arg("--num_gpu", type=int, help="number of GPUs")
    arg(
        "-m",
        "--trained_model",
        default="./weights/Resnet50_Final.pth",
        type=str,
        help="Trained state_dict file path to open",
    )
    arg("-i", "--input_path", type=Path, help="Path where images are stored", required=True)
    arg(
        "-o",
        "--output_path",
        type=Path,
        help="Path where results will be saved: " "images folder for images and" "labels folder for bounding boxes",
        required=True,
    )
    arg("--network", default="resnet50", help="Backbone network mobile0.25 or resnet50")
    arg("--cpu", action="store_true", default=False, help="Use cpu inference")
    arg("-c", "--confidence_threshold", default=0.7, type=float, help="confidence_threshold")
    arg("--top_k", default=5000, type=int, help="top_k")
    arg("--nms_threshold", default=0.4, type=float, help="nms_threshold")
    arg("--keep_top_k", default=750, type=int, help="keep_top_k")

    arg("-j", "--num_jobs", type=int, help="Number of CPU threads", default=64)
    arg("-s", "--save_crops", action="store_true", default=False, help="If we want to store crops.")
    arg("-b", "--save_boxes", action="store_true", default=False, help="If we want to store bounding boxes.")
    arg("--origin_size", default=True, type=str, help="Whether use origin image size to evaluate")
    arg("--fp16", action="store_true", help="Whether use fp16")
    return parser.parse_args()


def prepare_image(image_path: (str, Path), origin_size, device):
    try:
        raw_image = load_rgb(image_path, lib="jpeg4py")
    except JPEGRuntimeError:
        raw_image = load_rgb(image_path, lib="cv2")

    img = raw_image.copy().astype(np.float32)

    # testing scale
    target_size = 1600
    max_size = 2150
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    if origin_size:
        resize = 1

    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    im_height, im_width = img.shape[:2]
    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    return img, scale, raw_image, resize


def main():
    args = get_args()
    torch.set_grad_enabled(False)

    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    else:
        raise NotImplementedError(f"Only mobile0.25 and resnet50 are suppoted.")

    # net and model
    net = RetinaFace(cfg=cfg, phase="test")
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    if args.fp16:
        net = net.half()

    print("Finished loading model!")
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    file_paths = sorted(args.input_path.rglob("*.jpg"))

    if args.num_gpu is not None:
        start, end = split_array(len(file_paths), args.num_gpu, args.gpu_id)
        file_paths = file_paths[start:end]

    output_path = args.output_path

    if args.save_boxes:
        output_label_path = output_path / "labels"
        output_label_path.mkdir(exist_ok=True, parents=True)

    if args.save_crops:
        output_image_path = output_path / "images"
        output_image_path.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        # testing begin
        for i, image_path in enumerate(tqdm(file_paths)):
            labels = []

            file_id = image_path.stem

            torched_image, scale, raw_image, resize = prepare_image(image_path, args.origin_size, device)

            im_height, im_width = torched_image.shape[2:]

            if args.fp16:
                torched_image = torched_image.half()
            loc, conf, landmarks = net(torched_image)  # forward pass

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data

            boxes = decode(loc.data.squeeze(0), prior_data, cfg["variance"])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

            landmarks = decode_landm(landmarks.data.squeeze(0), prior_data, cfg["variance"])
            scale1 = torch.Tensor(
                [
                    im_width,
                    im_height,
                    im_width,
                    im_height,
                    im_width,
                    im_height,
                    im_width,
                    im_height,
                    im_width,
                    im_height,
                ]
            )
            scale1 = scale1.to(device)
            landmarks = landmarks * scale1 / resize
            landmarks = landmarks.cpu().numpy()

            # ignore low scores
            valid_index = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[valid_index]
            landmarks = landmarks[valid_index]
            scores = scores[valid_index]

            # keep top-K before NMS
            order = scores.argsort()[::-1]
            # order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landmarks = landmarks[order]
            scores = scores[order]

            # do NMS
            detection = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(detection, args.nms_threshold)
            # keep = nms(detection, args.nms_threshold,force_cpu=args.cpu)

            # x_min, y_min, x_max, y_max, score
            detection = detection[keep, :]
            landmarks = landmarks[keep].astype(int)

            if detection.shape[0] == 0:
                continue

            bboxes = detection[:, :4].astype(int)
            confidence = detection[:, 4].astype(np.float64)

            for crop_id in range(len(detection)):

                bbox = bboxes[crop_id]

                labels += [{"crop_id": crop_id, "bbox": bbox.tolist(), "score": confidence[crop_id]}]

                if args.save_crops:
                    x_min, y_min, x_max, y_max = bbox

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)

                    crop = raw_image[y_min:y_max, x_min:x_max]

                    target_folder = output_image_path / f"{file_id}"
                    target_folder.mkdir(exist_ok=True, parents=True)

                    crop_file_path = target_folder / f"{file_id}_{crop_id}.jpg"

                    if crop_file_path.exists():
                        continue

                    cv2.imwrite(
                        str(crop_file_path),
                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                    )

            if args.save_boxes:
                result = {
                    "file_path": str(image_path),
                    "file_id": file_id,
                    "bboxes": labels,
                    "landmarks": landmarks[crop_id].tolist(),
                }

                with open(output_label_path / f"{file_id}.json", "w") as f:
                    json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
