"""
This script detects faces in frames in videos, crops them and saves to disk.
Input:
<video_id>.jpg, <video_id>.jpg

Result:

images
    <video_id>
        <video_id>_<frame_id>_<crop_id>.jpg
labels
    <video_id>.json
"""
import argparse
import gc
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Union

import albumentations as albu
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from decord import VideoReader, cpu, gpu
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils import dlpack
from tqdm import tqdm

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.general import load_model
from utils.general import split_array
from utils.nms.py_cpu_nms import py_cpu_nms


@contextmanager
def video_reader(*args, **kwds):
    # Code to acquire resource, e.g.:
    resource = VideoReader(*args, **kwds)
    try:
        yield resource
    finally:
        del resource


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
    arg(
        "--network",
        default="resnet50",
        help="Backbone network mobile0.25 or resnet50",
        choices=["resnet50", "mobile0.25"],
    )
    arg("--cpu", action="store_true", default=False, help="Use cpu inference")
    arg("-c", "--confidence_threshold", default=0.7, type=float, help="confidence_threshold")
    arg("--top_k", default=5000, type=int, help="top_k")
    arg("--nms_threshold", default=0.4, type=float, help="nms_threshold")
    arg("--keep_top_k", default=750, type=int, help="keep_top_k")

    arg("-s", "--save_crops", action="store_true", default=False, help="If we want to store crops.")
    arg("-b", "--save_boxes", action="store_true", default=False, help="If we want to store bounding boxes.")
    arg("--origin_size", default=True, type=str, help="Whether use origin image size to evaluate")
    arg("--fp16", action="store_true", help="Whether use fp16")
    arg("-n", "--num_videos", type=int, help="Number of videos to use")

    arg("-v", "--video_decoder", type=str, help="Where to decode videos.", choices=["cpu", "gpu"], default="cpu")
    arg("-f", "--num_frames", type=int, help="Number of frames to extract", default=1)
    arg("--resize_coeff", nargs=2, help="min and max sizes for images", type=int, default=[1600, 2150])
    return parser.parse_args()


def prepare_image_cpu(raw_frame: Union[np.array, torch.tensor], transform: albu.Compose) -> torch.tensor:
    frame = raw_frame.astype(np.float32)
    new_frame = transform(image=frame)["image"]
    new_frame = tensor_from_rgb_image(new_frame)
    return new_frame.unsqueeze(0)


def prepare_image_gpu(raw_frame: Union[np.array, torch.tensor]) -> torch.tensor:
    new_frame = raw_frame.float()  # Not sure if I need this.
    new_frame -= torch.Tensor((104, 117, 123)).cuda()
    new_frame = new_frame.permute(2, 0, 1)
    return new_frame.unsqueeze(0)


def prepare_image(
    raw_frame: Union[np.array, torch.tensor], transform: albu.Compose, video_decoder: str = "cpu"
) -> torch.tensor:
    if video_decoder == "cpu":
        return prepare_image_cpu(raw_frame, transform)

    return prepare_image_gpu(raw_frame)


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
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    file_paths = sorted(args.input_path.rglob("*.mp4"))[: args.num_videos]

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

    if args.video_decoder == "cpu":
        decode_device = cpu(0)
    elif args.video_decoder == "gpu":
        decode_device = gpu(0)
    else:
        raise NotImplementedError(f"Only CPU and GPU devices are supported by decard, but got {args.video_decoder}")

    transform = albu.Compose([albu.Normalize(p=1, mean=(104, 117, 123), std=(1.0, 1.0, 1.0), max_pixel_value=1)], p=1)

    with torch.no_grad():
        for video_path in tqdm(file_paths):
            labels = []
            video_id = video_path.stem

            with video_reader(str(video_path), ctx=decode_device) as video:
                len_video = len(video)

                if args.num_frames is None or args.num_frames == 1:
                    frame_ids = list(range(args.num_frames))
                elif args.num_frames > 1:
                    if len_video < args.num_frames:
                        step = 1
                    else:
                        step = int(len_video / args.num_frames)

                    frame_ids = list(range(0, len_video, step))[: args.num_frames]
                else:
                    raise ValueError(f"Expect None or integer > 1 for args.num_frames, but got {args.num_frames}")

                frames = video.get_batch(frame_ids)

                if args.video_decoder == "cpu":
                    frames = frames.asnumpy()
                elif args.video_decoder == "gpu":
                    frames = dlpack.from_dlpack(frames.to_dlpack())

                if args.video_decoder == "gpu":
                    del video
                    torch.cuda.empty_cache()

                    gc.collect()

            num_frames = len(frames)

            image_height = frames.shape[1]
            image_width = frames.shape[2]

            scale1 = torch.Tensor(
                [
                    image_width,
                    image_height,
                    image_width,
                    image_height,
                    image_width,
                    image_height,
                    image_width,
                    image_height,
                    image_width,
                    image_height,
                ]
            )

            scale1 = scale1.to(device)

            scale = torch.Tensor([image_width, image_height, image_width, image_height])
            scale = scale.to(device)

            priorbox = PriorBox(cfg, image_size=(image_height, image_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data

            if args.resize_coeff is not None:
                target_size = min(args.resize_coeff)
                max_size = max(args.resize_coeff)

                image_height = frames.shape[1]
                image_width = frames.shape[2]

                image_size_min = min([image_width, image_height])
                image_size_max = max([image_width, image_height])

                resize = float(target_size) / float(image_size_min)
                if np.round(resize * image_size_max) > max_size:
                    resize = float(max_size) / float(image_size_max)
            else:
                resize = 1

            for pred_id in range(num_frames):
                frame = frames[pred_id]

                torched_image = prepare_image(frame, transform, args.video_decoder).to(device)

                if args.fp16:
                    torched_image = torched_image.half()

                loc, conf, land = net(torched_image)  # forward pass

                frame_id = frame_ids[pred_id]

                boxes = decode(loc.data[0], prior_data, cfg["variance"])

                boxes *= scale / resize

                boxes = boxes.cpu().numpy()
                scores = conf[0].data.cpu().numpy()[:, 1]

                landmarks = decode_landm(land.data[0], prior_data, cfg["variance"])

                landmarks *= scale1 / resize
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

                    labels += [
                        {
                            "frame_id": int(frame_id),
                            "crop_id": crop_id,
                            "bbox": bbox.tolist(),
                            "score": confidence[crop_id],
                            "landmarks": landmarks[crop_id].tolist(),
                        }
                    ]

                    if args.save_crops:
                        x_min, y_min, x_max, y_max = bbox

                        x_min = max(0, x_min)
                        y_min = max(0, y_min)

                        crop = frame[y_min:y_max, x_min:x_max]

                        target_folder = output_image_path / f"{video_id}"
                        target_folder.mkdir(exist_ok=True, parents=True)

                        crop_file_path = target_folder / f"{frame_id}_{crop_id}.jpg"

                        if crop_file_path.exists():
                            continue

                        cv2.imwrite(
                            str(crop_file_path),
                            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                        )

                if args.save_boxes:
                    result = {
                        "file_path": str(video_path),
                        "file_id": video_id,
                        "bboxes": labels,
                    }

                    with open(output_label_path / f"{video_id}.json", "w") as f:
                        json.dump(result, f, indent=2)


#

if __name__ == "__main__":
    main()
