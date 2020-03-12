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
import json
from pathlib import Path
from typing import Union, List, Tuple, Optional

import albumentations as albu
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from decord import VideoReader, cpu
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms
from tqdm import tqdm

from data import cfg_mnet_test, cfg_re50_test
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.general import load_model, split_array


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

    arg("-j", "--num_workers", type=int, help="Number of CPU threads", default=64)
    arg("-s", "--save_crops", action="store_true", default=False, help="If we want to store crops.")
    arg("-b", "--save_boxes", action="store_true", default=False, help="If we want to store bounding boxes.")
    arg("--fp16", action="store_true", help="Whether use fp16")
    arg("-n", "--num_videos", type=int, help="Number of videos to use")
    arg(
        "--batch_size",
        type=int,
        help="Size of the batch size. Use non 1 value only if you are sure that" "all images are of the same size.",
        default=1,
    )
    arg("-v", "--video_decoder", type=str, help="Where to decode videos.", choices=["cpu", "gpu"], default="cpu")
    arg("-f", "--num_frames", type=int, help="Number of frames to extract")
    arg("--resize_coeff", nargs=2, help="min and max sizes for images", type=int, default=[1600, 2150])
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(
        self,
        video_paths: List[Path],
        num_frames: Union[int, None],
        transform: albu.Compose,
        resize_coeff: Optional[tuple],
        output_label_path: Union[Path, None],
    ):
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.transform = transform
        self.resize_coeff = resize_coeff
        self.output_label_path = output_label_path

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> dict:
        video_path = self.video_paths[idx]

        video_id = video_path.stem

        if self.output_label_path is not None:
            output_json_path = self.output_label_path / f"{video_id}.json"

            if output_json_path.exists():
                return {}

        video = VideoReader(str(video_path), ctx=cpu(0))
        len_video = len(video)

        if self.num_frames is None:
            frame_ids = list(range(len_video))
        else:
            if len_video < self.num_frames:
                step = 1
            else:
                step = int(len_video / self.num_frames)

            frame_ids = list(range(0, len_video, step))[: self.num_frames]

        frames = video.get_batch(frame_ids).asnumpy()

        torched_frames, resize = self.prepare_frames(frames)

        result = {
            "torched_frames": torched_frames,
            "resize": resize,
            "video_path": str(video_path),
            "frame_ids": np.array(frame_ids),
            "frames": frames,
        }

        return result

    def prepare_frames(self, frames):
        if self.resize_coeff is not None:
            target_size = min(self.resize_coeff)
            max_size = max(self.resize_coeff)

            image_height = frames.shape[1]
            image_width = frames.shape[2]

            image_size_min = min([image_width, image_height])
            image_size_max = max([image_width, image_height])

            resize = float(target_size) / float(image_size_min)
            if np.round(resize * image_size_max) > max_size:
                resize = float(max_size) / float(image_size_max)
        else:
            resize = 1

        result = []

        for frame in frames:
            if self.resize_coeff is not None and resize != 1:
                frame = cv2.resize(frame, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

            new_frame = self.transform(image=frame)["image"]

            result += [tensor_from_rgb_image(new_frame)]

        if len(result) != 1:
            result = torch.stack(result)
        else:
            result = torch.unsqueeze(result[0], 0)

        return result, resize


def main():
    args = get_args()

    file_paths = sorted(args.input_path.rglob("*.mp4"))[: args.num_videos]

    parameters = {
        "network": args.network,
        "trained_model": args.trained_model,
        "if_cpu": args.cpu,
        "if_fp16": args.fp16,
        "file_paths": file_paths,
        "num_gpu": args.num_gpu,
        "gpu_id": args.gpu_id,
        "output_path": args.output_path,
        "if_save_boxes": args.save_boxes,
        "if_save_crops": args.save_crops,
        "num_frames": args.num_frames,
        "resize_coeff": args.resize_coeff,
        "confidence_threshold": args.confidence_threshold,
        "num_workers": args.num_workers,
        "nms_threshold": args.nms_threshold,
        "batch_size": args.batch_size,
    }

    process_video_files(**parameters)


def process_video_files(
    network: str,
    trained_model: str,
    if_cpu: bool,
    if_fp16: bool,
    file_paths: list,
    num_gpu: Optional[int],
    gpu_id: int,
    output_path: Path,
    if_save_boxes: bool,
    if_save_crops: bool,
    num_frames: int,
    resize_coeff: Optional[Tuple],
    confidence_threshold: float,
    num_workers: int,
    nms_threshold: float,
    batch_size: int,
) -> None:
    torch.set_grad_enabled(False)

    if network == "mobile0.25":
        cfg = cfg_mnet_test
    elif network == "resnet50":
        cfg = cfg_re50_test
    else:
        raise NotImplementedError(f"Only mobile0.25 and resnet50 are suppoted.")

    # net and model
    net = RetinaFace(cfg=cfg, phase="test")
    net = load_model(net, trained_model, if_cpu)
    net.eval()

    if if_fp16:
        net = net.half()

    device = torch.device("cpu" if if_cpu else "cuda")
    net.to(device)

    print("Finished loading model!")
    cudnn.benchmark = True

    transform = albu.Compose([albu.Normalize(p=1, mean=(104, 117, 123), std=(1.0, 1.0, 1.0), max_pixel_value=1)], p=1)

    if num_gpu is not None:
        start, end = split_array(len(file_paths), num_gpu, gpu_id)
        file_paths = file_paths[start:end]

    if if_save_crops:
        output_image_path = output_path / "images"
        output_image_path.mkdir(exist_ok=True, parents=True)

    if if_save_boxes:
        output_label_path: Path = output_path / "labels"
        output_label_path.mkdir(exist_ok=True, parents=True)

        test_loader = DataLoader(
            InferenceDataset(
                file_paths,
                num_frames,
                transform=transform,
                resize_coeff=resize_coeff,
                output_label_path=output_label_path,
            ),
            batch_size=1,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        test_loader = DataLoader(
            InferenceDataset(
                file_paths, num_frames, transform=transform, resize_coeff=resize_coeff, output_label_path=None,
            ),
            batch_size=1,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    with torch.no_grad():
        for raw_input in tqdm(test_loader):
            if not raw_input:
                continue
            torched_frames = raw_input["torched_frames"][0]

            if if_fp16:
                torched_frames = torched_frames.half()

            resize = raw_input["resize"][0]
            video_path = Path(raw_input["video_path"][0])
            frame_ids = raw_input["frame_ids"][0].numpy()
            frames = raw_input["frames"][0]

            num_frames = torched_frames.shape[0]

            video_id = video_path.stem

            labels: List[dict] = []

            image_height, image_width = torched_frames.shape[2:]

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

            for start_index in range(0, num_frames, batch_size):
                end_index = min(start_index + batch_size, num_frames)

                loc, conf, land = net(torched_frames[start_index:end_index].to(device))

                for pred_id in range(loc.shape[0]):
                    frame_id = frame_ids[start_index + pred_id]

                    boxes = decode(loc.data[pred_id], prior_data, cfg["variance"])

                    boxes *= scale / resize
                    scores = conf[pred_id][:, 1]

                    landmarks = decode_landm(land.data[pred_id], prior_data, cfg["variance"])
                    landmarks *= scale1 / resize

                    # ignore low scores
                    valid_index = torch.where(scores > confidence_threshold)[0]
                    boxes = boxes[valid_index]
                    landmarks = landmarks[valid_index]
                    scores = scores[valid_index]

                    order = scores.argsort(descending=True)

                    boxes = boxes[order]
                    landmarks = landmarks[order]
                    scores = scores[order]

                    # do NMS
                    keep = nms(boxes, scores, nms_threshold)
                    boxes = boxes[keep, :].int()

                    landmarks = landmarks[keep].int()

                    if boxes.shape[0] == 0:
                        continue

                    scores = scores[keep].cpu().numpy().astype(np.float64)

                    for crop_id, bbox in enumerate(boxes):
                        bbox = bbox.cpu().numpy()

                        labels += [
                            {
                                "frame_id": int(frame_id),
                                "crop_id": crop_id,
                                "bbox": bbox.tolist(),
                                "score": scores[crop_id],
                                "landmarks": landmarks[crop_id].tolist(),
                            }
                        ]

                        if if_save_crops:
                            x_min, y_min, x_max, y_max = bbox

                            x_min = max(0, x_min)
                            y_min = max(0, y_min)

                            crop = frames[pred_id][y_min:y_max, x_min:x_max]

                            target_folder = output_image_path / f"{video_id}"
                            target_folder.mkdir(exist_ok=True, parents=True)

                            crop_file_path = target_folder / f"{frame_id}_{crop_id}.jpg"

                            if crop_file_path.exists():
                                continue

                            cv2.imwrite(
                                str(crop_file_path),
                                cv2.cvtColor(crop.numpy(), cv2.COLOR_BGR2RGB),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                            )

                    if if_save_boxes:
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
