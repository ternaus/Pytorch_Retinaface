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

import albumentations as albu
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from iglovikov_helper_functions.utils.img_tools import load_rgb
from jpeg4py import JPEGRuntimeError
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms
from tqdm import tqdm

from data import cfg_mnet, cfg_re50
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
    arg("--origin_size", default=True, type=str, help="Whether use origin image size to evaluate")
    arg("--fp16", action="store_true", help="Whether use fp16")
    arg(
        "--batch_size",
        type=int,
        help="Size of the batch size. Use non 1 value only if you are sure that" "all images are of the same size.",
        default=1,
    )
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(self, file_paths, origin_size, transform):
        self.file_paths = file_paths
        self.transform = transform

        self.origin_size = origin_size

    def __len__(self) -> int:

        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]

        try:
            raw_image = load_rgb(image_path, lib="jpeg4py")
        except JPEGRuntimeError:
            raw_image = load_rgb(image_path, lib="cv2")

        image = raw_image.astype(np.float32)

        if self.origin_size:
            resize = 1
        else:
            # testing scale
            target_size = 1600
            max_size = 2150
            im_shape = image.shape
            image_size_min = np.min(im_shape[0:2])
            image_size_max = np.max(im_shape[0:2])
            resize = float(target_size) / float(image_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(resize * image_size_max) > max_size:
                resize = float(max_size) / float(image_size_max)

            image = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        image = self.transform(image=image)["image"]

        return {
            "torched_image": tensor_from_rgb_image(image),
            "resize": resize,
            "raw_image": raw_image,
            "image_path": str(image_path),
        }


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

    transform = albu.Compose([albu.Normalize(p=1, mean=(104, 117, 123), std=(1.0, 1.0, 1.0), max_pixel_value=1)], p=1)

    test_loader = DataLoader(
        InferenceDataset(file_paths, args.origin_size, transform=transform),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    with torch.no_grad():
        for raw_input in tqdm(test_loader):
            torched_images = raw_input["torched_image"]

            if args.fp16:
                torched_images = torched_images.half()

            resizes = raw_input["resize"]
            image_paths = Path(raw_input["image_path"])
            raw_images = raw_input["raw_image"]

            labels = []

            if (
                args.batch_size == 1
                and args.save_boxes
                and (output_label_path / f"{Path(image_paths[0]).stem}.json").exists()
            ):
                continue

            loc, conf, land = net(torched_images.to(device))  # forward pass

            batch_size = torched_images.shape[0]

            image_height, image_width = torched_images.shape[2:]

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

            for batch_id in range(batch_size):
                image_path = image_paths[batch_id]
                file_id = Path(image_path).stem
                raw_image = raw_images[batch_id]

                resize = resizes[batch_id].float()

                boxes = decode(loc.data[batch_id], prior_data, cfg["variance"])

                boxes *= scale / resize
                scores = conf[batch_id][:, 1]

                landmarks = decode_landm(land.data[batch_id], prior_data, cfg["variance"])
                landmarks *= scale1 / resize

                # ignore low scores
                valid_index = torch.where(scores > args.confidence_threshold)[0]
                boxes = boxes[valid_index]
                landmarks = landmarks[valid_index]
                scores = scores[valid_index]

                order = scores.argsort(descending=True)

                boxes = boxes[order]
                landmarks = landmarks[order]
                scores = scores[order]

                # do NMS
                keep = nms(boxes, scores, args.nms_threshold)
                boxes = boxes[keep, :].int()

                landmarks = landmarks[keep].int()

                if boxes.shape[0] == 0:
                    continue

                scores = scores[keep].cpu().numpy().astype(np.float64)

                for crop_id, bbox in enumerate(boxes):

                    bbox = bbox.cpu().numpy()

                    labels += [
                        {
                            "crop_id": crop_id,
                            "bbox": bbox.tolist(),
                            "score": scores[crop_id],
                            "landmarks": landmarks[crop_id].tolist(),
                        }
                    ]

                    if args.save_crops:
                        x_min, y_min, x_max, y_max = bbox

                        x_min = max(0, x_min)
                        y_min = max(0, y_min)

                        crop = raw_image[y_min:y_max, x_min:x_max].cpu().numpy()

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
                        "file_path": image_path,
                        "file_id": file_id,
                        "bboxes": labels,
                    }

                    with open(output_label_path / f"{file_id}.json", "w") as f:
                        json.dump(result, f, indent=2)


#

if __name__ == "__main__":
    main()
