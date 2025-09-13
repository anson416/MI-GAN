"""
Create an ONNX-exportable MI-GAN pipeline that supports batch processing
with per-sample cropping via grid_sample. Unlike the global-union crop,
each sample gets its own crop while keeping the graph batch-friendly.

Behavior:
- For each image in the batch, we compute a square crop that covers its
  masked region with padding and min-size >= resolution.
- We use grid_sample to:
  1) extract the per-sample crop to R x R for the model,
  2) paste the predicted result back into the full-resolution image.

This avoids aspect-ratio distortion and unnecessary downscaling for some images
while keeping a single batched forward pass.

Usage (same style as original):
python -m scripts.create_onnx_pipeline_batch \
    --resolution 512 \
    --model-path ./models/migan_512_places2.pt \
    --images-dir ./examples/places2_512_object_batch/images \
    --masks-dir ./examples/places2_512_object_batch/masks \
    --output-dir ./exported_models/places2_512_onnx_batch \
    --device cpu \
    --invert-mask
"""

import argparse
import math
import numbers
import os
from glob import glob
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxsim import simplify
from PIL import Image
from tqdm import tqdm

from lib.model_zoo.migan_inference import Generator as MIGAN


# ---------------------------
# Helpers (same as original)
# ---------------------------
def read_mask(mask_path, invert=False):
    mask = Image.open(mask_path)
    mask = np.array(mask)
    if len(mask.shape) == 3:
        if mask.shape[2] == 4:
            _r, _g, _b, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 2:
            _l, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 3:
            _r, _g, _b = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_r, _r, _r])
    else:
        mask = np.dstack([mask, mask, mask])
    if invert:
        mask = 255 - mask
    mask[mask < 255] = 0
    return Image.fromarray(mask).convert("L")


# GaussianSmoothing (kept identical in spirit to original, ONNX-friendly)
class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.padding = (kernel_size - 1) // 2

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-((mgrid - mean) ** 2) / (2 * std**2))
            )

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1,2,3 dims supported.")

    def forward(self, input):
        input = F.pad(
            input,
            (self.padding, self.padding, self.padding, self.padding),
            mode="reflect",
        )
        return self.conv(
            input,
            weight=self.weight.to(input.dtype),
            groups=self.groups,
            padding=0,
        )


# -------------------------------------------------
# Per-sample crop pipeline with grid_sample (ONNX)
# -------------------------------------------------
class MIGANPipelineBatch(nn.Module):
    def __init__(self, model_path, resolution, padding=128, device="cpu"):
        super().__init__()
        self.model = MIGAN(resolution=resolution)
        # Load weights on CPU for safety; remove weights_only=True if unsupported in your PyTorch
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
        except TypeError:
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu")
            )
        self.model = self.model.to(device)
        self.model.eval()

        self.gaussian_blur = GaussianSmoothing(
            channels=1, kernel_size=5, sigma=1.0, dim=2
        ).to(device)

        self.resolution = int(resolution)  # R
        self.padding = int(padding)
        # buffers/constants
        self.register_buffer("inv_255", torch.tensor(1.0 / 255.0))

    @staticmethod
    def _bbox_from_mask(mask_uint8: torch.Tensor):
        """
        Compute per-sample bounding boxes from mask (B,1,H,W) uint8 where 255=known, 0=hole.
        Returns (x_min, x_max, y_min, y_max, any_hole) each as int64 tensors of shape (B,).
        """
        B, H, W = mask_uint8.size(0), mask_uint8.size(2), mask_uint8.size(3)
        device = mask_uint8.device
        # hole locations
        hole = mask_uint8[:, 0].to(torch.float32) < 255.0  # (B,H,W)
        any_hole = hole.view(B, -1).any(dim=1)  # (B,)

        # indices
        w_idx = torch.arange(W, dtype=torch.int64, device=device)  # (W,)
        h_idx = torch.arange(H, dtype=torch.int64, device=device)  # (H,)

        col_has = hole.any(dim=1)  # (B, W)
        row_has = hole.any(dim=2)  # (B, H)

        w_sentinel_max = torch.full((1,), W, dtype=torch.int64, device=device)
        w_sentinel_zero = torch.zeros((1,), dtype=torch.int64, device=device)
        h_sentinel_max = torch.full((1,), H, dtype=torch.int64, device=device)
        h_sentinel_zero = torch.zeros((1,), dtype=torch.int64, device=device)

        col_idx_b = w_idx.unsqueeze(0).expand(B, -1)  # (B,W)
        row_idx_b = h_idx.unsqueeze(0).expand(B, -1)  # (B,H)

        col_for_min = torch.where(col_has, col_idx_b, w_sentinel_max)
        col_for_max = torch.where(col_has, col_idx_b, w_sentinel_zero)
        row_for_min = torch.where(row_has, row_idx_b, h_sentinel_max)
        row_for_max = torch.where(row_has, row_idx_b, h_sentinel_zero)

        x_min = torch.min(col_for_min, dim=1).values  # (B,)
        x_max = torch.max(col_for_max, dim=1).values
        y_min = torch.min(row_for_min, dim=1).values
        y_max = torch.max(row_for_max, dim=1).values

        # fallback for samples with no holes: set a tiny 1x1 at (0,0)
        x_min = torch.where(any_hole, x_min, torch.zeros_like(x_min))
        y_min = torch.where(any_hole, y_min, torch.zeros_like(y_min))
        x_max = torch.where(any_hole, x_max, torch.ones_like(x_max))
        y_max = torch.where(any_hole, y_max, torch.ones_like(y_max))

        return x_min, x_max, y_min, y_max, any_hole

    def _square_padded_boxes(
        self, x_min, x_max, y_min, y_max, H, W, resolution, padding
    ):
        """
        From per-sample bboxes, compute a square crop for each sample:
        - crop_size = max(w, h) + 2*padding, then ensure >= resolution
        - center at (cx, cy)
        - clamp to image bounds, try to preserve crop_size if possible.
        Inputs: int64 tensors (B,), H/W ints.
        Returns: final x_min,x_max,y_min,y_max (B,) int64.
        """
        device = x_min.device
        # width/height of hole region
        w = (x_max - x_min).clamp(min=1)
        h = (y_max - y_min).clamp(min=1)

        # centers
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        # desired crop size
        desired = torch.maximum(w, h) + 2 * padding
        desired = torch.maximum(
            desired, torch.tensor(resolution, dtype=torch.int64, device=device)
        )

        # half size
        offset = desired // 2

        # initial bounds
        xi = cx - offset
        xa = cx + offset
        yi = cy - offset
        ya = cy + offset

        # clamp to image bounds
        xi = torch.clamp(xi, 0, W)
        xa = torch.clamp(xa, 0, W)
        yi = torch.clamp(yi, 0, H)
        ya = torch.clamp(ya, 0, H)

        # Extend to reach desired size when hitting borders (if possible)
        need_x = (desired - (xa - xi)).clamp(min=0)
        need_y = (desired - (ya - yi)).clamp(min=0)

        xi = torch.clamp(xi - need_x, 0, W)
        xa = torch.clamp(xa + need_x, 0, W)
        yi = torch.clamp(yi - need_y, 0, H)
        ya = torch.clamp(ya + need_y, 0, H)

        # Ensure at least 1x1
        xa = torch.maximum(xa, xi + 1)
        ya = torch.maximum(ya, yi + 1)

        return xi, xa, yi, ya  # (B,)

    def _build_crop_grid(self, boxes, H, W, R, device):
        """
        Build per-sample crop grids for grid_sample to extract R x R crops.

        boxes: (x_min, x_max, y_min, y_max) each (B,) int64
        returns grid: (B, R, R, 2) float32 in [-1,1], align_corners=False
        """
        x_min, x_max, y_min, y_max = boxes
        B = x_min.shape[0]

        # widths/heights
        w = (x_max - x_min).to(torch.float32)  # (B,)
        h = (y_max - y_min).to(torch.float32)  # (B,)

        # 1D normalized sampling positions along X and Y for the crop
        # Output pixel indices i=0..R-1 -> input pixel positions:
        # p_x(i) = x_min + (i + 0.5) * w / R
        # Then normalized: x_norm = 2 * (p_x + 0.5) / W - 1  (align_corners=False)
        i = (
            torch.arange(R, device=device, dtype=torch.float32) + 0.5
        ) / float(R)  # (R,)
        # Broadcast to (B,R)
        x_pos = x_min.to(torch.float32).unsqueeze(1) + i.unsqueeze(
            0
        ) * w.unsqueeze(1)
        y_pos = y_min.to(torch.float32).unsqueeze(1) + i.unsqueeze(
            0
        ) * h.unsqueeze(1)

        x_norm_1d = 2.0 * ((x_pos + 0.5) / W) - 1.0  # (B,R)
        y_norm_1d = 2.0 * ((y_pos + 0.5) / H) - 1.0  # (B,R)

        # mesh to (B,R,R)
        grid_x = x_norm_1d.unsqueeze(1).expand(B, R, R)  # broadcast along rows
        grid_y = y_norm_1d.unsqueeze(2).expand(B, R, R)  # broadcast along cols

        grid = torch.stack([grid_x, grid_y], dim=-1)  # (B,R,R,2)
        return grid

    def _build_paste_grid(self, boxes, H, W, device):
        """
        Build per-sample paste grids mapping full image (H,W) -> patch (R,R).
        Used to paste model outputs back to full res:
          grid[...,0] = 2*(X - x_min)/w - 1
          grid[...,1] = 2*(Y - y_min)/h - 1
        grid outside [-1,1] samples zeros (padding_mode='zeros').

        returns grid: (B, H, W, 2) float32 in [-1,1], align_corners=False
        """
        x_min, x_max, y_min, y_max = boxes
        B = x_min.shape[0]

        w = (x_max - x_min).to(torch.float32).clamp(min=1.0)  # (B,)
        h = (y_max - y_min).to(torch.float32).clamp(min=1.0)

        # output pixel centers (X+0.5, Y+0.5)
        xx = (
            torch.arange(W, device=device, dtype=torch.float32)
            .view(1, 1, W)
            .expand(B, H, W)
            + 0.5
        )
        yy = (
            torch.arange(H, device=device, dtype=torch.float32)
            .view(1, H, 1)
            .expand(B, H, W)
            + 0.5
        )

        x_min_f = x_min.to(torch.float32).view(B, 1, 1)
        y_min_f = y_min.to(torch.float32).view(B, 1, 1)
        w_f = w.view(B, 1, 1)
        h_f = h.view(B, 1, 1)

        # normalized coords in patch space (align_corners=False)
        x_norm = 2.0 * ((xx - x_min_f) / w_f) - 1.0
        y_norm = 2.0 * ((yy - y_min_f) / h_f) - 1.0

        grid = torch.stack([x_norm, y_norm], dim=-1)  # (B,H,W,2)
        return grid

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Batch forward with per-sample cropping
        image: (B, 3, H, W) uint8
        mask:  (B, 1, H, W) uint8 (255 known, 0 hole)
        Returns:
           composed batch: (B, 3, H, W) uint8
        """
        assert image.dim() == 4 and mask.dim() == 4, "Expect batched tensors"
        H, W = image.size(2), image.size(3)
        device = image.device

        # 1) Per-sample bounding boxes from mask
        x_min, x_max, y_min, y_max, _ = self._bbox_from_mask(mask)
        x_min, x_max, y_min, y_max = self._square_padded_boxes(
            x_min, x_max, y_min, y_max, H, W, self.resolution, self.padding
        )

        # 2) Build crop grids and extract R x R per-sample crops for image & mask
        grid_crop = self._build_crop_grid(
            (x_min, x_max, y_min, y_max), H, W, self.resolution, device
        )

        # image to float in [-1,1], mask to 0..1 for model input
        image_f = image.to(torch.float32) * (2.0 / 255.0) - 1.0
        mask_f = mask.to(torch.float32) / 255.0

        image_crop = F.grid_sample(
            image_f,
            grid_crop,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # (B,3,R,R)
        mask_crop = F.grid_sample(
            mask_f,
            grid_crop,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )  # (B,1,R,R)

        # 3) Model input and inference
        model_input = torch.cat(
            [mask_crop - 0.5, image_crop * mask_crop], dim=1
        )  # (B,4,R,R)
        model_out = self.model(model_input)  # (B,3,R,R), typically in [-1,1]

        # 4) Convert model output to 0..255 float for blending
        pred_crop = ((model_out * 0.5 + 0.5) * 255.0).clamp(
            0, 255
        )  # (B,3,R,R)

        # 5) Paste prediction back to full resolution using a per-sample paste grid
        grid_paste = self._build_paste_grid(
            (x_min, x_max, y_min, y_max), H, W, device
        )
        pred_full = F.grid_sample(
            pred_crop,
            grid_paste,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # (B,3,H,W)

        # 6) Smooth original full-size mask and blend
        mask_full_f = mask.to(torch.float32)
        mask_sm = F.max_pool2d(mask_full_f, kernel_size=3, stride=1, padding=1)
        mask_sm = self.gaussian_blur(mask_sm) * self.inv_255  # 0..1
        mask_sm_3 = mask_sm.repeat(1, 3, 1, 1)

        image_u8 = image.to(torch.float32)  # 0..255
        composed = image_u8 * mask_sm_3 + pred_full * (1.0 - mask_sm_3)
        composed = composed.clamp(0, 255).to(torch.uint8)
        return composed


# ---------------------------
# Argument parsing & main
# ---------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution", type=int, help="256 or 512", required=True
    )
    parser.add_argument(
        "--model-path", type=str, help="Saved .pt model path.", required=True
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Path to images directory.",
        required=True,
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        help="Path to masks directory.",
        required=True,
    )
    parser.add_argument(
        "--invert-mask",
        action="store_true",
        help="Invert mask polarity (use if your masks are 255=hole). After inversion, 255=known, 0=hole.",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory.", required=True
    )
    parser.add_argument("--device", type=str, help="Device.", default="cpu")
    return parser.parse_args()


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    pipeline = MIGANPipelineBatch(
        model_path=args.model_path, resolution=args.resolution, device=device
    )

    print("Exporting ONNX model (batch-capable, per-sample crops).")
    # Use batch dummy inputs to capture batch dynamic axis
    dummy_image = (
        torch.ones(2, 3, 256, 256, device="cpu", dtype=torch.uint8) * 255
    )
    dummy_mask = (
        torch.ones(2, 1, 256, 256, device="cpu", dtype=torch.uint8) * 255
    )
    # create some holes in dummy masks
    dummy_mask[0, :, 10:20, 10:20] = 0
    dummy_mask[1, :, 30:40, 40:50] = 0

    input_names = ["image", "mask"]
    output_names = ["result"]
    (args.output_dir / "models").mkdir(parents=True, exist_ok=True)
    onnx_export_path = args.output_dir / "models" / "migan_batch.onnx"

    torch.onnx.export(
        pipeline,
        (dummy_image, dummy_mask),
        onnx_export_path,
        verbose=False,
        export_params=True,
        dynamic_axes={
            "image": {0: "batch_size", 2: "height", 3: "width"},
            "mask": {0: "batch_size", 2: "height", 3: "width"},
            "result": {0: "batch_size", 2: "height", 3: "width"},
        },
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        opset_version=17,
    )
    print("ONNX model exported to:", onnx_export_path)

    # Simplify the ONNX model
    simp_export_path = args.output_dir / "models" / "migan_batch_simp.onnx"
    model = onnx.load(onnx_export_path)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, simp_export_path)

    # Optional: quick run using onnxruntime with mini-batches
    ort_sess = ort.InferenceSession(str(simp_export_path))

    img_extensions = {".jpg", ".jpeg", ".png"}
    img_paths = []
    for img_extension in img_extensions:
        img_paths += glob(
            os.path.join(args.images_dir, "**", f"*{img_extension}"),
            recursive=True,
        )
    img_paths = sorted(img_paths)
    (args.output_dir / "sample_results").mkdir(parents=True, exist_ok=True)

    # process images in batches
    batch_size = 4
    batch_imgs = []
    batch_masks = []
    batch_names = []

    for img_path in tqdm(img_paths):
        mask_path = os.path.join(
            args.masks_dir,
            "".join(os.path.basename(img_path).split(".")[:-1]) + ".png",
        )
        img = Image.open(img_path).convert("RGB")
        mask = read_mask(mask_path, invert=args.invert_mask)

        img_np = np.array(img)  # (H,W,3) uint8
        mask_np = np.array(mask)[:, :, np.newaxis]  # (H,W,1) uint8

        batch_imgs.append(img_np)
        batch_masks.append(mask_np)
        batch_names.append(img_path)

        if len(batch_imgs) == batch_size:
            imgs_arr = (
                np.stack(batch_imgs, axis=0)
                .transpose(0, 3, 1, 2)
                .astype(np.uint8)
            )
            masks_arr = (
                np.stack(batch_masks, axis=0)
                .transpose(0, 3, 1, 2)
                .astype(np.uint8)
            )
            result = ort_sess.run(
                None, {"image": imgs_arr, "mask": masks_arr}
            )[0]
            for i in range(result.shape[0]):
                out_np = result[i].transpose(1, 2, 0).astype(np.uint8)
                out_img = Image.fromarray(out_np)
                out_img.save(
                    args.output_dir
                    / "sample_results"
                    / f"{Path(batch_names[i]).stem}.png"
                )
            batch_imgs = []
            batch_masks = []
            batch_names = []

    # leftover
    if len(batch_imgs) > 0:
        imgs_arr = (
            np.stack(batch_imgs, axis=0).transpose(0, 3, 1, 2).astype(np.uint8)
        )
        masks_arr = (
            np.stack(batch_masks, axis=0)
            .transpose(0, 3, 1, 2)
            .astype(np.uint8)
        )
        result = ort_sess.run(None, {"image": imgs_arr, "mask": masks_arr})[0]
        for i in range(result.shape[0]):
            out_np = result[i].transpose(1, 2, 0).astype(np.uint8)
            out_img = Image.fromarray(out_np)
            out_img.save(
                args.output_dir
                / "sample_results"
                / f"{Path(batch_names[i]).stem}.png"
            )

    print("Sample results saved to:", args.output_dir / "sample_results")


if __name__ == "__main__":
    main()
