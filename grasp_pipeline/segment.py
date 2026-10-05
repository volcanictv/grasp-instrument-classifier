"""Box-prompted instrument segmentation: SAM2.1-large and SAM3, each fine-tuned on GraSP with ground-truth boxes, their mask logits averaged.

Every model is run on the image and on its horizontal flip and the two logit maps are averaged (flip averaging); the ensemble is the equal-weight
mean of the models' flip-averaged logits, thresholded at 0. With only the SAM2 delta given, the segmenter is SAM2 alone, which needs no gated
download; adding the SAM3 delta also needs the base SAM3 weights (facebook/sam3 on Hugging Face, a gated model that needs an access token).

The fine-tuned weights are shipped as deltas: the tensors that differ from the base model (decoder, prompt encoder, neck and the last four encoder
blocks). The base checkpoint plus the delta, loaded with strict=False, is the fine-tuned model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def xywh_to_xyxy(boxes_xywh) -> np.ndarray:
    b = np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)


def flip_boxes(boxes_xyxy: np.ndarray, width: int) -> np.ndarray:
    out = boxes_xyxy.copy()
    out[:, [0, 2]] = width - boxes_xyxy[:, [2, 0]]
    return out


def _sam2_embed(predictor, image: np.ndarray) -> None:
    """The image embedding of SAM2ImagePredictor.set_image, without its no_grad decorator being required by callers."""
    model = predictor.model
    predictor._orig_hw = [image.shape[:2]]
    x = predictor._transforms(image)[None, ...].to(predictor.device)
    backbone_out = model.forward_image(x)
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
    feats = [f.permute(1, 2, 0).view(1, -1, *s) for f, s in zip(vision_feats[::-1], predictor._bb_feat_sizes[::-1])][::-1]
    predictor._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
    predictor._is_image_set = True


def _sam2_logits(predictor, boxes_xyxy: np.ndarray) -> torch.Tensor:
    """Mask logits (n, H, W) for n boxes of the embedded image (single-mask output)."""
    _mi, _uc, _l, unnorm_box = predictor._prep_prompts(None, None, boxes_xyxy, None, True)
    coords = unnorm_box.reshape(-1, 2, 2)
    labels = torch.tensor([[2, 3]], dtype=torch.int, device=coords.device).repeat(coords.size(0), 1)
    sparse, dense = predictor.model.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
    high_res = [f[-1].unsqueeze(0) for f in predictor._features["high_res_feats"]]
    low, _iou, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False, repeat_image=True, high_res_features=high_res)
    return predictor._transforms.postprocess_masks(low, predictor._orig_hw[-1])[:, 0].float()


class BoxSegmenter:
    def __init__(
        self,
        sam2_checkpoint: Path | str,
        sam2_delta: Path | str | None = None,
        sam3_delta: Path | str | None = None,
        sam2_config: str = SAM2_CONFIG,
        device: str = "cuda",
        flip: bool = True,
        clip_to_box: bool = False,
    ):
        """`sam2_checkpoint`: the public sam2.1_hiera_large.pt. `sam2_delta`: the fine-tuned SAM2 delta (None = zero-shot SAM2).
        `sam3_delta`: the fine-tuned SAM3 delta; when given, the SAM3 base weights are loaded from Hugging Face (facebook/sam3, gated) and the
        output is the SAM2 + SAM3 ensemble. `clip_to_box`: restrict each mask to its box. Only meaningful when the boxes are the ground-truth
        boxes (tight bounds of the true masks); with detector boxes it can cut true pixels, so it is off by default."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = torch.device(device)
        self.flip = flip
        self.clip_to_box = clip_to_box
        self.sam2 = SAM2ImagePredictor(build_sam2(sam2_config, str(sam2_checkpoint), device=device))
        if sam2_delta is not None:
            result = self.sam2.model.load_state_dict(torch.load(sam2_delta, map_location=device), strict=False)
            assert not result.unexpected_keys, f"unexpected keys in the SAM2 delta: {result.unexpected_keys[:3]}"
        self.sam2.model.eval()
        self.sam3 = self.sam3_processor = None
        if sam3_delta is not None:
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor

            self.sam3 = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
            self.sam3_processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
            result = self.sam3.load_state_dict(torch.load(sam3_delta, map_location=device), strict=False)
            assert not result.unexpected_keys, f"unexpected keys in the SAM3 delta: {result.unexpected_keys[:3]}"
            self.sam3.eval()
        self._autocast = self.device.type == "cuda" and torch.cuda.is_bf16_supported()

    @torch.no_grad()
    def _logits_sam2(self, image: np.ndarray, boxes_xyxy: np.ndarray) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self._autocast):
            _sam2_embed(self.sam2, image)
            return _sam2_logits(self.sam2, boxes_xyxy)

    @torch.no_grad()
    def _logits_sam3(self, image: Image.Image, boxes_xyxy: np.ndarray) -> torch.Tensor:
        proc = self.sam3_processor
        inputs = proc(images=image, input_boxes=[boxes_xyxy.tolist()], return_tensors="pt")
        emb = [e.float() for e in self.sam3.get_image_embeddings(inputs["pixel_values"].to(self.device))]
        out = self.sam3(image_embeddings=emb, input_boxes=inputs["input_boxes"].to(self.device).float(), multimask_output=False)
        return proc.post_process_masks(out.pred_masks, inputs["original_sizes"], binarize=False)[0][:, 0].float()

    def logits(self, image: np.ndarray, boxes_xywh) -> torch.Tensor:
        """Mean of the models' (flip-averaged) mask logits, (n, H, W), on the segmenter's device."""
        boxes = xywh_to_xyxy(boxes_xywh)
        width = image.shape[1]
        total, count = None, 0

        def add(fn_plain, fn_flipped):
            nonlocal total, count
            logits = fn_plain()
            if self.flip:
                logits = (logits + fn_flipped().flip(-1)) / 2
            total = logits if total is None else total + logits
            count += 1

        flipped_np = image[:, ::-1].copy() if self.flip else None
        add(lambda: self._logits_sam2(image, boxes), lambda: self._logits_sam2(flipped_np, flip_boxes(boxes, width)))
        if self.sam3 is not None:
            pil = Image.fromarray(image)
            flipped_pil = pil.transpose(Image.FLIP_LEFT_RIGHT) if self.flip else None
            add(lambda: self._logits_sam3(pil, boxes), lambda: self._logits_sam3(flipped_pil, flip_boxes(boxes, width)))
        return total / count

    def segment(self, image: np.ndarray, boxes_xywh) -> list[np.ndarray]:
        """One boolean HxW mask per box. `image`: RGB uint8 HxWx3. `boxes_xywh`: (x, y, w, h) in native pixels."""
        boxes = np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return []
        masks = (self.logits(image, boxes) > 0).cpu().numpy()
        if self.clip_to_box:
            h, w = masks.shape[1:]
            xs, ys = np.arange(w) + 0.5, np.arange(h) + 0.5
            for k, (x, y, bw, bh) in enumerate(boxes):
                inside = ((ys >= y) & (ys <= y + bh))[:, None] & ((xs >= x) & (xs <= x + bw))[None, :]
                masks[k] &= inside
        return [m for m in masks]
