"""DETR accident-scene model (hilmantm/detr-traffic-accident-detection, Apache-2.0, safetensors) as the accident cue.

The model looks at a whole frame and boxes the region it takes to be an accident. It is a cue for the investigator, not a decision:
it says "this frame looks like an accident scene", not when contact happened, and its boxes tend to persist after the impact. Its
model card notes it was built to cope with traffic jams, and we have not measured its false-alarm rate on our own footage.

The weights are fetched at a pinned revision (hf_models.lock.json, key "accident_detr") and nothing else is executed from the repo
(safetensors holds tensors only; `trust_remote_code` is never enabled).
"""
import json
from pathlib import Path

import torch

REPO_ID = "hilmantm/detr-traffic-accident-detection"
LOCK = Path(__file__).resolve().parents[2] / "hf_models.lock.json"
ACCIDENT_LABELS = {"accident"}


def pinned_revision() -> str:
    return json.loads(LOCK.read_text())["accident_detr"]["revision"]


class AccidentDetr:
    def __init__(self, processor, model, device):
        self.processor, self.model, self.device = processor, model, device

    @classmethod
    def load(cls, device: str | None = None, cache_dir: str | None = None) -> "AccidentDetr":
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        dev = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        rev = pinned_revision()
        kw = {"revision": rev, "cache_dir": cache_dir, "trust_remote_code": False}
        processor = AutoImageProcessor.from_pretrained(REPO_ID, **kw)
        model = AutoModelForObjectDetection.from_pretrained(REPO_ID, use_safetensors=True, **kw).to(dev).eval()
        return cls(processor, model, dev)

    @torch.no_grad()
    def accident_boxes(self, image, conf: float = 0.48) -> list[tuple[float, list[float]]]:
        """[(confidence, [x1, y1, x2, y2])] of accident boxes in source pixels for one PIL image."""
        inputs = {k: v.to(self.device) for k, v in self.processor(images=image, return_tensors="pt").items()}
        out = self.model(**inputs)
        res = self.processor.post_process_object_detection(out, threshold=conf, target_sizes=torch.tensor([image.size[::-1]], device=self.device))[0]
        id2label = self.model.config.id2label
        return [(float(s), [float(x) for x in b.cpu().tolist()]) for s, l, b in zip(res["scores"], res["labels"], res["boxes"])
                if id2label[int(l)].lower() in ACCIDENT_LABELS]
