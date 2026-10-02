from pathlib import Path
from types import SimpleNamespace
import sys, time
import numpy as np
import torch

USR2 = Path(sys.argv[1]).resolve()
CHECKPOINT = Path(sys.argv[2]).resolve()
CLIP = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(USR2))

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from utils.utils import UNIGRAM1000_LIST

args = SimpleNamespace(
    idim=512, adim=768, aheads=12, eunits=3072, elayers=12,
    ddim=768, dheads=12, dunits=3072, dlayers=6,
    gamma_init=0.1, ctc_rel_weight=0.1,
)

t0 = time.time()
model = E2E(len(UNIGRAM1000_LIST), args)
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
if any(k.startswith("_orig_mod.") for k in ckpt):
    ckpt = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt.items()}
if any(k.startswith("model.backbone.") for k in ckpt):
    ckpt = {k.replace("model.backbone.", "", 1): v for k, v in ckpt.items()
            if k.startswith("model.backbone.")}
model.load_state_dict(ckpt)
model.eval()
print("MODEL_LOAD_S", round(time.time() - t0, 2), flush=True)

rois = np.load(CLIP, allow_pickle=False)["rois"].astype(np.float32)
off = (rois.shape[-1] - 88) // 2
rois = rois[:, off:off+88, off:off+88] / 255.0
rois = (rois - 0.421) / 0.165
x = torch.from_numpy(rois).unsqueeze(0)
print("INPUT", tuple(x.shape), flush=True)

t1 = time.time()
with torch.inference_mode():
    feat = model.encoder(xs_v=x)
    logp = model.ctc_v.log_softmax(feat)[0]
ids = logp.argmax(-1).tolist()
collapsed = []
prev = None
eos = len(UNIGRAM1000_LIST) - 1
for i in ids:
    if i != prev and i not in (0, eos):
        collapsed.append(i)
    prev = i
tokens = [UNIGRAM1000_LIST[i] for i in collapsed]
text = "".join(tokens).replace("▁", " ").strip()
print("ENCODE_S", round(time.time() - t1, 2), flush=True)
print("CTC_IDS", collapsed, flush=True)
print("CTC_TOKENS", [t.encode("unicode_escape").decode("ascii") for t in tokens], flush=True)
print("CTC_TEXT_ESC", text.encode("unicode_escape").decode("ascii"), flush=True)
