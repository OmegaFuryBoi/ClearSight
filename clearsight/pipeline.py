from clearsight.degrade import degrade
from clearsight.enhance import gamma_correct, auto_gamma, equalize_global, equalize_clahe, denoise, sharpen, smooth_skin
from clearsight.segment import skin_mask_ycrcb, skin_mask_kmeans, clean_mask, locate_face

MODES = {
    "Forensic": dict(
        denoise_method="Auto", strength=3, denoise_first=True,
        auto_boost=True, boost=1.0, contrast="None", clip=2.0, tile=8,
        sharpen_method="None", amount=1.0, smooth=0.0,
        up_method="Bicubic", progressive=True, up_sharpen=0.5,
    ),
    "Beauty": dict(
        denoise_method="Auto", strength=3, denoise_first=True,
        auto_boost=True, boost=1.0, contrast="CLAHE", clip=1.5, tile=8,
        sharpen_method="None", amount=1.0, smooth=0.8,
        up_method="Bicubic", progressive=False, up_sharpen=0.0,
    ),
    "Off": dict(
        denoise_method="None", strength=2, denoise_first=True,
        auto_boost=False, boost=1.0, contrast="None", clip=2.0, tile=8,
        sharpen_method="None", amount=1.0, smooth=0.0,
        up_method="Bicubic", progressive=False, up_sharpen=0.0,
    ),
}

DETECT = dict(
    denoise_method="Auto", strength=3, denoise_first=True,
    auto_boost=True, boost=1.0, contrast="CLAHE", clip=2.0, tile=8,
    sharpen_method="None", amount=1.0,
)


def enhance(img, s, stages=None):
    stages = [] if stages is None else stages
    out = img
    if s["denoise_first"] and s["denoise_method"] != "None":
        out = denoise(out, s["denoise_method"], s["strength"])
        stages.append((f"Denoised ({s['denoise_method']})", out))
    g = auto_gamma(out) if s["auto_boost"] else 1 / s["boost"]
    if g != 1.0:
        out = gamma_correct(out, g)
        stages.append((f"Brightened (gamma {g:.2f})", out))
    if s["contrast"] == "Global equalization":
        out = equalize_global(out)
        stages.append(("Global equalization", out))
    elif s["contrast"] == "CLAHE":
        out = equalize_clahe(out, s["clip"], s["tile"])
        stages.append(("CLAHE", out))
    if not s["denoise_first"] and s["denoise_method"] != "None":
        out = denoise(out, s["denoise_method"], s["strength"])
        stages.append((f"Denoised ({s['denoise_method']})", out))
    if s["sharpen_method"] != "None":
        out = sharpen(out, s["sharpen_method"], s["amount"])
        stages.append((f"Sharpened ({s['sharpen_method']})", out))
    return out


def skin_mask(img, s):
    if s["seg_method"] == "K-means":
        raw = skin_mask_kmeans(img, s["k"])
    else:
        raw = skin_mask_ycrcb(img, *s["cr_range"], *s["cb_range"])
    return clean_mask(raw, s["mask_clean"])


def run(img, deg, enh, seg, tracker=None):
    degraded = degrade(img, **deg)
    stages = [("Original", img), ("Degraded", degraded)]
    enhanced = enhance(degraded, enh, stages)
    source = img if seg["detect_on"] == "Uploaded image" else degraded
    person = head = None
    box = tracker.update(source) if tracker is not None and tracker.active() else None
    tracked = box is not None
    if tracked:
        detect_view = source
        mask = skin_mask(enhanced, seg)
    else:
        detect_view = enhance(source, DETECT)
        mask = skin_mask(detect_view, seg)
        person, head, box = locate_face(detect_view, mask)
        if tracker is not None and box:
            tracker.start(source, box)
        elif tracker is not None and tracker.box is not None:
            tracker.frames = 0
            box = tracker.update(source)
            tracked = box is not None
    final = smooth_skin(enhanced, mask, box, enh["smooth"])
    if final is not enhanced:
        stages.append(("Skin smoothed", final))
    return dict(
        degraded=degraded, enhanced=final, detect_view=detect_view,
        mask=mask, person=person, head=head, box=box, tracked=tracked, stages=stages,
    )