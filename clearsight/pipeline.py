from clearsight.degrade import degrade
from clearsight.enhance import gamma_correct, auto_gamma, equalize_global, equalize_clahe, denoise, sharpen, smooth_skin
from clearsight.segment import skin_mask_ycrcb, skin_mask_kmeans, clean_mask, locate_face

MODES = {
    "Forensic": dict(
        denoise_method="Auto", strength=2, denoise_first=True,
        auto_boost=True, boost=1.0, contrast="CLAHE", clip=2.0, tile=8,
        sharpen_method="Unsharp mask", amount=0.6, smooth=0.0,
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


def enhance(img, s):
    out = img
    if s["denoise_first"]:
        out = denoise(out, s["denoise_method"], s["strength"])
    out = gamma_correct(out, auto_gamma(out) if s["auto_boost"] else 1 / s["boost"])
    if s["contrast"] == "Global equalization":
        out = equalize_global(out)
    elif s["contrast"] == "CLAHE":
        out = equalize_clahe(out, s["clip"], s["tile"])
    if not s["denoise_first"]:
        out = denoise(out, s["denoise_method"], s["strength"])
    return sharpen(out, s["sharpen_method"], s["amount"])


def skin_mask(img, s):
    if s["seg_method"] == "K-means":
        raw = skin_mask_kmeans(img, s["k"])
    else:
        raw = skin_mask_ycrcb(img, *s["cr_range"], *s["cb_range"])
    return clean_mask(raw, s["mask_clean"])


def run(img, deg, enh, seg):
    degraded = degrade(img, **deg)
    enhanced = enhance(degraded, enh)
    source = img if seg["detect_on"] == "Uploaded image" else degraded
    detect_view = enhance(source, DETECT)
    mask = skin_mask(detect_view, seg)
    person, head, box = locate_face(detect_view, mask)
    final = smooth_skin(enhanced, mask, box, enh["smooth"])
    return dict(
        degraded=degraded, enhanced=final, detect_view=detect_view,
        mask=mask, person=person, head=head, box=box,
    )