import cv2
import numpy as np
from pathlib import Path


# Keep the largest connected component in a binary mask
def keep_largest(mask_u8):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return mask_u8
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask_u8)
    out[labels == k] = 255
    return out


# Fill holes inside foreground regions
def fill_holes(mask_u8):
    h, w = mask_u8.shape
    ff = mask_u8.copy()
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(mask_u8, holes)


# Fill the convex hull of the largest contour
def convex_hull_fill(mask_u8):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask_u8
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    out = np.zeros_like(mask_u8)
    cv2.fillConvexPoly(out, hull, 255)
    return out


# Build a skin-like mask in YCrCb 
def skin_mask_ycrcb_debug(bgr, dilate_iter=3):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, ker, iterations=1)
    skin = cv2.dilate(skin, ker, iterations=dilate_iter)
    return skin


# Get red seed (strict) and red allow-region (loose) in HSV
def red_seed_allow(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lo1 = cv2.inRange(hsv, (0, 30, 20), (15, 255, 255))
    lo2 = cv2.inRange(hsv, (165, 30, 20), (180, 255, 255))
    allow = cv2.bitwise_or(lo1, lo2)

    hi1 = cv2.inRange(hsv, (0, 80, 50), (12, 255, 255))
    hi2 = cv2.inRange(hsv, (168, 80, 50), (180, 255, 255))
    seed = cv2.bitwise_or(hi1, hi2)

    return seed, allow


# Constrained growing/closing inside the allow-region
def constrained_close(seed_u8, allow_u8, iters=25):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cur = (seed_u8 > 0).astype(np.uint8)
    allow = (allow_u8 > 0).astype(np.uint8)

    for _ in range(iters):
        dil = cv2.dilate(cur, ker, iterations=1)
        nxt = (dil & allow)
        if np.array_equal(nxt, cur):
            break
        cur = nxt

    out = (cur * 255).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker, iterations=2)
    return out


# Refine mask boundary using GrabCut initialized by a mask
def grabcut_refine(bgr, init_mask_u8, iters=10):
    h, w = init_mask_u8.shape
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[init_mask_u8 > 0] = cv2.GC_PR_FGD

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure = cv2.erode(init_mask_u8, ker, iterations=2)
    gc[sure > 0] = cv2.GC_FGD

    border = 8
    gc[:border, :] = cv2.GC_BGD
    gc[-border:, :] = cv2.GC_BGD
    gc[:, :border] = cv2.GC_BGD
    gc[:, -border:] = cv2.GC_BGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)

    # Keep both definite and probable foreground 
    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return out


# Remove tiny disconnected islands while keeping the main component intact
def remove_small_islands(mask_u8, min_area=800):
    m = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return mask_u8

    main_k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask_u8)

    for k in range(1, num):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if k == main_k:
            out[labels == k] = 255
        else:
            if area >= min_area:
                out[labels == k] = 255

    return out


# Convert a grayscale/binary image to 3-channel BGR for visualization
def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# Put a short title on the image 
def put_title(bgr, title):
    out = bgr.copy()
    cv2.putText(out, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 1, cv2.LINE_AA)
    return out


# Build a single summary image that contains all step images in order
def build_summary_grid(images, titles, cols=5, cell_w=330, cell_h=300, pad=10):
    assert len(images) == len(titles)
    n = len(images)
    rows = int(np.ceil(n / cols))

    grid_w = cols * cell_w + (cols + 1) * pad
    grid_h = rows * cell_h + (rows + 1) * pad
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, (img, title) in enumerate(zip(images, titles)):
        r = i // cols
        c = i % cols
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_h + pad)

        bgr = to_bgr(img)
        bgr = cv2.resize(bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        bgr = put_title(bgr, title)

        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = bgr

    return canvas


def segment_red_helmet_submit(img_path, out_dir="helmet_submit_out", min_island_area=800):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 01: Input
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(img_path)
    h, w = bgr.shape[:2]

    # Step 02: Red seed
    seed, allow = red_seed_allow(bgr)
    seed = keep_largest(seed)

    # Step 03: Red allow region
    # (allow is already computed above in Step 02)

    # Step 04: Constrained growth
    grown = constrained_close(seed, allow, iters=25)
    grown = keep_largest(grown)
    grown = fill_holes(grown)
    # Skin mask
    skin = skin_mask_ycrcb_debug(bgr, dilate_iter=3)

    # Step 05: Geometry constrained
    grown_no_skin = cv2.bitwise_and(grown, cv2.bitwise_not(skin))
    ys = np.where(grown_no_skin > 0)[0]
    if ys.size > 0:
        y_min, y_max = int(ys.min()), int(ys.max())
        cut_y = int(y_min + 0.88 * (y_max - y_min + 1))
        geom = np.zeros((h, w), np.uint8)
        geom[:cut_y, :] = 255
        grown_no_skin = cv2.bitwise_and(grown_no_skin, geom)

    # Step 06: Hull + allow constrained
    hull = convex_hull_fill(grown_no_skin)
    hull = cv2.bitwise_and(hull, allow)
    hull = keep_largest(hull)
    hull = fill_holes(hull)

    # Step 07: GrabCut raw
    final_raw = grabcut_refine(bgr, hull, iters=10)

    # Step 08: Remove small islands (final mask)
    final_clean = remove_small_islands(final_raw, min_area=min_island_area)
    final_mask = final_clean

    # Step 09: Contour overlay
    overlay = bgr.copy()
    cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2)

    # Step 10: Cutout
    cutout = cv2.bitwise_and(bgr, bgr, mask=final_mask)

    # Combine 10 step images 
    step_images = [
        bgr,            # 01
        seed,           # 02
        allow,          # 03
        grown,          # 04
        grown_no_skin,  # 05
        hull,           # 06
        final_raw,      # 07
        final_mask,     # 08
        overlay,        # 09
        cutout,         # 10
    ]
    step_titles = [
        "Step 01 - Input",
        "Step 02 - Red Seed",
        "Step 03 - Red Allow Region",
        "Step 04 - Constrained Growth",
        "Step 05 - Geometry Constrained",
        "Step 06 - Hull + Allow Constrained",
        "Step 07 - GrabCut Raw",
        "Step 08 - Final Mask",
        "Step 09 - Contour Overlay",
        "Step 10 - Cutout",
    ]

    # Build and write the final summary (2 rows x 5 cols)
    summary = build_summary_grid(step_images, step_titles, cols=5, cell_w=330, cell_h=300, pad=10)
    cv2.imwrite(str(out_dir / "pipeline_summary.png"), summary)
    print("Save summary image:", str(out_dir / "pipeline_summary.png"))


if __name__ == "__main__":
    segment_red_helmet_submit(
        img_path="helmet.png",
        out_dir="helmet_out",
        min_island_area=800
    )
