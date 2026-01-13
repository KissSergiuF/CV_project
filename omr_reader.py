import os
import json
import math
import argparse
import cv2
import numpy as np


# Geometry helpers
def order_points(pts):
    # Orders 4 corner points as: top-left, top-right, bottom-right, bottom-left
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)              # x+y is smallest at top-left, largest at bottom-right
    diff = np.diff(pts, axis=1)      # x-y is smallest at top-right, largest at bottom-left
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    # Warps the paper so it becomes a flat "scanned" rectangle
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute output width using the top and bottom edge lengths
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    # Compute output height using the left and right edge lengths
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    # Target rectangle coordinates (top-left -> top-right -> bottom-right -> bottom-left)
    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype="float32",
    )

    # Perspective transform + warp
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def find_document_contour(gray):
    # Finds a 4-corner contour that most likely represents the paper
    blur = cv2.GaussianBlur(gray, (5, 5), 0)          # remove noise
    edged = cv2.Canny(blur, 50, 150)                  # detect edges
    edged = cv2.dilate(edged, None, iterations=1)     # connect broken edges
    edged = cv2.erode(edged, None, iterations=1)      # remove extra thickness

    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Try the largest contours first (paper is usually large)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:12]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # approximate contour with fewer points
        if len(approx) == 4 and cv2.contourArea(approx) > 10000:
            return approx.reshape(4, 2)

    return None


def choose_best_k_by_spacing(r_sorted, k=5):
    # If a row contains extra bubble-like contours, keep the k bubbles with the most consistent spacing
    if len(r_sorted) <= k:
        return r_sorted

    centers = [b[1] + b[3] / 2.0 for b in r_sorted]  # bubble x-center = x + w/2
    best = None
    best_score = 1e9

    for start in range(0, len(r_sorted) - k + 1):
        c = centers[start:start + k]
        gaps = np.diff(c)                 # distances between neighboring centers
        mean_gap = float(np.mean(gaps))   # average spacing
        std_gap = float(np.std(gaps))     # how uneven spacing is

        # Soft penalties to avoid weird "groups" that are too tight or too wide
        penalty = 0.0
        if mean_gap < 30:
            penalty += (30 - mean_gap) * 0.5
        if mean_gap > 220:
            penalty += (mean_gap - 220) * 0.1

        score = std_gap + penalty
        if score < best_score:
            best_score = score
            best = r_sorted[start:start + k]

    return best


# Thresholding + bubble extraction (used only to find bubble contours)
def build_threshold_images(warped_gray):
    # CLAHE improves contrast so bubbles are easier to separate from the background
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(warped_gray)

    # Adaptive threshold handles uneven lighting/shadows
    thresh_adapt = cv2.adaptiveThreshold(
        norm, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 12
    )
    # Morph open removes small noise blobs
    thresh_adapt = cv2.morphologyEx(
        thresh_adapt,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )

    # Otsu threshold is good when lighting is cleaner/more uniform
    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh_otsu = cv2.morphologyEx(
        thresh_otsu,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )

    return thresh_adapt, thresh_otsu


def extract_bubbles(thresh, W):
    # Extracts contours that look like bubbles (size + shape filters)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []

    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)

        # Ignore punch holes / border junk near left/right edges
        if x < int(0.08 * W) or x + w > int(0.99 * W):
            continue

        # Ignore small blobs and tiny contours
        if area < 70:
            continue
        if w < 10 or h < 10:
            continue

        # Bubble should be roughly circular (width close to height)
        ar = w / float(h)
        if ar < 0.65 or ar > 1.55:
            continue

        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue

        # Circularity score: 1 is a perfect circle; lower means less circle-like
        circularity = 4 * math.pi * area / (peri * peri)
        if circularity < 0.25:
            continue

        bubbles.append((c, x, y, w, h, area))

    return bubbles


# Otsu threshold on 1D values (used to split "empty" vs "filled" darkness scores per question)
def otsu_threshold_1d(values):
    # Finds a split value that best separates low scores from high scores
    v = np.array(values, dtype=np.float32)
    if len(v) < 2:
        return float(v[0]) if len(v) == 1 else 0.0

    v_sorted = np.sort(v)
    candidates = [(v_sorted[i] + v_sorted[i + 1]) / 2.0 for i in range(len(v_sorted) - 1)]

    best_t = candidates[0]
    best_bcvar = -1.0

    overall_var = float(np.var(v)) + 1e-6

    for t in candidates:
        w0 = v[v < t]    # group below threshold (usually empty bubbles)
        w1 = v[v >= t]   # group above threshold (usually filled bubbles)
        if len(w0) == 0 or len(w1) == 0:
            continue

        p0 = len(w0) / len(v)       # fraction of values in the low group
        p1 = len(w1) / len(v)       # fraction of values in the high group
        m0 = float(np.mean(w0))     # mean of the low group
        m1 = float(np.mean(w1))     # mean of the high group

        bcvar = p0 * p1 * (m0 - m1) ** 2     # between-class variance (bigger = better separation)
        bcvar_norm = bcvar / overall_var     # normalized so it behaves well across ranges

        if bcvar_norm > best_bcvar:
            best_bcvar = bcvar_norm
            best_t = t

    return float(best_t)


# Main detection
def detect_answers(image_path, choices_per_q=5, debug_dir=None, print_scores=False):
    # Load the image from disk
    image = cv2.imread(image_path)
    if image is None:
        return None, {"error": f"Image not found: {image_path}"}

    # Convert to grayscale for detection/scoring
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the paper corners in the original photo
    doc = find_document_contour(gray)
    if doc is None:
        return None, {"error": "Could not find document contour (4 corners)."}

    # Warp the page so bubbles become aligned and easier to process
    warped_color = four_point_transform(image, doc)
    warped_gray = four_point_transform(gray, doc)

    # Resize to a fixed width to keep thresholds/tolerances stable across photos
    target_w = 900
    scale = target_w / warped_gray.shape[1]
    warped_gray = cv2.resize(warped_gray, (target_w, int(warped_gray.shape[0] * scale)))
    warped_color = cv2.resize(warped_color, (target_w, int(warped_color.shape[0] * scale)))
    H, W = warped_gray.shape

    # Build both threshold variants
    thresh_adapt, thresh_otsu = build_threshold_images(warped_gray)

    # Extract bubble candidates from both variants
    bubbles_a = extract_bubbles(thresh_adapt, W)
    bubbles_o = extract_bubbles(thresh_otsu, W)

    # Pick the threshold variant that finds more bubble candidates
    if len(bubbles_o) > len(bubbles_a):
        thresh = thresh_otsu
        bubbles = bubbles_o
        thresh_method = "otsu"
    else:
        thresh = thresh_adapt
        bubbles = bubbles_a
        thresh_method = "adaptive"

    if len(bubbles) < choices_per_q:
        return None, {"error": "Not enough bubbles detected.", "bubbles": len(bubbles), "threshold": thresh_method}

    # Sort by y then x so grouping into rows works reliably
    bubbles_sorted = sorted(bubbles, key=lambda b: (b[2], b[1]))

    # Use median bubble height to decide what counts as the "same row"
    med_h = float(np.median([b[4] for b in bubbles_sorted]))
    y_tol = med_h * 1.15

    # Group bubbles into rows (top-to-bottom)
    rows = []
    cur = []
    cur_y = None
    for b in bubbles_sorted:
        y = b[2]
        if cur_y is None or abs(y - cur_y) <= y_tol:
            cur.append(b)
            cur_y = y if cur_y is None else (cur_y * 0.7 + y * 0.3)
        else:
            rows.append(cur)
            cur = [b]
            cur_y = y
    if cur:
        rows.append(cur)

    # Keep only rows that have at least choices_per_q bubbles
    question_rows = []
    for r in rows:
        r_sorted = sorted(r, key=lambda b: b[1])
        if len(r_sorted) >= choices_per_q:
            question_rows.append(choose_best_k_by_spacing(r_sorted, choices_per_q))

    # Prepare a kernel that shrinks the bubble mask so the printed border is ignored
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner_erode_iters = 4

    # Enforce answer count limits to reduce noisy "everything selected" results
    MIN_FILLED_COUNT = 1
    MAX_FILLED_COUNT = 2

    detected = {}
    debug = warped_color.copy()

    for qi, row in enumerate(question_rows, start=1):
        scores = []
        inner_boxes = []

        for (c, x, y, w, h, area) in row:
            # Mask isolates one bubble area
            mask = np.zeros(warped_gray.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)

            # Erode mask so only the inside of the bubble is measured (not the printed ring)
            inner = cv2.erode(mask, inner_kernel, iterations=inner_erode_iters)

            # Mean gray inside bubble -> convert to "darkness" score
            mean_gray = cv2.mean(warped_gray, mask=inner)[0]
            darkness = 1.0 - (mean_gray / 255.0)

            scores.append(darkness)
            inner_boxes.append((x, y, w, h))

        # Use Otsu on these 5 numbers to split empty vs filled for this question
        best_idx = int(np.argmax(scores))
        t = otsu_threshold_1d(scores)
        selected_idx = [i for i, s in enumerate(scores) if s >= t]

        # Ensure at least one answer is returned
        if len(selected_idx) < MIN_FILLED_COUNT:
            selected_idx = [best_idx]

        # If too many were selected, keep only the most plausible ones
        if len(selected_idx) > MAX_FILLED_COUNT:
            order = np.argsort(scores)[::-1]
            top1 = int(order[0])
            top2 = int(order[1])
            if float(scores[top2]) >= float(scores[top1]) * 0.90:
                selected_idx = [top1, top2]
            else:
                selected_idx = [top1]

        # Convert indices 0..4 into letters A..E
        ans = [chr(65 + i) for i in sorted(selected_idx)]
        detected[qi] = ans

        if print_scores:
            print(f"Q{qi} darkness scores:", [round(s, 3) for s in scores], "otsu_t:", round(t, 3), "->", ans)

        # Draw debug overlays: green boxes show all bubbles, blue boxes show selected answers
        for j, (x, y, w, h) in enumerate(inner_boxes):
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(debug, f"{scores[j]:.2f}", (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        for idx in selected_idx:
            x, y, w, h = inner_boxes[idx]
            cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)

    info = {
        "rows_detected": len(question_rows),
        "bubbles_detected": len(bubbles),
        "threshold_method": thresh_method,
    }

    if debug_dir:
        # Save intermediate images for inspection
        os.makedirs(debug_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(debug_dir, f"{base}_warped.png"), warped_color)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_thresh.png"), thresh)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_debug.png"), debug)

    return detected, info


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--debug", default="debug_out")
    parser.add_argument("--choices", type=int, default=5)
    parser.add_argument("--print-scores", action="store_true")
    args = parser.parse_args()

    detected, info = detect_answers(
        args.image,
        choices_per_q=args.choices,
        debug_dir=args.debug,
        print_scores=args.print_scores,
    )

    print("\n--- INFO ---")
    print(json.dumps(info, indent=4))

    print("\n--- DETECTED ANSWERS ---")
    print(json.dumps(detected, indent=4))


if __name__ == "__main__":
    main()
