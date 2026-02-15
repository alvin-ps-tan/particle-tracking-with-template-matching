"""
Template-matching particle tracker.

Provides a rolling-template tracker that uses ``cv2.matchTemplate`` with
normalised cross-correlation (``TM_CCOEFF_NORMED``) to follow particles
across video frames.  Each tracked particle maintains its own template that
is refreshed every frame so the tracker can adapt to appearance changes.
"""

import cv2
import numpy as np

from .visualization import draw_tracks as _draw_tracks

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_TEMPLATE_SIZE = 30
DEFAULT_MASK_RADIUS = 15
DEFAULT_SEARCH_RADIUS = 16
DEFAULT_NEW_MIN_SCORE = 0.35
DEFAULT_TRACK_MIN_SCORE = 0.7
DEFAULT_MAX_NEW_PER_FRAME = 1
DEFAULT_MATCH_METHOD = cv2.TM_CCOEFF_NORMED


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------
def make_vanilla_circle_template(size=31, radius=15, color_bgr=(255, 255, 255)):
    """Create a simple filled-circle template on a black background.

    This is used for *initial* detection of new particles that have not
    yet been assigned a rolling template.
    """
    tmpl = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(tmpl, (size // 2, size // 2), radius, color_bgr, thickness=-1)
    return tmpl


def extract_patch_with_padding(frame_bgr, cx, cy, size):
    """Extract a square patch centred at (cx, cy) with zero-padding at edges.

    Returns an ``(size, size, 3)`` BGR image.
    """
    h, w = frame_bgr.shape[:2]
    half = size // 2
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = x0 + size
    y1 = y0 + size

    patch = np.zeros((size, size, 3), dtype=frame_bgr.dtype)

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)

    dx0, dy0 = sx0 - x0, sy0 - y0
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)

    if sx1 > sx0 and sy1 > sy0:
        patch[dy0:dy1, dx0:dx1] = frame_bgr[sy0:sy1, sx0:sx1]

    return patch


def extract_particle_only_template(frame_bgr, cx, cy, size, radius):
    """Extract a circular patch centred on (cx, cy), black outside the circle.

    The result isolates the particle's appearance so that the background
    does not pollute the template during matching.
    """
    patch = extract_patch_with_padding(frame_bgr, cx, cy, size)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(radius), 255, thickness=-1)
    outp = patch.copy()
    outp[mask == 0] = (0, 0, 0)
    return outp


def near_edge(cx, cy, w, h, r=11):
    """Return True if the point (cx, cy) is within *r* pixels of the frame edge."""
    return cx < r or cy < r or cx > (w - 1 - r) or cy > (h - 1 - r)


def local_circle_template_match(frame_bgr, tmpl_bgr, prev_cx, prev_cy, search_r,
                                method=DEFAULT_MATCH_METHOD):
    """Search for *tmpl_bgr* near ``(prev_cx, prev_cy)`` within *search_r*.

    Only the circular region of the correlation map centred on the
    previous position is considered, preventing the tracker from jumping
    to a distant look-alike.

    Returns
    -------
    new_cx, new_cy : float or None
        Updated centre coordinates (None on failure).
    max_val : float or None
        Best match score.
    res_in_circle : ndarray or None
        Correlation map restricted to the search circle.
    """
    th, tw = tmpl_bgr.shape[:2]
    h, w = frame_bgr.shape[:2]

    x0 = int(round(prev_cx - search_r - tw / 2))
    y0 = int(round(prev_cy - search_r - th / 2))
    x1 = int(round(prev_cx + search_r + tw / 2))
    y1 = int(round(prev_cy + search_r + th / 2))

    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)

    roi = frame_bgr[y0c:y1c, x0c:x1c]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return None, None, None, None

    res = cv2.matchTemplate(roi, tmpl_bgr, method)

    rh, rw = res.shape[:2]
    xs = (np.arange(rw) + x0c) + tw / 2.0
    ys = (np.arange(rh) + y0c) + th / 2.0
    X, Y = np.meshgrid(xs, ys)
    mask = ((X - prev_cx) ** 2 + (Y - prev_cy) ** 2) <= (search_r ** 2)

    res_masked = res.copy()
    res_masked[~mask] = -1.0
    _, max_val, _, max_loc = cv2.minMaxLoc(res_masked)

    top_left = (x0c + max_loc[0], y0c + max_loc[1])
    new_cx = top_left[0] + tw / 2.0
    new_cy = top_left[1] + th / 2.0

    res_in_circle = res.copy()
    res_in_circle[~mask] = 0.0
    return new_cx, new_cy, max_val, res_in_circle


def mask_circle(img_bgr, cx, cy, radius):
    """Black out a circular region in *img_bgr* (in-place)."""
    cv2.circle(img_bgr, (int(round(cx)), int(round(cy))), int(radius), (0, 0, 0), thickness=-1)


# ---------------------------------------------------------------------------
# Track dataclass
# ---------------------------------------------------------------------------
class Track:
    """State for a single tracked particle.

    Attributes
    ----------
    tid : int
        Unique track identifier.
    cx, cy : float
        Current centre coordinates.
    tmpl : ndarray
        Rolling BGR template extracted from the previous frame.
    last_score : float
        Most recent match score.
    fully_formed : bool
        True when the track is away from edges and has a good score.
    """

    def __init__(self, tid, cx, cy, tmpl_bgr):
        self.tid = tid
        self.cx = float(cx)
        self.cy = float(cy)
        self.tmpl = tmpl_bgr
        self.last_score = 1.0
        self.fully_formed = False


# ---------------------------------------------------------------------------
# High-level tracker loop
# ---------------------------------------------------------------------------
def run_tracker(
    video_path,
    output_path="simulation_tracking.mp4",
    template_size=DEFAULT_TEMPLATE_SIZE,
    mask_radius=DEFAULT_MASK_RADIUS,
    search_radius=DEFAULT_SEARCH_RADIUS,
    new_min_score=DEFAULT_NEW_MIN_SCORE,
    track_min_score=DEFAULT_TRACK_MIN_SCORE,
    max_new_per_frame=DEFAULT_MAX_NEW_PER_FRAME,
    fps=15.0,
    show_preview=False,
):
    """Run the rolling-template particle tracker on a video file.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    output_path : str
        Path for the annotated output video.
    template_size : int
        Side length of the square template patch.
    mask_radius : int
        Radius used for circular masking of templates and detections.
    search_radius : int
        Maximum pixel distance to search from the previous position.
    new_min_score : float
        Minimum match score to accept a *new* detection.
    track_min_score : float
        Minimum match score to keep tracking an existing particle.
    max_new_per_frame : int
        Maximum number of new detections per frame.
    fps : float
        Frames per second for the output video.
    show_preview : bool
        If True, display a live OpenCV window.

    Returns
    -------
    dict[int, Track]
        Final state of all active tracks when the video ends.
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    vanilla_tmpl = make_vanilla_circle_template(size=template_size, radius=mask_radius)

    tracks: dict[int, Track] = {}
    free_ids: list[int] = []
    next_id = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_bgr = frame
        h, w = frame_bgr.shape[:2]

        # --- A) Track existing particles ---
        dead_ids = []
        recycled_this_frame = []

        for tid, tr in list(tracks.items()):
            if near_edge(tr.cx, tr.cy, w, h, r=mask_radius):
                dead_ids.append(tid)
                continue

            new_cx, new_cy, score, _ = local_circle_template_match(
                frame_bgr, tr.tmpl, tr.cx, tr.cy, search_radius
            )

            if score is None or near_edge(new_cx, new_cy, w, h, r=mask_radius):
                dead_ids.append(tid)
                continue

            if score >= track_min_score:
                tr.cx, tr.cy = new_cx, new_cy
                tr.last_score = float(score)
                tr.tmpl = extract_particle_only_template(
                    frame_bgr, tr.cx, tr.cy, template_size, mask_radius
                )
            else:
                dead_ids.append(tid)

        for tid in dead_ids:
            if tid in tracks:
                del tracks[tid]
                recycled_this_frame.append(tid)

        for tid, tr in tracks.items():
            tr.fully_formed = (
                not near_edge(tr.cx, tr.cy, w, h, r=mask_radius)
                and tr.last_score >= track_min_score
            )

        # --- B) Mask tracked particles for new detection ---
        masked = frame_bgr.copy()
        for tid, tr in tracks.items():
            if tr.fully_formed:
                mask_circle(masked, tr.cx, tr.cy, mask_radius)

        # --- C) Detect new particles ---
        for _ in range(max_new_per_frame):
            res = cv2.matchTemplate(masked, vanilla_tmpl, DEFAULT_MATCH_METHOD)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val < new_min_score:
                break

            cx = max_loc[0] + vanilla_tmpl.shape[1] / 2.0
            cy = max_loc[1] + vanilla_tmpl.shape[0] / 2.0

            if near_edge(cx, cy, w, h, r=mask_radius):
                mask_circle(masked, cx, cy, mask_radius)
                continue

            if recycled_this_frame:
                tid = recycled_this_frame.pop()
            elif free_ids:
                tid = free_ids.pop()
            else:
                tid = next_id
                next_id += 1

            tmpl0 = extract_particle_only_template(frame_bgr, cx, cy, template_size, mask_radius)
            tracks[tid] = Track(tid, cx, cy, tmpl0)
            mask_circle(masked, cx, cy, mask_radius)

        while recycled_this_frame:
            free_ids.append(recycled_this_frame.pop())

        # --- D) Draw and write ---
        _draw_tracks(frame_bgr, tracks, template_size)
        out.write(frame_bgr)

        if show_preview:
            cv2.imshow("Particle tracking", frame_bgr)
            if cv2.waitKey(40) & 0xFF == 27:
                break

    out.release()
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    return tracks
