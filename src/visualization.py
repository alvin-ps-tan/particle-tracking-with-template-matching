"""
Visualization helpers for particle tracking.

Drawing utilities for annotating video frames with bounding boxes,
track IDs, and other overlays.
"""

import cv2
import numpy as np


def draw_tracks(frame_bgr, tracks, template_size=30):
    """Draw bounding boxes and track IDs on *frame_bgr* (in-place).

    Parameters
    ----------
    frame_bgr : ndarray
        BGR frame to annotate.
    tracks : dict[int, Track]
        Mapping of track ID to Track objects.
    template_size : int
        Side length of the bounding box.
    """
    half = template_size // 2
    for tid, tr in tracks.items():
        x1 = int(round(tr.cx - half))
        y1 = int(round(tr.cy - half))
        x2 = x1 + template_size
        y2 = y1 + template_size
        cv2.rectangle(frame_bgr, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 0), 1)
        cv2.putText(
            frame_bgr,
            f"#{tid}",
            (x1, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
        )


def overlay_bounding_boxes(frame_bgr, bounding_boxes, color=(0, 255, 0), thickness=1):
    """Draw bounding boxes from simulation metadata onto a frame.

    Parameters
    ----------
    frame_bgr : ndarray
        BGR frame to annotate.
    bounding_boxes : list[dict]
        Each dict must contain ``x_center``, ``y_center``, ``width``,
        ``height``.
    color : tuple
        BGR colour for the boxes.
    thickness : int
        Line thickness.
    """
    for bb in bounding_boxes:
        cx, cy = bb["x_center"], bb["y_center"]
        w, h = bb["width"], bb["height"]
        x1, y1 = int(cx - w), int(cy - h)
        x2, y2 = int(cx + w), int(cy + h)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
