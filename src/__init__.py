"""
Particle Tracking with Template Matching.

A modular library for simulating and tracking particles using
OpenCV template matching with rolling template updates.
"""

from .simulation import (
    create_particle,
    propose_move,
    move_particle,
    is_off_screen,
    draw_frame,
    simulate_particles,
)
from .tracker import (
    Track,
    make_vanilla_circle_template,
    extract_patch_with_padding,
    extract_particle_only_template,
    near_edge,
    local_circle_template_match,
    mask_circle,
    run_tracker,
)
from .visualization import draw_tracks, overlay_bounding_boxes
