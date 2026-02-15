"""
Particle simulation module.

Generates synthetic particle data — colored circles that spawn at the edges
of a frame and perform directional random walks. Used to produce test videos
for the template-matching tracker.
"""

import random

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------
DEFAULT_FRAME_WIDTH = 600
DEFAULT_FRAME_HEIGHT = 600
DEFAULT_PARTICLE_RADIUS = 15
DEFAULT_MIN_CENTER_DIST = 28
DEFAULT_MAX_PARTICLES = 80


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _is_too_close(pos, other_positions, min_dist):
    """Return True if *pos* is within *min_dist* of any position in the list."""
    x, y = pos
    min_dist2 = min_dist * min_dist
    for ox, oy in other_positions:
        dx = x - ox
        dy = y - oy
        if dx * dx + dy * dy < min_dist2:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_particle(
    frame_width=DEFAULT_FRAME_WIDTH,
    frame_height=DEFAULT_FRAME_HEIGHT,
    radius=DEFAULT_PARTICLE_RADIUS,
    min_center_dist=DEFAULT_MIN_CENTER_DIST,
    existing_positions=None,
):
    """Spawn a new particle at a random edge location.

    Parameters
    ----------
    frame_width, frame_height : int
        Dimensions of the simulation canvas.
    radius : int
        Particle radius in pixels.
    min_center_dist : int
        Minimum allowed distance between the new particle's center and any
        existing particle center.
    existing_positions : list[tuple] or None
        Centers of already-existing particles.

    Returns
    -------
    dict
        Particle dictionary with keys: position, color, radius, angle,
        start_pos.
    """
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    uniform_random = np.random.uniform()

    if uniform_random <= 0.25:
        position = (random.randint(radius, frame_width - radius), radius)
        start_pos = "bottom"
    elif uniform_random <= 0.5:
        position = (random.randint(radius, frame_width - radius), frame_height - radius)
        start_pos = "top"
    elif uniform_random <= 0.75:
        position = (radius, random.randint(radius, frame_height - radius))
        start_pos = "left"
    else:
        position = (frame_width - radius, random.randint(radius, frame_height - radius))
        start_pos = "right"

    if existing_positions is not None:
        for _ in range(100):
            if not _is_too_close(position, existing_positions, min_center_dist):
                break
            if start_pos == "bottom":
                position = (random.randint(radius, frame_width - radius), radius)
            elif start_pos == "top":
                position = (random.randint(radius, frame_width - radius), frame_height - radius)
            elif start_pos == "left":
                position = (radius, random.randint(radius, frame_height - radius))
            else:
                position = (frame_width - radius, random.randint(radius, frame_height - radius))

    return {
        "position": position,
        "color": color,
        "radius": radius,
        "angle": 0,
        "start_pos": start_pos,
    }


def propose_move(particle):
    """Propose a new position for *particle* based on a random directional step.

    The movement direction is biased toward the interior of the frame
    depending on which edge the particle originally spawned from.
    """
    start_pos = particle["start_pos"]
    radius = particle["radius"]

    if start_pos == "bottom":
        angle = random.randint(0, 180)
    elif start_pos == "top":
        angle = random.randint(180, 360)
    elif start_pos == "left":
        angle = random.randint(-90, 90)
    else:
        angle = random.randint(90, 270)

    angle_rad = np.deg2rad(angle)
    dx = int(radius * np.cos(angle_rad))
    dy = int(radius * np.sin(angle_rad))
    x, y = particle["position"]
    return (x + dx, y + dy)


def move_particle(particle, other_positions=None, min_center_dist=DEFAULT_MIN_CENTER_DIST, max_retries=40):
    """Move *particle* in-place, avoiding centers that are too close.

    If no valid move can be found within *max_retries*, the particle stays
    at its current position.
    """
    if other_positions is None:
        particle["position"] = propose_move(particle)
        return

    for _ in range(max_retries):
        new_pos = propose_move(particle)
        if not _is_too_close(new_pos, other_positions, min_center_dist):
            particle["position"] = new_pos
            return


def is_off_screen(particle, frame_width=DEFAULT_FRAME_WIDTH, frame_height=DEFAULT_FRAME_HEIGHT):
    """Return True if the particle's center is outside the frame bounds."""
    x, y = particle["position"]
    return x < 1 or x > frame_width - 1 or y < 1 or y > frame_height - 1


def draw_frame(particles, frame_width=DEFAULT_FRAME_WIDTH, frame_height=DEFAULT_FRAME_HEIGHT):
    """Render all particles onto a blank frame.

    Returns
    -------
    frame : ndarray
        BGR image of shape (frame_height, frame_width, 3).
    bounding_boxes : list[dict]
        Per-particle bounding-box metadata.
    """
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    bounding_boxes = []
    for particle in particles:
        cv2.circle(frame, particle["position"], particle["radius"], particle["color"], -1)
        x, y = particle["position"]
        bounding_boxes.append({
            "x_center": x,
            "y_center": y,
            "width": particle["radius"],
            "height": particle["radius"],
        })
    return frame, bounding_boxes


def simulate_particles(
    output_path="simulation_detection.mp4",
    frame_width=DEFAULT_FRAME_WIDTH,
    frame_height=DEFAULT_FRAME_HEIGHT,
    max_particles=DEFAULT_MAX_PARTICLES,
    min_center_dist=DEFAULT_MIN_CENTER_DIST,
    fps=15.0,
    show_preview=False,
):
    """Run the full particle simulation, writing frames to *output_path*.

    Parameters
    ----------
    output_path : str
        Path for the output .mp4 video.
    frame_width, frame_height : int
        Canvas dimensions.
    max_particles : int
        Total number of particles spawned over the simulation.
    min_center_dist : int
        Minimum center-to-center distance enforced during spawning and
        movement.
    fps : float
        Frames per second for the output video.
    show_preview : bool
        If True, display a live OpenCV window during simulation.

    Returns
    -------
    list[dict]
        Each entry contains ``'frame'`` (ndarray) and ``'boundary_boxes'``
        (list of dicts).
    """
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    particles = []
    total_data = []
    total_particles_created = 0
    timer = 0

    while len(particles) > 0 or total_particles_created < max_particles:
        positions = [p["position"] for p in particles]

        if total_particles_created < max_particles and timer % 2 == 0:
            total_particles_created += 1
            new_p = create_particle(
                frame_width=frame_width,
                frame_height=frame_height,
                min_center_dist=min_center_dist,
                existing_positions=positions,
            )
            particles.append(new_p)
            positions.append(new_p["position"])

        for particle in particles[:]:
            other_positions = [p["position"] for p in particles if p is not particle]
            move_particle(particle, other_positions=other_positions, min_center_dist=min_center_dist)
            if is_off_screen(particle, frame_width, frame_height):
                particles.remove(particle)

        frame, bounding_boxes = draw_frame(particles, frame_width, frame_height)
        total_data.append({"frame": frame, "boundary_boxes": bounding_boxes})
        out.write(frame)

        if show_preview:
            cv2.imshow("Simulation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        timer += 1

    out.release()
    if show_preview:
        cv2.destroyAllWindows()
    return total_data
