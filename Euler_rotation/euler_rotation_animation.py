#!/usr/bin/env python3
"""
Euler angle attitude change animation.

This script visualizes the process of changing attitude by Euler rotations,
rotating one axis at a time and plotting the body-frame X, Y, Z axes.

Convention used here follows the lecture note 0300:
    {b} = C_axis3(phi3) C_axis2(phi2) C_axis1(phi1) {a}
where each C_axis is the coordinate transformation matrix in the lecture note.
For visualization in a fixed inertial 3D plot, body axes are drawn as columns of
C.T, because {b}^T = {a}^T C.T.

Examples:
    python3 euler_rotation_animation.py --sequence ZYX --initial 0 0 0 --final 45 30 60
    python3 euler_rotation_animation.py --sequence ZXZ --initial 0 10 0 --final 90 45 30 --save attitude.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    from matplotlib.animation import FFMpegWriter
except Exception:  # pragma: no cover
    FFMpegWriter = None


VALID_SEQUENCES = {
    "XYX", "XYZ", "XZX", "XZY",
    "YXY", "YXZ", "YZX", "YZY",
    "ZXY", "ZXZ", "ZYX", "ZYZ",
}


@dataclass(frozen=True)
class EulerAnimationConfig:
    sequence: str
    initial_deg: np.ndarray
    final_deg: np.ndarray
    frames_per_axis: int = 60
    interval_ms: int = 40
    axis_length: float = 1.0


def c_x(angle_rad: float) -> np.ndarray:
    """Lecture-note coordinate transformation matrix about X axis."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, s],
        [0.0, -s, c],
    ])


def c_y(angle_rad: float) -> np.ndarray:
    """Lecture-note coordinate transformation matrix about Y axis."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, 0.0, -s],
        [0.0, 1.0, 0.0],
        [s, 0.0, c],
    ])


def c_z(angle_rad: float) -> np.ndarray:
    """Lecture-note coordinate transformation matrix about Z axis."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


ROTATION_MATRIX = {"X": c_x, "Y": c_y, "Z": c_z}


def euler_to_dcm(sequence: str, angles_rad: Iterable[float]) -> np.ndarray:
    """
    Convert Euler angles to the lecture-note direction cosine matrix.

    For sequence='ZYX' and angles=[phi1, phi2, phi3], this returns
        C = C_X(phi3) C_Y(phi2) C_Z(phi1)
    which is the C_321 (= C_ZYX) form used in the lecture note.

    More generally, if sequence = a1 a2 a3,
        C = C_a3(phi3) C_a2(phi2) C_a1(phi1)
    """
    angles = list(angles_rad)
    if len(angles) != 3:
        raise ValueError("Euler angle vector must have exactly 3 elements.")
    if sequence not in VALID_SEQUENCES:
        raise ValueError(f"Unsupported Euler sequence '{sequence}'. Use one of {sorted(VALID_SEQUENCES)}.")

    c_total = np.eye(3)
    for axis, angle in zip(sequence, angles):
        # Left multiplication implements C_a3 C_a2 C_a1 after the loop.
        c_total = ROTATION_MATRIX[axis](angle) @ c_total
    return c_total


def make_one_axis_at_a_time_path(initial_rad: np.ndarray, final_rad: np.ndarray, frames_per_axis: int) -> np.ndarray:
    """
    Generate an Euler-angle path that changes angle 1, then angle 2, then angle 3.
    The returned shape is (3*frames_per_axis, 3).
    """
    if frames_per_axis < 2:
        raise ValueError("frames_per_axis must be >= 2.")

    path = []
    current = initial_rad.astype(float).copy()
    for k in range(3):
        start = current[k]
        stop = final_rad[k]
        for value in np.linspace(start, stop, frames_per_axis, endpoint=True):
            a = current.copy()
            a[k] = value
            path.append(a)
        current[k] = stop
    return np.vstack(path)


def body_axes_in_inertial(sequence: str, angles_rad: np.ndarray) -> np.ndarray:
    """
    Return body-axis unit vectors expressed in the inertial/reference frame.

    Lecture note defines {b} = C {a}; therefore {b}^T = {a}^T C.T.
    The columns of C.T are the body X, Y, Z axes drawn in the reference frame.
    """
    c = euler_to_dcm(sequence, angles_rad)
    return c.T


def set_equal_3d_axes(ax, limit: float) -> None:
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("Reference X")
    ax.set_ylabel("Reference Y")
    ax.set_zlabel("Reference Z")




def format_angle_for_filename(angle_deg: float) -> str:
    """Format an angle value for use in an output filename."""
    if np.isclose(angle_deg, round(angle_deg)):
        text = str(int(round(angle_deg)))
    else:
        text = f"{angle_deg:.6g}"
    return text.replace("-", "m").replace(".", "p")


def format_euler_triplet_for_filename(angles_deg: np.ndarray) -> str:
    """Format three Euler angle values as A1-A2-A3 for use in filenames."""
    return "-".join(format_angle_for_filename(v) for v in angles_deg)


def png_filenames(sequence: str, initial_deg: np.ndarray, final_deg: np.ndarray) -> tuple[str, str]:
    """Return filenames for initial and rotated attitude snapshots.

    Example:
        sequence=ZXZ, initial=(0,0,0), final=(45,60,30)
        -> ZXZ_0-0-0.png
        -> ZXZ_0-0-0_45-60-30.png
    """
    initial_text = format_euler_triplet_for_filename(initial_deg)
    final_text = format_euler_triplet_for_filename(final_deg)
    return f"{sequence}_{initial_text}.png", f"{sequence}_{initial_text}_{final_text}.png"


def draw_attitude_snapshot(
    sequence: str,
    angles_deg: np.ndarray,
    title_text: str,
    output_path: str,
    axis_length: float = 1.0,
) -> None:
    """Save a static PNG image of the body axes for a given Euler attitude."""
    angles_rad = np.deg2rad(angles_deg)
    axes = body_axes_in_inertial(sequence, angles_rad) * axis_length

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    set_equal_3d_axes(ax, 1.25 * axis_length)
    ax.view_init(elev=25, azim=35)

    unit = np.eye(3)
    for i, name in enumerate(["X_ref", "Y_ref", "Z_ref"]):
        ax.plot([0, unit[0, i]], [0, unit[1, i]], [0, unit[2, i]], linestyle="--", linewidth=1)
        ax.text(unit[0, i] * 1.08, unit[1, i] * 1.08, unit[2, i] * 1.08, name)

    for i, name in enumerate(["X_B", "Y_B", "Z_B"]):
        v = axes[:, i]
        ax.plot([0.0, v[0]], [0.0, v[1]], [0.0, v[2]], linewidth=3)
        ax.text(v[0] * 1.12, v[1] * 1.12, v[2] * 1.12, name)

    ax.set_title(title_text)
    ax.text2D(
        0.03,
        0.03,
        f"Euler sequence {sequence}\nangles [deg] = [{angles_deg[0]:.2f}, {angles_deg[1]:.2f}, {angles_deg[2]:.2f}]",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def animate_euler_rotation(config: EulerAnimationConfig, save_path: str | None = None) -> None:
    sequence = config.sequence.upper()
    initial_rad = np.deg2rad(config.initial_deg)
    final_rad = np.deg2rad(config.final_deg)
    angle_path = make_one_axis_at_a_time_path(initial_rad, final_rad, config.frames_per_axis)

    init_filename, rotated_filename = png_filenames(sequence, config.initial_deg, config.final_deg)
    init_png = Path(init_filename)
    rotated_png = Path(rotated_filename)
    draw_attitude_snapshot(
        sequence,
        config.initial_deg,
        f"Initial attitude: Euler sequence {sequence}",
        str(init_png),
        axis_length=config.axis_length,
    )
    draw_attitude_snapshot(
        sequence,
        config.final_deg,
        f"Rotated attitude: Euler sequence {sequence}",
        str(rotated_png),
        axis_length=config.axis_length,
    )
    print(f"Saved initial PNG: {init_png}")
    print(f"Saved rotated PNG: {rotated_png}")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    set_equal_3d_axes(ax, 1.25 * config.axis_length)
    ax.view_init(elev=25, azim=35)

    # Reference frame axes: thin dashed lines.
    ref_lines = []
    ref_labels = []
    ref_names = ["X_ref", "Y_ref", "Z_ref"]
    unit = np.eye(3)
    for i, name in enumerate(ref_names):
        line, = ax.plot([0, unit[0, i]], [0, unit[1, i]], [0, unit[2, i]], linestyle="--", linewidth=1)
        ref_lines.append(line)
        ref_labels.append(ax.text(unit[0, i] * 1.08, unit[1, i] * 1.08, unit[2, i] * 1.08, name))

    # Body frame axes: thick solid lines. Use default matplotlib colors.
    body_lines = []
    body_labels = []
    for name in ["X_B", "Y_B", "Z_B"]:
        line, = ax.plot([0, 0], [0, 0], [0, 0], linewidth=3)
        body_lines.append(line)
        body_labels.append(ax.text(0, 0, 0, name))

    title = ax.set_title("")
    info = ax.text2D(0.03, 0.03, "", transform=ax.transAxes)

    def update(frame: int):
        angles = angle_path[frame]
        axes = body_axes_in_inertial(sequence, angles) * config.axis_length
        active_angle_index = min(frame // config.frames_per_axis, 2)
        active_axis = sequence[active_angle_index]

        for i, line in enumerate(body_lines):
            v = axes[:, i]
            line.set_data([0.0, v[0]], [0.0, v[1]])
            line.set_3d_properties([0.0, v[2]])
            body_labels[i].set_position((v[0] * 1.12, v[1] * 1.12))
            body_labels[i].set_3d_properties(v[2] * 1.12)

        deg = np.rad2deg(angles)
        title.set_text(f"Euler sequence {sequence}: rotating angle {active_angle_index + 1} about {active_axis}-axis")
        info.set_text(
            f"angles [deg] = [{deg[0]:7.2f}, {deg[1]:7.2f}, {deg[2]:7.2f}]\n"
            f"initial [deg] = {config.initial_deg.tolist()}\n"
            f"final   [deg] = {config.final_deg.tolist()}"
        )
        return body_lines + body_labels + [title, info]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(angle_path),
        interval=config.interval_ms,
        blit=False,
        repeat=False,
    )

    if save_path:
        if save_path.lower().endswith(".gif"):
            animation.save(save_path, writer=PillowWriter(fps=max(1, int(1000 / config.interval_ms))))
        else:
            if FFMpegWriter is None:
                raise RuntimeError("FFMpegWriter is unavailable. Save as .gif or install ffmpeg.")
            animation.save(save_path, writer=FFMpegWriter(fps=max(1, int(1000 / config.interval_ms))))
        print(f"Saved animation: {save_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate one-axis-at-a-time Euler-angle attitude change."
    )
    parser.add_argument(
        "--sequence",
        required=True,
        type=str.upper,
        choices=sorted(VALID_SEQUENCES),
        help="Euler rotation sequence, e.g. ZYX, ZXZ, XYZ."
    )
    parser.add_argument(
        "--initial",
        required=True,
        nargs=3,
        type=float,
        metavar=("A1", "A2", "A3"),
        help="Initial Euler angles in degrees, corresponding to --sequence."
    )
    parser.add_argument(
        "--final",
        required=True,
        nargs=3,
        type=float,
        metavar=("A1", "A2", "A3"),
        help="Final Euler angles in degrees, corresponding to --sequence."
    )
    parser.add_argument(
        "--frames-per-axis",
        type=int,
        default=60,
        help="Number of animation frames used for each of the three rotations."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=40,
        help="Animation interval in milliseconds."
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output path, e.g. attitude.gif or attitude.mp4."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EulerAnimationConfig(
        sequence=args.sequence,
        initial_deg=np.array(args.initial, dtype=float),
        final_deg=np.array(args.final, dtype=float),
        frames_per_axis=args.frames_per_axis,
        interval_ms=args.interval,
    )
    animate_euler_rotation(config, save_path=args.save)


if __name__ == "__main__":
    main()
