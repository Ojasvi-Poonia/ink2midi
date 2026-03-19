"""Staff line detection and symbol-to-staff assignment."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from omr.data.graph_builder import Detection
from omr.utils.logging import get_logger

logger = get_logger("sequencer.staff_analysis")


@dataclass
class Staff:
    """Represents a single 5-line music staff."""

    line_positions: list[float]  # Y-coordinates of 5 staff lines (top to bottom)
    staff_space: float  # Distance between adjacent lines
    top: float  # Y-coordinate of top line
    bottom: float  # Y-coordinate of bottom line
    left: float = 0.0  # X-coordinate of staff start
    right: float = 0.0  # X-coordinate of staff end

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2


class StaffAnalyzer:
    """Detect staff lines and assign symbols to staves.

    Uses horizontal projection profile on binarized input to find
    staff line y-coordinates, then assigns detected symbols to their
    nearest staff.
    """

    def __init__(
        self,
        min_staff_distance: int = 30,
        line_thickness_range: tuple[int, int] = (1, 5),
    ):
        self.min_staff_distance = min_staff_distance
        self.line_thickness_range = line_thickness_range

    def detect_staff_lines(self, image: np.ndarray) -> list[Staff]:
        """Detect staff lines using horizontal projection profile.

        Args:
            image: Input score image (grayscale or BGR).

        Returns:
            List of Staff objects, each with 5 line y-coordinates.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Horizontal projection: sum of black pixels per row
        h_proj = np.sum(binary, axis=1) / 255.0
        img_width = image.shape[1]

        # Find peaks (staff lines have high horizontal density)
        # A staff line spans most of the page width
        threshold = img_width * 0.3  # Line must cover >30% of width
        line_candidates = np.where(h_proj > threshold)[0]

        if len(line_candidates) == 0:
            logger.warning("No staff lines detected")
            return []

        # Cluster nearby rows into individual lines
        lines = self._cluster_line_positions(line_candidates)

        # Group lines into staves (5 lines each)
        staves = self._group_into_staves(lines, img_width)

        logger.info(f"Detected {len(staves)} staves with {len(lines)} total lines")
        return staves

    def _cluster_line_positions(self, candidates: np.ndarray) -> list[float]:
        """Cluster consecutive row indices into line center positions."""
        if len(candidates) == 0:
            return []

        lines = []
        group_start = candidates[0]
        prev = candidates[0]

        for y in candidates[1:]:
            if y - prev > self.line_thickness_range[1]:
                # End of current line group
                center = (group_start + prev) / 2.0
                lines.append(center)
                group_start = y
            prev = y

        # Last group
        center = (group_start + prev) / 2.0
        lines.append(center)

        return lines

    def _group_into_staves(
        self, lines: list[float], img_width: int
    ) -> list[Staff]:
        """Group detected lines into 5-line staves."""
        if len(lines) < 5:
            logger.warning(f"Found only {len(lines)} lines, need at least 5")
            return []

        # Compute inter-line distances
        distances = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]

        # Find the typical staff-space (most common small distance)
        small_dists = [d for d in distances if d < self.min_staff_distance * 2]
        if not small_dists:
            staff_space = np.median(distances)
        else:
            staff_space = np.median(small_dists)

        # Group lines where consecutive distances are near staff_space
        staves = []
        i = 0
        while i <= len(lines) - 5:
            # Check if lines[i:i+5] form a valid staff
            group = lines[i : i + 5]
            spacings = [group[j + 1] - group[j] for j in range(4)]

            # All spacings should be approximately equal to staff_space
            if all(abs(s - staff_space) < staff_space * 0.5 for s in spacings):
                avg_space = np.mean(spacings)
                staves.append(
                    Staff(
                        line_positions=group,
                        staff_space=avg_space,
                        top=group[0],
                        bottom=group[4],
                        left=0.0,
                        right=float(img_width),
                    )
                )
                i += 5  # Skip past this staff
            else:
                i += 1

        return staves

    def assign_symbols_to_staves(
        self,
        detections: list[Detection],
        staves: list[Staff],
    ) -> dict[int, list[Detection]]:
        """Assign each detection to its closest staff.

        Args:
            detections: List of detected symbols.
            staves: List of detected staves.

        Returns:
            Dict mapping staff index to list of detections on that staff.
        """
        assignments: dict[int, list[Detection]] = {i: [] for i in range(len(staves))}

        for det in detections:
            _, cy = det.center
            best_staff = 0
            best_dist = float("inf")

            for i, staff in enumerate(staves):
                # Distance from detection center to staff center
                dist = abs(cy - staff.center_y)
                # Also check if detection is within staff bounds (with margin)
                margin = staff.staff_space * 3  # Allow ledger lines
                if staff.top - margin <= cy <= staff.bottom + margin:
                    dist *= 0.5  # Prefer staves where symbol is within bounds

                if dist < best_dist:
                    best_dist = dist
                    best_staff = i

            assignments[best_staff].append(det)

        for staff_idx, dets in assignments.items():
            logger.debug(f"Staff {staff_idx}: {len(dets)} symbols")

        return assignments
