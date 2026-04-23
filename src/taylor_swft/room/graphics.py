import cv2
import numpy as np
from .spatial_model import SWFTRoom, make_demo_room


class Graphics:
    def __init__(
        self,
        room_object: SWFTRoom,
        width: int = 1022,
        height: int = 1024,
        z_coordinate: float = 2.0,
        scale: float = 1.3,
        source_color: tuple[int, int, int] = (32, 84, 233),
        mic_color: tuple[int, int, int] = (111, 33, 119),
    ):
        """Initialize graphics context for interactive room visualization.

        Args:
            room_object: Room object created from corners (SWFTRoom with
                from_corners_flag=True).
            width: Canvas width in pixels. Defaults to 1022.
            height: Canvas height in pixels. Defaults to 1024.
            z_coordinate: Z-coordinate for source and microphone placement.
                Defaults to 2.0.
            scale: Scaling factor for room visualization. Defaults to 1.3.
            source_color: BGR color tuple for source visualization.
                Defaults to (32, 84, 233).
            mic_color: BGR color tuple for microphone visualization.
                Defaults to (111, 33, 119).

        Raises:
            AssertionError: If room_object was not created from corners.
        """
        self.room_object = room_object
        assert (
            self.room_object.from_corners_flag
        ), "Graphics class requires a room created from corners."
        self.width = width
        self.height = height
        self.scale = scale
        self.img = np.ones((height, width, 3), np.uint8) * 255  # Create a white image
        self.default_img = self.img.copy()
        bbox3d = room_object.room.get_bbox()
        depth = bbox3d[2, 1] - bbox3d[2, 0]
        self.z_coordinate = min(
            max(z_coordinate, bbox3d[2, 0] + depth * 0.1), bbox3d[2, 1] - depth * 0.1
        )
        self.source_color = source_color
        self.mic_color = mic_color
        self.currently_dragging = (
            ""  # Placeholder. Variable to keep track of which point is being dragged
        )
        self.bbox = bbox3d[:2]  # Get the bounding box of the room in the x-y plane
        self.mean_point = self.bbox.mean(axis=1)
        self.bbox_width = self.bbox[0, 1] - self.bbox[0, 0]
        self.bbox_height = self.bbox[1, 1] - self.bbox[1, 0]
        self.room_dimension = max(self.bbox_width, self.bbox_height)
        self.frame_size = self.room_dimension * self.scale

        # Assuming the room is centered at (0, 0), the size is twice the maximum coordinate
        self.corners = self.build_reduced_polygon(
            room_object.corners, margin=0.01 * self.room_dimension
        )
        self.draw_room()
        self.microphone_coordinates = np.array(
            [
                self.mean_point[0] - self.bbox_width / 4,
                self.mean_point[1],
                self.z_coordinate,
            ]
        )
        self.source_coordinates = np.array(
            [
                self.mean_point[0] + self.bbox_width / 4,
                self.mean_point[1],
                self.z_coordinate,
            ]
        )

        self.microphone_coordinates[:2] = self.project_if_outside_room(
            self.microphone_coordinates[0],
            self.microphone_coordinates[1],
        )
        self.source_coordinates[:2] = self.project_if_outside_room(
            self.source_coordinates[0],
            self.source_coordinates[1],
        )

        self.draw_source_and_mic()

    def build_reduced_polygon(self, corners, margin):
        normals = []
        for i in range(len(corners)):
            start = corners[i]
            end = corners[(i + 1) % len(corners)]
            edge_vec = end - start
            edge_len = np.linalg.norm(edge_vec)
            if edge_len == 0:
                continue
            edge_unitvec = edge_vec / edge_len
            normal_vec = np.array([-edge_unitvec[1], edge_unitvec[0]])
            normals.append(normal_vec)
        reduced_corners = []
        for i in range(len(corners)):
            prev_normal = normals[i - 1]
            next_normal = normals[i]
            bisector = prev_normal + next_normal
            bisector_len = np.linalg.norm(bisector)
            if bisector_len == 0:
                continue
            reduced_corner = corners[i] + margin * bisector
            reduced_corners.append(reduced_corner)
        return np.array(reduced_corners)

    def _coordinates_to_pixels(self, x: float, y: float) -> tuple[int, int]:
        """Convert room coordinates to canvas pixel coordinates.

        Args:
            x: X coordinate in room space.
            y: Y coordinate in room space.

        Returns:
            Tuple of (pixel_x, pixel_y) in canvas coordinates.
        """

        pixel_x = int(
            (x - self.mean_point[0] + self.frame_size / 2)
            / self.frame_size
            * self.width
        )
        pixel_y = int(
            (y - self.mean_point[1] + self.frame_size / 2)
            / self.frame_size
            * self.height
        )
        return pixel_x, pixel_y

    def _pixels_to_coordinates(self, pixel_x: int, pixel_y: int) -> tuple[float, float]:
        """Convert canvas pixel coordinates to room coordinates.

        Args:
            pixel_x: X coordinate in canvas pixels.
            pixel_y: Y coordinate in canvas pixels.

        Returns:
            Tuple of (x, y) in room coordinates.
        """
        x = (
            (pixel_x / self.width) * self.frame_size
            - self.frame_size / 2
            + self.mean_point[0]
        )
        y = (
            (pixel_y / self.height) * self.frame_size
            - self.frame_size / 2
            + self.mean_point[1]
        )
        return x, y

    def project_point_to_nearest_wall(self, x: float, y: float) -> tuple[float, float]:
        """Project a point onto the nearest wall of the room.

        Args:
            x: X coordinate in room space.
            y: Y coordinate in room space.

        Returns:
            Tuple of (projected_x, projected_y) on the nearest wall.
        """
        min_distance = float("inf")
        nearest_point = (x, y)
        for i in range(len(self.corners)):
            start = self.corners[i]
            end = self.corners[(i + 1) % len(self.corners)]
            # Compute the projection of (x, y) onto the line segment defined by start and end
            line_vec = end - start
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                continue
            line_unitvec = line_vec / line_len
            point_vec = np.array([x, y]) - start
            t = np.dot(point_vec, line_unitvec)
            # Clamp t to the length of the segment
            t = max(0.01, min(line_len, 0.99 * t))

            projection = start + t * line_unitvec
            distance = np.linalg.norm(projection - np.array([x, y]))
            if distance < min_distance:
                min_distance = distance
                nearest_point = projection
        return nearest_point[0], nearest_point[1]

    def project_if_outside_room(self, x: float, y: float) -> tuple[float, float]:
        """Project point to nearest wall if outside room, otherwise return as-is.

        Args:
            x: X coordinate in room space.
            y: Y coordinate in room space.

        Returns:
            Tuple of (x, y) if inside room, or projected coordinates if outside.
        """
        is_inside_room = self.room_object.is_inside(np.array([x, y, self.z_coordinate]))
        if not is_inside_room:
            return self.project_point_to_nearest_wall(x, y)
        else:
            return x, y

    def draw_room(self):
        """Redraw the room layout with walls and legend.

        Creates a white canvas, draws room walls in black, and adds a legend
        showing microphone and source symbols in the top-right corner.
        """
        self.img = (
            np.ones((self.height, self.width, 3), np.uint8) * 255
        )  # Create a white image
        for i in range(len(self.corners)):
            start = self.corners[i]
            end = self.corners[(i + 1) % len(self.corners)]
            cv2.line(
                self.img,
                self._coordinates_to_pixels(start[0], start[1]),
                self._coordinates_to_pixels(end[0], end[1]),
                (0, 0, 0),
                2,
            )
        cv2.putText(
            self.img,
            "Press Esc. key to close",
            (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        # put a legend in the top right corner
        self.draw_mic_symbol(*self._pixels_to_coordinates(self.width - 210, 30))
        cv2.putText(
            self.img,
            "Microphone",
            (self.width - 170, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            1,
        )
        self.draw_source_symbol(*self._pixels_to_coordinates(self.width - 210, 70))
        cv2.putText(
            self.img,
            "Source",
            (self.width - 170, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            1,
        )
        self.default_img = self.img.copy()

    def draw_source_symbol(self, x: float, y: float):
        """Draw source symbol at specified room coordinates.

        Draws concentric circles at the given location using the configured
        source color.

        Args:
            x: X coordinate in room space.
            y: Y coordinate in room space.
        """
        cv2.circle(
            self.img,
            self._coordinates_to_pixels(x, y),
            4,
            self.source_color,
            -1,
        )
        cv2.circle(
            self.img,
            self._coordinates_to_pixels(x, y),
            8,
            self.source_color,
            3,
        )
        cv2.circle(
            self.img,
            self._coordinates_to_pixels(x, y),
            13,
            self.source_color,
            2,
        )

    def draw_mic_symbol(self, x: float, y: float):
        """Draw microphone symbol at specified room coordinates.

        Draws a rotated crosshair with circle at the given location using the
        configured microphone color.

        Args:
            x: X coordinate in room space.
            y: Y coordinate in room space.
        """
        pix_x, pix_y = self._coordinates_to_pixels(x, y)
        R = 13
        rot = np.array(
            [[np.sqrt(2) / 2, -np.sqrt(2) / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2]]
        )  # 45 degree rotation matrix
        line1 = np.array([[-R, 0], [R, 0]])  # Horizontal line
        line2 = np.array([[0, -R], [0, R]])  # Vertical line
        line1_rot = np.floor(line1 @ rot.T + np.array([pix_x, pix_y]))
        line2_rot = np.floor(line2 @ rot.T + np.array([pix_x, pix_y]))
        cv2.circle(
            self.img,
            self._coordinates_to_pixels(x, y),
            R,
            self.mic_color,
            2,
        )
        cv2.line(
            self.img,
            (int(line1_rot[0, 0]), int(line1_rot[0, 1])),
            (int(line1_rot[1, 0]), int(line1_rot[1, 1])),
            self.mic_color,
            2,
        )
        cv2.line(
            self.img,
            (int(line2_rot[0, 0]), int(line2_rot[0, 1])),
            (int(line2_rot[1, 0]), int(line2_rot[1, 1])),
            self.mic_color,
            2,
        )

    def draw_source_and_mic(self):
        """Redraw source and microphone symbols on current canvas.

        Updates the canvas by drawing both source and microphone symbols at
        their current coordinates over the room background.
        """
        self.img = self.default_img.copy()
        self.draw_mic_symbol(
            self.microphone_coordinates[0], self.microphone_coordinates[1]
        )
        self.draw_source_symbol(self.source_coordinates[0], self.source_coordinates[1])

    def _update_at_click(self, x, y, max_dist_percentage=0.05):
        """Update source or microphone position on mouse click.

        Selects the nearest object (source or microphone) within a distance
        threshold and marks it as being dragged.

        Args:
            x: Mouse X position in pixel coordinates.
            y: Mouse Y position in pixel coordinates.
            max_dist_percentage: Maximum distance threshold as percentage of
                room diagonal. Defaults to 0.05 (5%).
        """
        x_coord, y_coord = self._pixels_to_coordinates(x, y)
        x_coord, y_coord = self.project_if_outside_room(x_coord, y_coord)
        distance_to_mic = np.linalg.norm(
            np.array([x_coord, y_coord]) - self.microphone_coordinates[:2]
        )
        distance_to_source = np.linalg.norm(
            np.array([x_coord, y_coord]) - self.source_coordinates[:2]
        )
        max_dist = max_dist_percentage * max(self.bbox_width, self.bbox_height)
        if distance_to_mic < distance_to_source and distance_to_mic < max_dist:
            self.microphone_coordinates = np.array(
                [x_coord, y_coord, self.z_coordinate]
            )
            self.currently_dragging = "mic"
        elif distance_to_source <= distance_to_mic and distance_to_source < max_dist:
            self.source_coordinates = np.array([x_coord, y_coord, self.z_coordinate])
            self.currently_dragging = "source"
        else:
            self.currently_dragging = ""

        self.draw_source_and_mic()

    def _update_at_move(self, x, y):
        """Update position of dragged object during mouse movement.

        Updates the currently dragged object (source or microphone) to follow
        the mouse cursor.

        Args:
            x: Mouse X position in pixel coordinates.
            y: Mouse Y position in pixel coordinates.
        """
        x_coord, y_coord = self._pixels_to_coordinates(x, y)
        x_coord, y_coord = self.project_if_outside_room(x_coord, y_coord)
        if self.currently_dragging == "mic":
            self.microphone_coordinates = np.array(
                [x_coord, y_coord, self.z_coordinate]
            )
        elif self.currently_dragging == "source":
            self.source_coordinates = np.array([x_coord, y_coord, self.z_coordinate])

        self.draw_source_and_mic()

    def draw_point_callback(self, event, x: int, y: int, flags, param):
        """OpenCV mouse event callback for interactive point placement.

        Handles mouse events to allow dragging source and microphone positions
        within the room.

        Args:
            event: OpenCV event type (e.g., cv2.EVENT_LBUTTONDOWN).
            x: Mouse X position in pixel coordinates.
            y: Mouse Y position in pixel coordinates.
            flags: OpenCV event flags (e.g., cv2.EVENT_FLAG_LBUTTON).
            param: Additional parameter passed by OpenCV (unused).
        """
        # x and y are in pixel coordinates
        # first convert to room coordinates to check if the point is inside the room

        if event == cv2.EVENT_LBUTTONDOWN:
            self._update_at_click(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            self._update_at_move(x, y)

    def show(self):
        """Display the current frame and handle keyboard input.

        Shows the canvas and processes keyboard commands:
        - ESC: Closes the window and raises KeyboardInterrupt
        - 'a': Prints current microphone and source coordinates
        """
        cv2.imshow("Graphics", self.img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key to exit
            raise KeyboardInterrupt
        elif k == ord("a"):
            print("Microphone coordinates:", self.microphone_coordinates)
            print("Source coordinates:", self.source_coordinates)


class GraphicsContextManager(Graphics):
    """Context manager for interactive graphics with automatic cleanup.

    Extends Graphics to provide context manager interface for proper resource
    handling of OpenCV windows.
    """

    def __init__(self, *args, disable_interactive=False, **kwargs):
        """Initialize graphics context manager.

        Args:
            *args: Positional arguments passed to Graphics.__init__.
            disable_interactive: If True, disables mouse interaction.
                Defaults to False.
            **kwargs: Keyword arguments passed to Graphics.__init__.
        """
        super().__init__(*args, **kwargs)
        self.disable_interactive = disable_interactive

    def __enter__(self):
        """Enter context manager: create window and set up mouse callback.

        Returns:
            Self (GraphicsContextManager instance).
        """
        cv2.namedWindow("Graphics")
        if not self.disable_interactive:
            cv2.setMouseCallback("Graphics", self.draw_point_callback)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager: close all OpenCV windows.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_value: Exception value if an exception occurred.
            traceback: Exception traceback if an exception occurred.
        """
        cv2.destroyAllWindows()


def test_graphics():
    """Interactive test of graphics visualization and interaction.

    Demonstrates the GraphicsContextManager by creating a demo room and
    allowing interactive manipulation of source and microphone positions.
    Continues until user presses ESC key.
    """
    room_object = make_demo_room()
    with GraphicsContextManager(room_object) as g:
        while True:
            try:
                g.show()
            except KeyboardInterrupt:
                break
