from ..utils.utils import is_inside, wall_list_to_triangles
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pyroomacoustics as pra
import warnings


def compute_log_conductance_from_admittances(
    admittances: list[complex] | npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Compute logarithmic conductance from surface admittances.

    Uses an analytical expression derived from integration by parts to convert
    boundary admittances to logarithmic conductance terms used in SWFT calculations.
    Reference: "Statistical Wave Field Theory" by R. Badeau.

    Args:
        admittances: Complex admittances as list or array. List will be converted
            to numpy array.

    Returns:
        Array of complex logarithmic conductances with same shape as input.
    """
    if isinstance(admittances, list):
        admittances = np.array(admittances)
    complex_admittance_ratio = (1 + admittances) / (1 - admittances)
    # Analytic expression of integral(0,1, ln((u + admittances)/(u-admittances))u du)
    # where u is the integration variable.
    log_conductance = (
        np.log(complex_admittance_ratio)
        - admittances**2 * np.log(-complex_admittance_ratio)
        + 2 * admittances
    )
    # This expression is derived using integration by parts
    # and the properties of logarithmic functions.
    # See "Statistical Wave Field Theory" by R. Badeau for more details.
    return log_conductance


def admittances_to_absorption(
    admittances: list[complex] | npt.NDArray[np.complex64],
) -> npt.NDArray:
    """Convert surface admittances to absorption coefficients.

    Computes absorption coefficients from complex admittances using the
    relationship: absorption = 1 - exp(-2 * ln(conductance).real).

    Args:
        admittances: Complex admittances as list or array.

    Returns:
        Array of absorption coefficients (real values between 0 and 1).
    """
    log_conductances = compute_log_conductance_from_admittances(admittances)
    absorptions = 1 - np.exp(-2 * log_conductances.real)
    return absorptions


def make_demo_room(verbose: bool = False):
    """Create a sample SWFT room for testing and demonstrations.

    Generates a polyhedrally-shaped room with different material properties
    on walls, ceiling, and floor. Materials include curtains, wood panels,
    carpet, and rigid surfaces with frequency-dependent absorption.

    Args:
        verbose: If True, print progress messages during room creation.

    Returns:
        SWFTRoom object ready for RIR synthesis and analysis.
    """
    rigid = [0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06]
    wood = [0.18, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06]
    carpet = [0.07, 0.31, 0.49, 0.81, 0.66, 0.54, 0.43]
    curtains = [0.3, 0.45, 0.65, 0.56, 0.59, 0.71, 0.8]

    corners = np.array(
        [
            [10, 0],
            [40, 20],
            [70, 0],
            [70, 30],
            [10, 60],
            [20, 40],
            [0, 30],
            [20, 20],
        ]
    )
    height = 30
    room_absorption = {
        "list_walls": [
            curtains,
            wood,
            wood,
            curtains,
            curtains,
            wood,
            curtains,
            curtains,
        ],
        "ceiling": rigid,
        "floor": carpet,
    }
    return SWFTRoom.from_corners(
        corners,
        height,
        absorptions=room_absorption,
        disable_tqdm=not verbose,
    )


class SWFTRoom:
    def __init__(
        self,
        room: pra.room.Room,
        admittances: dict[str, npt.NDArray[np.complex64]] = {},
        voxel_size: float = -1,
        subsample_factor: int = -1,
        frequency_div_mode: str = "octave",
        disable_tqdm: bool = True,
    ):
        """Initialize a SWFT room from a pyroomacoustics room with optional admittances.

        Computes spatial mesh, wave number deformations, and variance field needed
        for Statistical Wave Field Theory calculations. Automatically sets voxel size
        and subsample factor if not provided.

        Args:
            room: Pyroomacoustics room object with geometry and absorption.
            admittances: Dict mapping wall names to complex admittance arrays.
                If empty, uses absorption coefficients instead. Defaults to empty dict.
            voxel_size: Spatial discretization in meters. If <= 0, auto-computed as
                5% of smallest room dimension. Defaults to -1.
            subsample_factor: Factor to reduce mesh resolution for variance computation.
                If <= 0, auto-computed for balance between speed and accuracy. Defaults to -1.
            frequency_div_mode: Frequency axis type - "octave" (recommended) or "linear"
                (deprecated). Defaults to "octave".
            disable_tqdm: If True, suppress progress bar during calculations. Defaults to True.
        """
        self.room = room
        # Flag to indicate whether the room was created from corners or not.
        self.from_corners_flag = False
        # Placeholder for the corners of the room,
        # to be used in the Graphics class if the room was created from corners.
        self.corners = np.array([])
        if voxel_size <= 0:
            # Default voxel size is 5% of the smallest room dimension
            bbox = self.room.get_bbox()
            room_dim = bbox[:, 1] - bbox[:, 0]
            self.voxel_size = min(room_dim) / 20
            print(
                f"Voxel size automatically set to {self.voxel_size:.2f} meters, "
                f"which is 5% of the smallest room dimension."
            )
        else:
            self.voxel_size = voxel_size
        if subsample_factor <= 0:
            bbox = self.room.get_bbox()
            mean_room_dim = np.mean(bbox[:, 1] - bbox[:, 0])
            smallest_room_dim = min(bbox[:, 1] - bbox[:, 0])
            n_voxels_per_dim_mean = np.floor(mean_room_dim / self.voxel_size)
            n_voxels_on_smallest_dim = np.floor(smallest_room_dim / self.voxel_size)
            maximum_subsample_factor = np.floor(n_voxels_on_smallest_dim / 3)
            mean_subsample_factor = np.floor(n_voxels_per_dim_mean / 8)
            self.subsample_factor = np.min(
                [maximum_subsample_factor, mean_subsample_factor]
            )
            print(
                f"Subsample factor automatically set to {self.subsample_factor} "
                "to ensure a reasonable computation time while keeping a good accuracy."
            )
        else:
            self.subsample_factor = subsample_factor
        self.frequency_div_mode = frequency_div_mode
        self.disable_tqdm = disable_tqdm
        self.admittances = admittances
        self.sound_speed = room.c  # Speed of sound in m/s
        # Assuming all walls have the same
        # number of frequency bands for the absorption coefficients.
        self.num_freq_bands = len(room.walls[0].absorption)
        if frequency_div_mode == "octave":
            self.center_freqs = self.room.octave_bands.centers
            # Center frequencies for the absorption coefficients (e.g., octave bands)
        elif frequency_div_mode == "linear":
            warnings.warn(
                "Linear frequency division is deprecated as it is not natively supported by pyroomacoustics."
                "Beware of potential inconsistencies between the absorption coefficients and the center frequencies."
                "It is recommended to use octave frequency division instead.",
                DeprecationWarning,
            )
            # Center frequencies for the absorption coefficients (e.g., octave bands)
            self.center_freqs = (
                room.fs / 2 * np.linspace(0, 1, 2 * self.num_freq_bands + 1)[1::2]
            )

        if len(admittances.keys()) > 0:
            self.use_admittances = True
            self._check_admittance_dict_format(admittances)
        else:
            self.use_admittances = False

        self.triangulated_surface = wall_list_to_triangles(
            [self.room.walls[i].corners.T for i in range(len(self.room.walls))]
        )

        self.kappa = self._compute_kappa()  # wave number deformation
        self.mesh, self.is_inside_mask = self._compute_mesh()
        self.subsampled_mesh, _ = self._compute_mesh(
            subsample_factor=self.subsample_factor, return_mask=False
        )
        self.variance_on_subsampled_mesh = self._compute_variance_on_mesh()
        self.interpolator = self._define_interpolator()

    def _check_admittance_dict_format(
        self, admittances: dict[str, npt.NDArray[np.complex64]]
    ) -> None:
        """Validate admittance dictionary format and consistency with room.

        Ensures all walls have admittances, correct data types, matching frequency
        bands, and that converted absorption coefficients match room definitions.

        Args:
            admittances: Dict mapping wall names to complex admittance arrays.

        Raises:
            AssertionError: If format is invalid or admittances don't match room.
        """
        keys = list(admittances.keys())
        for wall in self.room.walls:
            assert (
                wall.name in keys
            ), f"Admittance for wall {wall.name} is missing in the admittances dictionary."
            admittance = admittances[wall.name]
            assert isinstance(
                admittance, np.ndarray
            ), f"Admittance for wall {wall.name} must be a numpy array."
            assert (
                len(admittance) == self.num_freq_bands
            ), f"Admittance for wall {wall.name} must have the same number of frequency bands as the absorption coefficients."
            absorption = admittances_to_absorption(admittance)
            assert np.all(
                np.isclose(absorption, wall.absorption, rtol=1e-5)
            ), f"Admittance for wall {wall.name} does not match the absorption coefficients defined in the room. Please check the values."

    def _compute_kappa(self):
        """Compute wave number deformation accounting for boundary effects.

        Calculates complex wave numbers that incorporate the effect of
        room boundaries on wave propagation using admittance or absorption data.

        Returns:
            Complex array of deformed wave numbers, one per frequency band.
        """

        kappa = np.array(
            self.center_freqs / self.sound_speed, dtype=np.complex64
        )  # Start with the free-field wave number
        lbd = 1 / self.room.get_volume()

        if self.use_admittances:
            for wall in self.room.walls:
                admit = self.admittances[wall.name]
                log_conductance = compute_log_conductance_from_admittances(admit)
                kappa += 1j * lbd * wall.area() * log_conductance / (8 * np.pi)
        else:
            for wall in self.room.walls:
                absorb = wall.absorption
                kappa += (
                    1j * lbd * wall.area() * np.log(1 / (1 - absorb)) / (16 * np.pi)
                )

        return kappa

    def _compute_mesh(self, return_mask: bool = True, subsample_factor: int = 1):
        """Discretize room volume into a 3D mesh of points.

        Creates a regular grid of points covering the room with optional padding.
        Optionally returns a boolean mask indicating which points are inside the room.

        Args:
            return_mask: If True, compute and return inside/outside mask. Defaults to True.
            subsample_factor: Spacing multiplier to reduce mesh density. Defaults to 1.

        Returns:
            Tuple of (mesh, mask) where mesh is shape (3, nx, ny, nz) and mask is
            boolean array of same spatial dimensions (empty if return_mask=False).
        """
        # This is necessary to compute the eigenmodes of the room,
        # which are needed to compute the wave number deformation.
        # The mesh can be computed using the pyroomacoustics Room class,
        # which has a method to compute the mesh of the room.
        # The mesh can be computed using the finite element method,
        # which is implemented in the pyroomacoustics Room class.
        bbox = self.room.get_bbox()
        step_size = self.voxel_size * subsample_factor
        origin, end = bbox[:, 0], bbox[:, 1]
        X, Y, Z = (
            np.arange(
                origin[0] - step_size / 2,
                end[0] + step_size * 3 / 2,
                step_size,
            ),
            np.arange(
                origin[1] - step_size / 2,
                end[1] + step_size * 3 / 2,
                step_size,
            ),
            np.arange(
                origin[2] - step_size / 2,
                end[2] + step_size * 3 / 2,
                step_size,
            ),
        )
        mesh = np.stack(np.meshgrid(X, Y, Z, indexing="ij"))

        if return_mask:
            flattened_mesh = mesh.reshape(
                3, -1
            ).T  # Reshape to (N, 3) for is_inside method
            mask = self.is_inside(flattened_mesh).reshape(mesh.shape[1:])
        else:
            mask = np.array([])
        return mesh, mask

    def _compute_variance_on_mesh(self) -> npt.NDArray:
        """Compute wave field variance at each mesh point using SWFT formulation.

        Calculates the spatial distribution of acoustic energy using the deformed
        wave numbers and room volume. Variance is computed on the subsampled mesh
        for efficiency, then interpolated as needed. Expression from R. Badeau SWFT theory.

        Returns:
            Array of shape (subsampled_mesh_spatial_dims, n_freq_bands) containing
            variance values (float32).
        """
        Nfreqs = len(self.center_freqs)
        sinhc = lambda x: np.where(x == 0, 1, np.sinh(x) / x)

        points = self.mesh[:, self.is_inside_mask].reshape(3, -1).T
        # Reshape to (N, 3) for computation

        subsampled_points = self.subsampled_mesh.reshape(3, -1).T
        volume = self.room.get_volume()

        cst = (
            self.sound_speed
            * self.voxel_size**3
            * self.center_freqs**2
            / (
                volume**2
                * np.pi
                * (4 * self.center_freqs**2 + self.sound_speed**2 * self.kappa.imag**2)
            )
        )
        # The constant differs from the one in the paper by a factor of 1/(2*pi*f)**2,
        # because we compute the impulse response and not the source response.

        print("Computing the variance on the mesh...")
        variance = np.zeros((subsampled_points.shape[0], Nfreqs), dtype=np.float32)
        for i in tqdm(range(subsampled_points.shape[0]), disable=self.disable_tqdm):
            distances = np.linalg.norm(subsampled_points[i] - points, axis=-1)
            variance[i, :] = cst * np.sum(
                sinhc(4 * np.pi * self.kappa.imag[None, :] * distances[:, None]), axis=0
            )
        variance_on_mesh = variance.reshape((*self.subsampled_mesh.shape[1:], Nfreqs))
        return variance_on_mesh

    def _define_interpolator(self) -> RegularGridInterpolator:
        """Create cubic interpolator for smooth spatial variance field.

        Builds a 3D cubic spline interpolator on the subsampled variance mesh.
        Enables fast queries of variance at arbitrary points inside the room.

        Returns:
            RegularGridInterpolator instance ready to evaluate variance at any point.

        Raises:
            Raises np.nan if query point is outside the room bounds.
        """
        X, Y, Z = self.subsampled_mesh
        x = X[:, 0, 0]
        y = Y[0, :, 0]
        z = Z[0, 0, :]
        interpolator = RegularGridInterpolator(
            (x, y, z),
            self.variance_on_subsampled_mesh,
            method="cubic",
            bounds_error=True,
            fill_value=np.nan,
        )
        return interpolator

    def is_inside(self, points: npt.NDArray) -> npt.NDArray[np.bool_]:
        """Check if points are inside the room using triangulation.

        Uses ray-casting algorithm on triangulated room surfaces to determine
        which points lie within the room volume.

        Args:
            points: Single point as (3,) array or batch as (N, 3) array with
                coordinates (x, y, z).

        Returns:
            Boolean array of shape (N,) or scalar bool for single point.

        Raises:
            ValueError: If points are not shape (3,) or (N, 3).
        """
        if points.ndim == 1:
            points = np.expand_dims(points, 0)
        elif points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be of shape (3,) or (N, 3)")
        return is_inside(self.triangulated_surface, points)

    @classmethod
    def from_corners(
        cls,
        corners: list | npt.NDArray,
        height: float,
        admittances: dict = {},
        absorptions: dict = {},
        voxel_size: float = -1,
        subsample_factor: int = -1,
        disable_tqdm: bool = True,
    ):
        """Create a SWFT room from 2D floor plan corners and height.

        Builds a 3D room from a 2D floor plan by extruding to specified height.
        Materials are specified via admittances (complex) or absorption coefficients
        (real) for walls, ceiling, and floor. Admittances provide more detailed
        acoustic modeling but absorptions are more commonly available.

        Args:
            corners: 2D floor plan vertices as (N, 2) array or list of [x, y] coords.
            height: Room height in meters.
            admittances: Dict with keys "list_walls", "ceiling", "floor" containing
                complex admittances per octave band. Mutually exclusive with absorptions.
            absorptions: Dict with keys "list_walls", "ceiling", "floor" containing
                real absorption coefficients per octave band. Defaults to empty dict.
            voxel_size: Spatial discretization. If <= 0, auto-computed. Defaults to -1.
            subsample_factor: Mesh reduction factor. If <= 0, auto-computed. Defaults to -1.
            disable_tqdm: Suppress progress messages. Defaults to True.

        Returns:
            Initialized SWFTRoom object ready for analysis.

        Raises:
            ValueError: If both or neither admittances and absorptions are provided.
            AssertionError: If admittance/absorption counts don't match frequency bands.
        """
        # First, we convert admittances to absorptions if admittances are provided
        if len(admittances.keys()) > 0 and len(absorptions.keys()) == 0:
            use_admittances = True
            fs = 125 * 2 ** len(admittances["ceiling"])
            # Sampling rate must be at least
            # twice the highest center frequency of the absorption coefficients.
        elif len(absorptions.keys()) > 0 and len(admittances.keys()) == 0:
            use_admittances = False
            fs = 125 * 2 ** len(absorptions["ceiling"])
        else:
            raise ValueError(
                "Either admittances or absorptions must be provided, but not both."
            )
        octave_bands = pra.acoustics.OctaveBandsFactory(fs=fs)
        center_freqs = octave_bands.centers
        if isinstance(corners, list):
            corners = np.array(corners)

        if use_admittances:
            assert len(admittances["ceiling"]) == len(
                center_freqs
            ), "The number of ceiling admittances must match the number of center frequencies."
            assert len(admittances["floor"]) == len(
                center_freqs
            ), "The number of floor admittances must match the number of center frequencies."
            absorptions = {
                "list_walls": [],
                "ceiling": admittances_to_absorption(
                    np.array(admittances["ceiling"], dtype=np.complex64)
                ),
                "floor": admittances_to_absorption(
                    np.array(admittances["floor"], dtype=np.complex64)
                ),
            }

            for wall_admittance in admittances["list_walls"]:
                assert len(wall_admittance) == len(
                    center_freqs
                ), "The number of wall admittances must match the number of center frequencies."
                admittance_array = np.array(wall_admittance, dtype=np.complex64)
                absorptions["list_walls"].append(
                    admittances_to_absorption(admittance_array)
                )

        # Then, we define the materials for the room using the absorptions
        materials = []
        assert absorptions is not None, "Absorptions must be defined to build the room."

        for id, wall_absorption in enumerate(absorptions["list_walls"]):
            materials.append(
                pra.parameters.Material(
                    {
                        "description": f"wall_{id}",
                        "coeffs": wall_absorption,
                        "center_freqs": center_freqs,
                    }
                )
            )
        ceiling_material = pra.parameters.Material(
            {
                "description": "ceiling",
                "coeffs": absorptions["ceiling"],
                "center_freqs": center_freqs,
            }
        )
        floor_material = pra.parameters.Material(
            {
                "description": "floor",
                "coeffs": absorptions["floor"],
                "center_freqs": center_freqs,
            }
        )

        room = pra.room.Room.from_corners(
            corners.T,
            fs=fs,
            max_order=10,
            materials=materials,
        )
        room.extrude(
            height,
            materials={"floor": floor_material, "ceiling": ceiling_material},
        )

        # Create a mapping from wall names to their corresponding admittances,
        # to be used in the SWFTRoom class.
        # The wall names are defined by pyroomacoustics
        # as "0", "1", ..., "N-1" for the N walls defined by the corners,
        # and "ceiling" and "floor" for the ceiling and floor.
        admittances_map = {}
        if use_admittances:
            for wall in room.walls:
                wall_name = wall.name
                if wall_name == "ceiling" or wall_name == "floor":
                    admittances_map[wall_name] = np.array(admittances[wall_name])
                else:
                    wall_id = int(wall_name)
                    admittances_map[wall_name] = np.array(
                        admittances["list_walls"][wall_id]
                    )

        room_object = cls(
            room,
            admittances_map,
            voxel_size,
            subsample_factor,
            frequency_div_mode="octave",
            disable_tqdm=disable_tqdm,
        )
        room_object.from_corners_flag = True
        room_object.corners = corners

        return room_object

    def get_frequency_response_at_point(self, point: npt.NDArray | list) -> npt.NDArray:
        """Compute frequency response (modal decomposition) at a spatial point.

        Interpolates the precomputed variance field at the given point and extracts
        the frequency response magnitude across all frequency bands.

        Args:
            point: 3D position (x, y, z) as array or list.

        Returns:
            Array of frequency response values, one per frequency band.
        """
        variance_at_point = self.interpolator(point)
        # The frequency response is proportional to the square root of the variance.
        frequency_response = np.sqrt(variance_at_point)

        return frequency_response[0, :]

    def eyring_formula(self) -> float:
        """Compute overall reverberation time using Eyring's formula.

        Applies the more accurate Eyring formula (vs. Sabine) that accounts for
        non-linear absorption effects. Result is in seconds.

        Returns:
            RT60 in seconds (average across frequency bands).
        """
        V = self.room.get_volume()
        A = 0
        for wall in self.room.walls:
            absorb = wall.absorption
            # Eyring's formula uses the logarithm of the absorption coefficient
            A += wall.area() * np.log(1 / (1 - absorb))
        return 24 * np.log(10) * V / (self.sound_speed * A)

    def get_rt60_profile(self) -> npt.NDArray:
        """Compute RT60 reverberation time profile across frequency bands.

        Uses the deformed wave numbers (kappa) to estimate frequency-dependent
        reverberation time based on SWFT theory. Returns values in seconds.

        Returns:
            Array of RT60 values per frequency band.
        """
        return 3 * np.log(10) / (self.kappa.imag * 2 * np.pi * self.sound_speed)


def test_spatial_model():
    room_object = make_demo_room(verbose=True)
    print(dir(room_object.room.walls[0]))
    room_object.room.plot()
    plt.show()
    plt.plot(
        room_object.center_freqs,
        room_object.get_rt60_profile(),
        label="RT_60 profile from SWFT",
    )
    plt.plot(
        room_object.center_freqs,
        room_object.eyring_formula(),
        label="RT_60 profile with Eyring formula",
    )
    plt.legend()

    frequency_response_at_point = room_object.get_frequency_response_at_point(
        [2, np.pi, np.sqrt(2)]
    )

    plt.figure()
    plt.plot(room_object.center_freqs, frequency_response_at_point)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Frequency response (arbitrary units)")
    plt.title("Frequency response at point [2, pi, sqrt(2)]")
    plt.show()

    # plot a 2D slice of the variance field at freq_idx
    bbox = room_object.room.get_bbox()
    x = np.arange(bbox[0, 0], bbox[0, 1] + 0.1, 0.1)
    y = np.arange(bbox[1, 0], bbox[1, 1] + 0.1, 0.1)
    z = np.array([(bbox[2, 1] - bbox[2, 0]) / 2])
    mesh_2d = np.stack(np.meshgrid(x, y, z, indexing="ij"))
    freq_idx = 3
    plt.figure()
    plt.imshow(
        room_object.interpolator(mesh_2d.reshape(3, -1).T).reshape(
            mesh_2d.shape[1:3] + (-1,)
        )[:, :, freq_idx],
        origin="lower",
    )
    plt.title("2D slice of the room's mesh at z = height/2")
    plt.show()


def test_triangulation():
    room_object = make_demo_room(verbose=True)
    wall_list = [
        room_object.room.walls[i].corners.T for i in range(len(room_object.room.walls))
    ]
    triangles = wall_list_to_triangles(wall_list)
    plt.figure()
    ax = plt.axes(projection="3d")
    for triangle in triangles:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal = 10 * normal / np.linalg.norm(normal)
        mean_point = np.mean(triangle, axis=0)
        for i in range(3):
            ax.plot(
                [triangle[i, 0], triangle[(i + 1) % 3, 0]],
                [triangle[i, 1], triangle[(i + 1) % 3, 1]],
                [triangle[i, 2], triangle[(i + 1) % 3, 2]],
                color="b",
            )
        ax.quiver(
            mean_point[0],
            mean_point[1],
            mean_point[2],
            normal[0],
            normal[1],
            normal[2],
            length=0.5,
            color="r",
        )
    plt.title("Triangulation of the room's walls")
    plt.show()


def test_is_inside():
    room_object = make_demo_room(verbose=True)
    wall_list = [
        room_object.room.walls[i].corners.T for i in range(len(room_object.room.walls))
    ]
    triangles = wall_list_to_triangles(wall_list)
    bbox = room_object.room.get_bbox()
    # print(bbox)
    # point = np.array([[30, 30, 5]], dtype=np.float32)
    # print(is_inside(triangles, point))
    x = np.arange(bbox[0, 0] - 1, bbox[0, 1] + 3, 2)
    y = np.arange(bbox[1, 0] - 1, bbox[1, 1] + 3, 2)
    z = np.arange(bbox[2, 0] - 1, bbox[2, 1] + 3, 2)
    mesh_3d = np.stack(np.meshgrid(x, y, z))
    mask = is_inside(triangles, mesh_3d.reshape(3, -1).T).reshape(mesh_3d.shape[1:])
    flattened_mask = mask.reshape(-1)
    flattened_mesh = mesh_3d.reshape(3, -1).T
    inside_points = flattened_mesh[flattened_mask]
    plt.plot()
    ax = plt.axes(
        projection="3d",
        xlim=(bbox[0, 0] - 1, bbox[0, 1] + 3),
        ylim=(bbox[1, 0] - 1, bbox[1, 1] + 3),
        zlim=(bbox[2, 0] - 1, bbox[2, 1] + 3),
        box_aspect=(
            bbox[0, 1] - bbox[0, 0],
            bbox[1, 1] - bbox[1, 0],
            bbox[2, 1] - bbox[2, 0],
        ),
    )

    ax.scatter(
        inside_points[:, 0],
        inside_points[:, 1],
        inside_points[:, 2],
        color="g",
        label="Points inside the room",
    )
    ax.legend()
    plt.title("Points inside the room according to the is_inside function")
    plt.show()
