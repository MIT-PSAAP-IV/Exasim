#!/usr/bin/env pvpython
"""Render q-criterion isosurfaces from one VTU/PVTU file."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from paraview.simple import (  # type: ignore
    Calculator,
    ColorBy,
    Contour,
    CreateView,
    Clip,
    ExtractSurface,
    GenerateSurfaceNormals,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    GetScalarBar,
    Render,
    ResetSession,
    SaveScreenshot,
    Show,
    Threshold,
    UpdatePipeline,
    XMLPartitionedUnstructuredGridReader,
    XMLUnstructuredGridReader,
)


SUPPORTED_OUTPUT_EXTENSIONS = {
    ".png",
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
    ".bmp",
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Render q-criterion isosurfaces from one VTU/PVTU file and color "
            "them by velocity."
        )
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input .vtu or .pvtu dataset.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("qcrit.png"),
        help="Output image filename. Default: qcrit.png",
    )
    parser.add_argument(
        "--qcrit-field",
        default="qcrit",
        help='Point-data field used for isosurfaces. Default: "qcrit".',
    )
    parser.add_argument(
        "--velocity-field",
        default="u",
        help='Point-data field used for coloring. Default: "u".',
    )
    parser.add_argument(
        "--qcrit-values",
        nargs="+",
        type=float,
        default=[20.0, 40.0, 60.0, 80.0, 100.0,
                 120.0, 140.0, 160.0, 180.0, 200.0],
        help="Q-criterion isosurface values. Default: 20 40 ... 200.",
    )
    parser.add_argument(
        "--range",
        dest="scalar_range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Velocity color range. If omitted, use the data range.",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=(3200, 2400),
        metavar=("WIDTH", "HEIGHT"),
        help="Output resolution before optional cropping. Default: 3200 2400.",
    )
    parser.add_argument(
        "--camera-position",
        nargs=3,
        type=float,
        default=(-1.0, 1.0, 2.0),
        metavar=("X", "Y", "Z"),
        help="Camera position. Default: -1 1 2.",
    )
    parser.add_argument(
        "--camera-view-up",
        nargs=3,
        type=float,
        default=(0.28, 0.0, -1.0),
        metavar=("X", "Y", "Z"),
        help="Camera view-up direction. Default: 0.28 0 -1.",
    )
    parser.add_argument(
        "--camera-focal-point",
        nargs=3,
        type=float,
        default=(0.55, 0.0, 0.05),
        metavar=("X", "Y", "Z"),
        help=(
            "Camera focal point applied after ResetCamera. Default: "
            "0.55 0 0.05, appropriate for unit-chord airfoil data."
        ),
    )
    parser.add_argument(
        "--camera-zoom",
        type=float,
        default=2.15,
        help="Camera zoom factor after ResetCamera. Default: 2.15.",
    )
    parser.add_argument(
        "--camera-pan",
        nargs=2,
        type=float,
        default=(0.0, 0.0),
        metavar=("RIGHT", "UP"),
        help=(
            "Pan after ResetCamera in fractions of the parallel-view height. "
            "Positive values move the camera right/up. Default: 0 0."
        ),
    )
    parser.add_argument(
        "--background-color",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("R", "G", "B"),
        help="Background color in [0, 1]. Default: 1 1 1.",
    )
    parser.add_argument(
        "--show-colorbar",
        action="store_true",
        help="Show the velocity colorbar. Hidden by default.",
    )
    parser.add_argument(
        "--show-geometry",
        action="store_true",
        help=(
            "Also render the raw dataset surface. Use only when the dataset "
            "does not include a far-field outer boundary."
        ),
    )
    parser.add_argument(
        "--airfoil-surface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render a clipped near-airfoil surface layer so the full upper "
            "and lower airfoil surfaces are visible. Default: enabled."
        ),
    )
    parser.add_argument(
        "--airfoil-bounds",
        nargs=6,
        type=float,
        default=(-0.08, 1.08, -0.25, 0.25, -0.01, 0.11),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help=(
            "Bounding box used for the clipped airfoil surface. Default: "
            "-0.08 1.08 -0.25 0.25 -0.01 0.11."
        ),
    )
    parser.add_argument(
        "--airfoil-normal-z-max",
        type=float,
        default=0.2,
        help=(
            "Maximum abs(spanwise normal component) retained for the airfoil "
            "surface layer. This removes z-plane slices. Default: 0.2."
        ),
    )
    parser.add_argument(
        "--transparent-background",
        action="store_true",
        help="Save with a transparent background.",
    )
    parser.add_argument(
        "--tight-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop whitespace around rendered pixels. Default: enabled.",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=18,
        help="Padding in pixels for --tight-crop. Default: 18.",
    )
    parser.add_argument(
        "--right-crop-fraction",
        type=float,
        default=0.72,
        help=(
            "After tight cropping, keep this left fraction of the image to "
            "truncate the downstream wake. Use 1.0 to keep the full wake. "
            "Default: 0.72."
        ),
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line options."""
    if not args.input_file.is_file():
        raise FileNotFoundError(f"Input file does not exist: {args.input_file}")

    if args.input_file.suffix.lower() not in (".vtu", ".pvtu"):
        raise ValueError("Input file must have extension .vtu or .pvtu.")

    width, height = args.resolution
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")

    if args.camera_zoom <= 0.0:
        raise ValueError("Camera zoom must be positive.")

    if args.crop_padding < 0:
        raise ValueError("Crop padding must be nonnegative.")

    if args.right_crop_fraction <= 0.0 or args.right_crop_fraction > 1.0:
        raise ValueError("Right crop fraction must be in the interval (0, 1].")

    if not args.qcrit_values:
        raise ValueError("At least one q-criterion isosurface value is required.")

    if any(value < 0.0 or value > 1.0 for value in args.background_color):
        raise ValueError("Background color entries must be in [0, 1].")

    xmin, xmax, ymin, ymax, zmin, zmax = args.airfoil_bounds
    if xmax <= xmin or ymax <= ymin or zmax <= zmin:
        raise ValueError("Airfoil bounds must satisfy max > min in each axis.")

    if args.airfoil_normal_z_max < 0.0 or args.airfoil_normal_z_max > 1.0:
        raise ValueError("Airfoil normal-z threshold must be in [0, 1].")

    if args.scalar_range is not None:
        scalar_minimum, scalar_maximum = args.scalar_range
        if scalar_maximum <= scalar_minimum:
            raise ValueError(
                "Scalar range maximum must be greater than its minimum."
            )

    extension = args.output.suffix.lower()
    if extension not in SUPPORTED_OUTPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_EXTENSIONS))
        raise ValueError(
            f"Unsupported output extension '{extension}'. "
            f"Supported extensions: {supported}"
        )


def load_dataset(dataset_path: Path) -> Any:
    """Create the correct ParaView reader for a VTU or PVTU dataset."""
    absolute_path = str(dataset_path.resolve())
    extension = dataset_path.suffix.lower()
    if extension == ".pvtu":
        reader = XMLPartitionedUnstructuredGridReader(
            registrationName=dataset_path.name,
            FileName=[absolute_path],
        )
    elif extension == ".vtu":
        reader = XMLUnstructuredGridReader(
            registrationName=dataset_path.name,
            FileName=[absolute_path],
        )
    else:
        raise ValueError(f"Unsupported input extension: {dataset_path.suffix}")

    UpdatePipeline(proxy=reader)
    return reader


def get_point_array_information(source: Any, field_name: str) -> Optional[Any]:
    """Return point-data array metadata, or None if the array is missing."""
    point_info = source.GetDataInformation().GetPointDataInformation()
    return point_info.GetArrayInformation(field_name)


def list_point_arrays(source: Any) -> list[str]:
    """Return available point-data array names."""
    point_info = source.GetDataInformation().GetPointDataInformation()
    names = []
    for index in range(point_info.GetNumberOfArrays()):
        array_info = point_info.GetArrayInformation(index)
        if array_info is not None:
            names.append(array_info.GetName())
    return names


def require_point_array(source: Any, field_name: str) -> None:
    """Raise a clear error if a point-data array is unavailable."""
    if get_point_array_information(source, field_name) is not None:
        return
    available = list_point_arrays(source)
    available_text = ", ".join(available) if available else "(none)"
    raise RuntimeError(
        f'Point-data field "{field_name}" was not found. '
        f"Available point-data arrays: {available_text}"
    )


def get_point_array_range(source: Any, field_name: str) -> tuple[float, float]:
    """Return scalar range, or vector-magnitude range for vector arrays."""
    array_info = get_point_array_information(source, field_name)
    if array_info is None:
        require_point_array(source, field_name)
    assert array_info is not None

    if array_info.GetNumberOfComponents() == 1:
        value_range = array_info.GetComponentRange(0)
    else:
        value_range = array_info.GetComponentRange(-1)
    return float(value_range[0]), float(value_range[1])


def create_qcrit_contour(
    source: Any,
    qcrit_field: str,
    qcrit_values: Sequence[float],
) -> Any:
    """Create q-criterion isosurfaces from point data."""
    require_point_array(source, qcrit_field)
    contour = Contour(registrationName="qcrit_isosurfaces", Input=source)
    contour.ContourBy = ["POINTS", qcrit_field]
    contour.Isosurfaces = [float(value) for value in qcrit_values]
    UpdatePipeline(proxy=contour)
    return contour


def create_airfoil_surface(
    source: Any,
    bounds: Sequence[float],
    normal_z_max: float,
) -> Any:
    """Extract a clipped near-airfoil surface without the far-field boundary."""
    surface = ExtractSurface(registrationName="dataset_surface", Input=source)
    UpdatePipeline(proxy=surface)

    normals = GenerateSurfaceNormals(
        registrationName="dataset_surface_normals",
        Input=surface,
    )
    UpdatePipeline(proxy=normals)

    abs_nz = Calculator(registrationName="surface_abs_nz", Input=normals)
    abs_nz.ResultArrayName = "abs_nz"
    abs_nz.Function = "abs(Normals_Z)"
    UpdatePipeline(proxy=abs_nz)

    non_spanwise = Threshold(
        registrationName="non_spanwise_surface",
        Input=abs_nz,
    )
    non_spanwise.Scalars = ["POINTS", "abs_nz"]
    non_spanwise.ThresholdMethod = "Between"
    non_spanwise.LowerThreshold = 0.0
    non_spanwise.UpperThreshold = float(normal_z_max)
    UpdatePipeline(proxy=non_spanwise)

    xmin, xmax, ymin, ymax, zmin, zmax = [float(value) for value in bounds]
    clipped = Clip(registrationName="airfoil_surface", Input=non_spanwise)
    clipped.ClipType = "Box"
    clipped.ClipType.Position = [xmin, ymin, zmin]
    clipped.ClipType.Length = [xmax - xmin, ymax - ymin, zmax - zmin]
    clipped.Invert = 1
    clipped.Crinkleclip = 1
    UpdatePipeline(proxy=clipped)
    return clipped


def configure_view(
    view: Any,
    background_color: Sequence[float],
    camera_position: Sequence[float],
    camera_view_up: Sequence[float],
) -> None:
    """Configure the render view and camera."""
    color = [float(value) for value in background_color]
    if "UseColorPaletteForBackground" in view.ListProperties():
        view.UseColorPaletteForBackground = 0
    if "BackgroundColorMode" in view.ListProperties():
        view.BackgroundColorMode = "Single Color"
    view.Background = color
    view.Background2 = color
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = [0.0, 0.0, 0.0]
    view.CameraPosition = [float(value) for value in camera_position]
    view.CameraViewUp = [float(value) for value in camera_view_up]


def configure_coloring(
    source: Any,
    displays: Sequence[Any],
    view: Any,
    velocity_field: str,
    scalar_range: Optional[Sequence[float]],
    show_colorbar: bool,
) -> tuple[float, float]:
    """Color rendered objects by velocity."""
    require_point_array(source, velocity_field)
    if scalar_range is None:
        color_minimum, color_maximum = get_point_array_range(
            source,
            velocity_field,
        )
    else:
        color_minimum = float(scalar_range[0])
        color_maximum = float(scalar_range[1])

    lookup_table = GetColorTransferFunction(velocity_field)
    opacity_function = GetOpacityTransferFunction(velocity_field)
    lookup_table.RescaleTransferFunction(color_minimum, color_maximum)
    opacity_function.RescaleTransferFunction(color_minimum, color_maximum)

    for display in displays:
        ColorBy(display, ("POINTS", velocity_field))
        display.LookupTable = lookup_table
        display.SetScalarBarVisibility(view, show_colorbar)

    if show_colorbar:
        scalar_bar = GetScalarBar(lookup_table, view)
        scalar_bar.Title = velocity_field
        scalar_bar.ComponentTitle = ""
        scalar_bar.TitleColor = [0.0, 0.0, 0.0]
        scalar_bar.LabelColor = [0.0, 0.0, 0.0]

    return color_minimum, color_maximum


def zoom_camera(view: Any, zoom_factor: float) -> None:
    """Zoom a render view after camera reset."""
    if zoom_factor == 1.0:
        return
    if view.CameraParallelProjection:
        view.CameraParallelScale = view.CameraParallelScale/zoom_factor
    else:
        view.CameraViewAngle = view.CameraViewAngle/zoom_factor


def _normalize(vector: Sequence[float]) -> list[float]:
    """Return a normalized 3-vector."""
    norm = math.sqrt(sum(float(value)*float(value) for value in vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return [float(value)/norm for value in vector]


def _cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Return the cross product of two 3-vectors."""
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]


def pan_camera(view: Any, pan: Sequence[float]) -> None:
    """Pan the camera in screen-right/screen-up coordinates."""
    pan_right = float(pan[0])
    pan_up = float(pan[1])
    if pan_right == 0.0 and pan_up == 0.0:
        return

    position = [float(value) for value in view.CameraPosition]
    focal = [float(value) for value in view.CameraFocalPoint]
    view_up = _normalize(view.CameraViewUp)
    direction = _normalize([focal[i] - position[i] for i in range(3)])
    screen_right = _normalize(_cross(direction, view_up))
    screen_up = _normalize(_cross(screen_right, direction))
    scale = float(view.CameraParallelScale)

    shift = [
        scale*(pan_right*screen_right[i] + pan_up*screen_up[i])
        for i in range(3)
    ]
    view.CameraPosition = [position[i] + shift[i] for i in range(3)]
    view.CameraFocalPoint = [focal[i] + shift[i] for i in range(3)]


def set_camera_focal_point(view: Any, focal_point: Sequence[float]) -> None:
    """Translate the camera so it looks at a requested focal point."""
    old_focal = [float(value) for value in view.CameraFocalPoint]
    new_focal = [float(value) for value in focal_point]
    delta = [new_focal[i] - old_focal[i] for i in range(3)]
    position = [float(value) for value in view.CameraPosition]
    view.CameraFocalPoint = new_focal
    view.CameraPosition = [position[i] + delta[i] for i in range(3)]


def save_image(
    output_path: Path,
    view: Any,
    resolution: Sequence[int],
    transparent_background: bool,
    tight_crop: bool,
    crop_padding: int,
    right_crop_fraction: float,
    background_color: Sequence[float],
) -> None:
    """Save the render view to disk."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    SaveScreenshot(
        str(output_path),
        view,
        ImageResolution=[int(resolution[0]), int(resolution[1])],
        TransparentBackground=1 if transparent_background else 0,
    )
    if not output_path.is_file():
        raise RuntimeError(
            f"ParaView did not create the requested image: {output_path}"
        )
    if tight_crop:
        crop_image(
            output_path,
            background_color,
            crop_padding,
            right_crop_fraction,
        )


def crop_image(
    output_path: Path,
    background_color: Sequence[float],
    padding: int,
    right_crop_fraction: float,
) -> None:
    """Crop whitespace around non-background pixels."""
    try:
        from PIL import Image, ImageChops  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "--tight-crop requires Pillow in the ParaView Python environment."
        ) from exc

    image = Image.open(output_path).convert("RGB")
    bg = tuple(int(round(255.0*float(value))) for value in background_color)
    background = Image.new("RGB", image.size, bg)
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()
    if bbox is None:
        return

    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(image.size[0], bbox[2] + padding)
    bottom = min(image.size[1], bbox[3] + padding)
    cropped = image.crop((left, top, right, bottom))
    if right_crop_fraction < 1.0:
        cropped_width = max(1, int(round(cropped.size[0]*right_crop_fraction)))
        cropped = cropped.crop((0, 0, cropped_width, cropped.size[1]))
    cropped.save(output_path)


def main() -> int:
    """Run the q-criterion rendering workflow."""
    args = parse_arguments()
    validate_arguments(args)

    ResetSession()
    reader = load_dataset(args.input_file)
    contour = create_qcrit_contour(
        source=reader,
        qcrit_field=args.qcrit_field,
        qcrit_values=args.qcrit_values,
    )

    view = CreateView("RenderView")
    configure_view(
        view=view,
        background_color=args.background_color,
        camera_position=args.camera_position,
        camera_view_up=args.camera_view_up,
    )

    displays = []
    if args.show_geometry:
        geometry_display = Show(reader, view)
        geometry_display.Representation = "Surface"
        displays.append(geometry_display)

    if args.airfoil_surface:
        airfoil = create_airfoil_surface(
            reader,
            args.airfoil_bounds,
            args.airfoil_normal_z_max,
        )
        airfoil_display = Show(airfoil, view)
        airfoil_display.Representation = "Surface"
        displays.append(airfoil_display)

    contour_display = Show(contour, view)
    contour_display.Representation = "Surface"
    displays.append(contour_display)

    color_minimum, color_maximum = configure_coloring(
        source=reader,
        displays=displays,
        view=view,
        velocity_field=args.velocity_field,
        scalar_range=args.scalar_range,
        show_colorbar=args.show_colorbar,
    )

    Render(view)
    view.ResetCamera()
    Render(view)
    set_camera_focal_point(view, args.camera_focal_point)
    zoom_camera(view, args.camera_zoom)
    pan_camera(view, args.camera_pan)
    Render(view)

    save_image(
        output_path=args.output,
        view=view,
        resolution=args.resolution,
        transparent_background=args.transparent_background,
        tight_crop=args.tight_crop,
        crop_padding=args.crop_padding,
        right_crop_fraction=args.right_crop_fraction,
        background_color=args.background_color,
    )

    print(f"Rendered q-criterion field: {args.qcrit_field}")
    print(f"Colored by velocity field: {args.velocity_field}")
    print(f"Velocity color range: [{color_minimum}, {color_maximum}]")
    print(f"Saved image to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # pylint: disable=broad-except
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
