#!/usr/bin/env pvpython
"""Render four VTU/PVTU datasets in a 2x2 ParaView layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from paraview.simple import (  # type: ignore
    ColorBy,
    Contour,
    CreateLayout,
    CreateView,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    GetScalarBar,
    Render,
    ResetSession,
    SaveScreenshot,
    Show,
    Text,
    UpdatePipeline,
    XMLPartitionedUnstructuredGridReader,
    XMLUnstructuredGridReader,
)


CASE_NAMES = ("case1", "case2", "case3", "case4")
SUPPORTED_REPRESENTATIONS = (
    "Surface",
    "Surface With Edges",
    "Wireframe",
    "Points",
    "Outline",
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
            "Find datasets under case1 through case4, render them in a "
            "2x2 ParaView layout, and save the complete layout."
        )
    )
    parser.add_argument(
        "base_directory",
        type=Path,
        help="Directory containing case1, case2, case3, and case4.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("four_cases.png"),
        help="Output image filename. Default: four_cases.png",
    )
    parser.add_argument(
        "--field",
        default=None,
        help="Point-data or cell-data field used for coloring.",
    )
    parser.add_argument(
        "--contour-field",
        default=None,
        help="Field used to extract isosurfaces before rendering.",
    )
    parser.add_argument(
        "--contour-values",
        nargs="+",
        type=float,
        help="Isosurface values used with --contour-field.",
    )
    parser.add_argument(
        "--association",
        choices=("POINTS", "CELLS"),
        default="POINTS",
        help="Field association. Default: POINTS",
    )
    parser.add_argument(
        "--range",
        dest="scalar_range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Common scalar range for all four panels.",
    )
    parser.add_argument(
        "--titles",
        nargs=4,
        metavar=("TITLE1", "TITLE2", "TITLE3", "TITLE4"),
        help="Titles for case1, case2, case3, and case4.",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=(3200, 2400),
        metavar=("WIDTH", "HEIGHT"),
        help="Output resolution. Default: 3200 2400",
    )
    parser.add_argument(
        "--representation",
        choices=SUPPORTED_REPRESENTATIONS,
        default="Surface",
        help="ParaView representation. Default: Surface",
    )
    parser.add_argument(
        "--view-direction",
        choices=("xy", "xz", "yz", "isometric"),
        default="isometric",
        help="Camera orientation. Default: isometric",
    )
    parser.add_argument(
        "--camera-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Override --view-direction with an explicit camera position.",
    )
    parser.add_argument(
        "--camera-view-up",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Camera up direction used with --camera-position.",
    )
    parser.add_argument(
        "--parallel-projection",
        action="store_true",
        help="Enable parallel camera projection.",
    )
    parser.add_argument(
        "--camera-zoom",
        type=float,
        default=1.0,
        help="Camera zoom factor applied after ResetCamera. Default: 1.0",
    )
    parser.add_argument(
        "--show-colorbars",
        action="store_true",
        help="Display a color bar in each panel.",
    )
    parser.add_argument(
        "--transparent-background",
        action="store_true",
        help="Save with a transparent background.",
    )
    parser.add_argument(
        "--tight-crop-panels",
        action="store_true",
        help="Crop each 2x2 panel around non-background pixels after saving.",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=20,
        help="Padding in pixels for --tight-crop-panels. Default: 20",
    )
    parser.add_argument(
        "--background-color",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("R", "G", "B"),
        help="Render-view background color. Default: 1 1 1",
    )
    parser.add_argument(
        "--separator-width",
        type=int,
        default=4,
        help="Width of panel separators in pixels. Default: 4",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate paths and numeric command-line options."""
    if not args.base_directory.is_dir():
        raise FileNotFoundError(
            f"Base directory does not exist: {args.base_directory}"
        )

    width, height = args.resolution
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")

    if args.separator_width < 0:
        raise ValueError("Separator width must be nonnegative.")

    if args.camera_zoom <= 0.0:
        raise ValueError("Camera zoom must be positive.")

    if args.crop_padding < 0:
        raise ValueError("Crop padding must be nonnegative.")

    if any(value < 0.0 or value > 1.0 for value in args.background_color):
        raise ValueError("Background color entries must be in [0, 1].")

    if args.contour_field is None and args.contour_values is not None:
        raise ValueError("--contour-values requires --contour-field.")

    if args.contour_field is not None:
        if args.contour_values is None or len(args.contour_values) == 0:
            raise ValueError("--contour-field requires --contour-values.")
        if args.association != "POINTS":
            raise ValueError("Contour extraction currently requires POINTS data.")

    if args.scalar_range is not None:
        scalar_minimum, scalar_maximum = args.scalar_range
        if scalar_maximum <= scalar_minimum:
            raise ValueError(
                "Scalar range maximum must be greater than its minimum."
            )

    output_extension = args.output.suffix.lower()
    if output_extension not in SUPPORTED_OUTPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_EXTENSIONS))
        raise ValueError(
            f"Unsupported output extension '{output_extension}'. "
            f"Supported extensions: {supported}"
        )


def find_dataset(case_directory: Path) -> Path:
    """
    Find exactly one PVTU or VTU dataset in a case directory.

    PVTU files have priority because VTU files may be partition files
    referenced by the PVTU metadata file.
    """
    if not case_directory.is_dir():
        raise FileNotFoundError(
            f"Required case directory does not exist: {case_directory}"
        )

    pvtu_files = sorted(case_directory.rglob("*.pvtu"))
    if len(pvtu_files) == 1:
        return pvtu_files[0]
    if len(pvtu_files) > 1:
        candidates = "\n  ".join(str(path) for path in pvtu_files)
        raise RuntimeError(
            f"Multiple PVTU files found in {case_directory}:\n"
            f"  {candidates}\n"
            "The dataset is ambiguous."
        )

    vtu_files = sorted(case_directory.rglob("*.vtu"))
    if len(vtu_files) == 1:
        return vtu_files[0]
    if len(vtu_files) > 1:
        candidates = "\n  ".join(str(path) for path in vtu_files)
        raise RuntimeError(
            f"Multiple VTU files found in {case_directory}:\n"
            f"  {candidates}\n"
            "Provide a PVTU metadata file or remove the ambiguity."
        )

    raise FileNotFoundError(
        f"No .pvtu or .vtu dataset found under {case_directory}"
    )


def find_all_datasets(base_directory: Path) -> list[Path]:
    """Find one visualization dataset for each required case."""
    return [find_dataset(base_directory / case_name) for case_name in CASE_NAMES]


def create_layout() -> tuple[Any, list[Any]]:
    """
    Create a 2x2 ParaView layout.

    The panel ordering is:
        case1 | case2
        ------+------
        case3 | case4
    """
    layout = CreateLayout(name="Four-case comparison")
    layout.SplitHorizontal(0, 0.5)
    layout.SplitVertical(1, 0.5)
    layout.SplitVertical(2, 0.5)

    views = [CreateView("RenderView") for _ in range(4)]
    layout.AssignView(3, views[0])
    layout.AssignView(5, views[1])
    layout.AssignView(4, views[2])
    layout.AssignView(6, views[3])
    return layout, views


def configure_view(
    view: Any,
    parallel_projection: bool,
    background_color: Sequence[float],
) -> None:
    """Apply common render-view properties."""
    color = [float(value) for value in background_color]
    if "UseColorPaletteForBackground" in view.ListProperties():
        view.UseColorPaletteForBackground = 0
    if "BackgroundColorMode" in view.ListProperties():
        view.BackgroundColorMode = "Single Color"
    view.Background = color
    view.Background2 = color
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1 if parallel_projection else 0


def load_dataset(dataset_path: Path, registration_name: str) -> Any:
    """Create the correct reader for a PVTU or VTU dataset."""
    absolute_path = str(dataset_path.resolve())
    extension = dataset_path.suffix.lower()

    if extension == ".pvtu":
        reader = XMLPartitionedUnstructuredGridReader(
            registrationName=registration_name,
            FileName=[absolute_path],
        )
    elif extension == ".vtu":
        reader = XMLUnstructuredGridReader(
            registrationName=registration_name,
            FileName=[absolute_path],
        )
    else:
        raise ValueError(f"Unsupported dataset extension: {dataset_path.suffix}")

    UpdatePipeline(proxy=reader)
    return reader


def create_contour_filter(
    source: Any,
    registration_name: str,
    association: str,
    field_name: str,
    contour_values: Sequence[float],
) -> Any:
    """Extract isosurfaces from a source dataset."""
    if get_array_information(source, association, field_name) is None:
        available = list_available_arrays(source, association)
        available_text = ", ".join(available) if available else "(none)"
        raise RuntimeError(
            f'Contour field "{field_name}" was not found in {association} '
            f"data. Available arrays: {available_text}"
        )

    contour = Contour(registrationName=registration_name, Input=source)
    contour.ContourBy = [association, field_name]
    contour.Isosurfaces = [float(value) for value in contour_values]
    UpdatePipeline(proxy=contour)
    return contour


def get_attribute_information(reader: Any, association: str) -> Any:
    """Return point-data or cell-data metadata."""
    data_information = reader.GetDataInformation()
    if association == "POINTS":
        return data_information.GetPointDataInformation()
    if association == "CELLS":
        return data_information.GetCellDataInformation()
    raise ValueError(f"Unsupported association: {association}")


def get_array_information(
    reader: Any,
    association: str,
    field_name: str,
) -> Optional[Any]:
    """Return metadata for a named array, or None if unavailable."""
    attribute_information = get_attribute_information(reader, association)
    return attribute_information.GetArrayInformation(field_name)


def list_available_arrays(reader: Any, association: str) -> list[str]:
    """Return all array names for the selected field association."""
    attribute_information = get_attribute_information(reader, association)
    names = []
    for index in range(attribute_information.GetNumberOfArrays()):
        array_information = attribute_information.GetArrayInformation(index)
        if array_information is not None:
            names.append(array_information.GetName())
    return names


def get_array_range(
    reader: Any,
    association: str,
    field_name: str,
) -> tuple[float, float]:
    """Return a scalar or vector-magnitude range for an array."""
    array_information = get_array_information(
        reader,
        association,
        field_name,
    )
    if array_information is None:
        available = list_available_arrays(reader, association)
        available_text = ", ".join(available) if available else "(none)"
        raise RuntimeError(
            f'Field "{field_name}" was not found in {association} data. '
            f"Available arrays: {available_text}"
        )

    if array_information.GetNumberOfComponents() == 1:
        value_range = array_information.GetComponentRange(0)
    else:
        value_range = array_information.GetComponentRange(-1)

    return float(value_range[0]), float(value_range[1])


def compute_global_range(
    readers: Sequence[Any],
    association: str,
    field_name: str,
) -> tuple[float, float]:
    """Compute one common field range across all datasets."""
    ranges = [
        get_array_range(reader, association, field_name)
        for reader in readers
    ]
    global_minimum = min(value_range[0] for value_range in ranges)
    global_maximum = max(value_range[1] for value_range in ranges)
    if global_maximum <= global_minimum:
        raise RuntimeError(
            f'Field "{field_name}" has a degenerate global range: '
            f"[{global_minimum}, {global_maximum}]"
        )
    return global_minimum, global_maximum


def configure_coloring(
    readers: Sequence[Any],
    displays: Sequence[Any],
    views: Sequence[Any],
    field_name: str,
    association: str,
    scalar_range: Optional[Sequence[float]],
    show_colorbars: bool,
) -> tuple[float, float]:
    """Apply one shared lookup table and scalar range to all panels."""
    for case_index, reader in enumerate(readers, start=1):
        if get_array_information(reader, association, field_name) is None:
            available = list_available_arrays(reader, association)
            available_text = ", ".join(available) if available else "(none)"
            raise RuntimeError(
                f'Field "{field_name}" is missing from case{case_index}. '
                f"Available {association.lower()} arrays: {available_text}"
            )

    if scalar_range is None:
        color_minimum, color_maximum = compute_global_range(
            readers,
            association,
            field_name,
        )
    else:
        color_minimum = float(scalar_range[0])
        color_maximum = float(scalar_range[1])

    lookup_table = GetColorTransferFunction(field_name)
    opacity_function = GetOpacityTransferFunction(field_name)
    lookup_table.RescaleTransferFunction(color_minimum, color_maximum)
    opacity_function.RescaleTransferFunction(color_minimum, color_maximum)

    for display, view in zip(displays, views):
        ColorBy(display, (association, field_name))
        display.LookupTable = lookup_table
        display.SetScalarBarVisibility(view, show_colorbars)
        if show_colorbars:
            scalar_bar = GetScalarBar(lookup_table, view)
            scalar_bar.Title = field_name
            scalar_bar.ComponentTitle = ""
            scalar_bar.TitleColor = [0.0, 0.0, 0.0]
            scalar_bar.LabelColor = [0.0, 0.0, 0.0]

    return color_minimum, color_maximum


def configure_solid_coloring(displays: Sequence[Any]) -> None:
    """Render all datasets with a neutral solid color."""
    for display in displays:
        display.ColorArrayName = [None, ""]
        display.DiffuseColor = [0.8, 0.8, 0.8]


def set_camera_direction(view: Any, direction: str) -> None:
    """Set a standard camera viewing direction before camera reset."""
    view.CameraFocalPoint = [0.0, 0.0, 0.0]
    if direction == "xy":
        view.CameraPosition = [0.0, 0.0, 1.0]
        view.CameraViewUp = [0.0, 1.0, 0.0]
    elif direction == "xz":
        view.CameraPosition = [0.0, -1.0, 0.0]
        view.CameraViewUp = [0.0, 0.0, 1.0]
    elif direction == "yz":
        view.CameraPosition = [1.0, 0.0, 0.0]
        view.CameraViewUp = [0.0, 0.0, 1.0]
    elif direction == "isometric":
        view.CameraPosition = [1.0, 1.0, 1.0]
        view.CameraViewUp = [0.0, 0.0, 1.0]
    else:
        raise ValueError(f"Unsupported view direction: {direction}")


def set_custom_camera(
    view: Any,
    camera_position: Optional[Sequence[float]],
    camera_view_up: Optional[Sequence[float]],
) -> bool:
    """Set an explicit camera orientation if requested."""
    if camera_position is None:
        return False
    view.CameraFocalPoint = [0.0, 0.0, 0.0]
    view.CameraPosition = [float(value) for value in camera_position]
    if camera_view_up is None:
        view.CameraViewUp = [0.0, 0.0, 1.0]
    else:
        view.CameraViewUp = [float(value) for value in camera_view_up]
    return True


def copy_camera(source_view: Any, target_view: Any) -> None:
    """Copy the final camera configuration between render views."""
    target_view.CameraPosition = list(source_view.CameraPosition)
    target_view.CameraFocalPoint = list(source_view.CameraFocalPoint)
    target_view.CameraViewUp = list(source_view.CameraViewUp)
    target_view.CameraViewAngle = source_view.CameraViewAngle
    target_view.CameraParallelProjection = source_view.CameraParallelProjection
    target_view.CameraParallelScale = source_view.CameraParallelScale


def zoom_camera(view: Any, zoom_factor: float) -> None:
    """Zoom a render view after camera reset."""
    if zoom_factor == 1.0:
        return
    if view.CameraParallelProjection:
        view.CameraParallelScale = view.CameraParallelScale/zoom_factor
    else:
        view.CameraViewAngle = view.CameraViewAngle/zoom_factor


def add_title(view: Any, title: str) -> tuple[Any, Any]:
    """Add a title at the upper center of a render view."""
    text_source = Text()
    text_source.Text = title
    text_display = Show(text_source, view)
    text_display.WindowLocation = "Upper Center"
    text_display.FontSize = 24
    text_display.Bold = 1
    text_display.Color = [0.0, 0.0, 0.0]
    return text_source, text_display


def save_layout(
    layout: Any,
    output_path: Path,
    resolution: Sequence[int],
    separator_width: int,
    separator_color: Sequence[float],
    transparent_background: bool,
    tight_crop_panels: bool,
    crop_padding: int,
    background_color: Sequence[float],
) -> None:
    """Save the complete 2x2 layout to one image."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    SaveScreenshot(
        str(output_path),
        layout,
        ImageResolution=[int(resolution[0]), int(resolution[1])],
        SeparatorWidth=int(separator_width),
        SeparatorColor=[float(value) for value in separator_color],
        TransparentBackground=1 if transparent_background else 0,
    )
    if not output_path.is_file():
        raise RuntimeError(
            f"ParaView did not create the requested image: {output_path}"
        )
    if tight_crop_panels:
        crop_panels(output_path, background_color, crop_padding)


def crop_panels(
    output_path: Path,
    background_color: Sequence[float],
    padding: int,
) -> None:
    """Crop each quadrant around non-background pixels and reassemble."""
    try:
        from PIL import Image, ImageChops  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "--tight-crop-panels requires Pillow in the ParaView Python "
            "environment."
        ) from exc

    image = Image.open(output_path).convert("RGB")
    width, height = image.size
    if width % 2 != 0 or height % 2 != 0:
        raise RuntimeError("Panel cropping requires even image dimensions.")

    bg = tuple(int(round(255.0*float(value))) for value in background_color)
    half_width = width//2
    half_height = height//2
    panel_boxes = [
        (0, 0, half_width, half_height),
        (half_width, 0, width, half_height),
        (0, half_height, half_width, height),
        (half_width, half_height, width, height),
    ]

    cropped_panels = []
    for box in panel_boxes:
        panel = image.crop(box)
        background = Image.new("RGB", panel.size, bg)
        diff = ImageChops.difference(panel, background)
        bbox = diff.getbbox()
        if bbox is None:
            cropped_panels.append(panel)
            continue
        left = max(0, bbox[0] - padding)
        top = max(0, bbox[1] - padding)
        right = min(panel.size[0], bbox[2] + padding)
        bottom = min(panel.size[1], bbox[3] + padding)
        cropped_panels.append(panel.crop((left, top, right, bottom)))

    cell_width = max(panel.size[0] for panel in cropped_panels)
    cell_height = max(panel.size[1] for panel in cropped_panels)
    output = Image.new("RGB", (2*cell_width, 2*cell_height), bg)
    offsets = [
        (cell_width - cropped_panels[0].size[0],
         cell_height - cropped_panels[0].size[1]),
        (cell_width, cell_height - cropped_panels[1].size[1]),
        (cell_width - cropped_panels[2].size[0], cell_height),
        (cell_width, cell_height),
    ]
    for panel, offset in zip(cropped_panels, offsets):
        xoff = offset[0]
        yoff = offset[1]
        output.paste(panel, (xoff, yoff))

    output.save(output_path)


def main() -> int:
    """Run the complete four-panel rendering workflow."""
    args = parse_arguments()
    validate_arguments(args)
    dataset_paths = find_all_datasets(args.base_directory)

    print("Located datasets:")
    for case_name, dataset_path in zip(CASE_NAMES, dataset_paths):
        print(f"  {case_name}: {dataset_path}")

    ResetSession()
    layout, views = create_layout()
    for view in views:
        configure_view(view, args.parallel_projection, args.background_color)

    readers = []
    render_sources = []
    displays = []
    title_objects = []
    for case_index, (dataset_path, view) in enumerate(
        zip(dataset_paths, views),
        start=1,
    ):
        reader = load_dataset(
            dataset_path,
            registration_name=f"case{case_index}",
        )
        if args.contour_field is None:
            render_source = reader
        else:
            render_source = create_contour_filter(
                source=reader,
                registration_name=f"case{case_index}_contour",
                association=args.association,
                field_name=args.contour_field,
                contour_values=args.contour_values,
            )
        display = Show(render_source, view)
        display.Representation = args.representation
        readers.append(reader)
        render_sources.append(render_source)
        displays.append(display)

    if args.field is not None:
        color_minimum, color_maximum = configure_coloring(
            readers=render_sources,
            displays=displays,
            views=views,
            field_name=args.field,
            association=args.association,
            scalar_range=args.scalar_range,
            show_colorbars=args.show_colorbars,
        )
        print(
            f'Using common range for "{args.field}": '
            f"[{color_minimum}, {color_maximum}]"
        )
    else:
        configure_solid_coloring(displays)

    if args.titles is not None:
        for view, title in zip(views, args.titles):
            title_objects.append(add_title(view, title))

    first_view = views[0]
    if not set_custom_camera(
        first_view,
        args.camera_position,
        args.camera_view_up,
    ):
        set_camera_direction(first_view, args.view_direction)
    Render(first_view)
    first_view.ResetCamera()
    Render(first_view)
    zoom_camera(first_view, args.camera_zoom)
    Render(first_view)
    for view in views[1:]:
        copy_camera(first_view, view)
        Render(view)

    for view in views:
        Render(view)

    save_layout(
        layout=layout,
        output_path=args.output,
        resolution=args.resolution,
        separator_width=args.separator_width,
        separator_color=args.background_color,
        transparent_background=args.transparent_background,
        tight_crop_panels=args.tight_crop_panels,
        crop_padding=args.crop_padding,
        background_color=args.background_color,
    )
    print(f"Saved four-panel image to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # pylint: disable=broad-except
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
