import os
import tempfile

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import svgwrite
from contourpy import contour_generator
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from shapely import affinity, geometry
from shapely.validation import make_valid


def plot_level_polygon(ax, polygon, color="red"):
    for p in polygon:
        x, y = p.exterior.xy
        coords = np.column_stack((x, y))
        patch = mpatches.Polygon(
            coords, closed=True, facecolor=color, edgecolor="black", alpha=0.3
        )
        ax.add_patch(patch)


def plot_preview(ax, obj):
    im = ax.imshow(obj.elevation, cmap="terrain")
    mpl_contours = ax.contour(obj.elevation, levels=obj.layers, colors="red")
    return im, mpl_contours


def extract_all_polygons(geom):
    """Recursively yield all Polygon objects from a geometry, including MultiPolygons and GeometryCollections."""
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        for p in geom.geoms:
            yield from extract_all_polygons(p)
    elif geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            yield from extract_all_polygons(g)
    # Ignore other geometry types (Point, LineString, etc.)


def downsample_and_save(src_path, scale_factor=2):
    with rasterio.open(src_path) as src:
        # Compute target shape
        new_height = src.height // scale_factor
        new_width = src.width // scale_factor
        # Create new affine transform (increase pixel size)
        new_transform = src.transform * src.transform.scale(scale_factor, scale_factor)
        # Prepare downsampled array
        arr = np.empty((src.count, new_height, new_width), dtype=src.dtypes[0])
        for i in range(src.count):
            reproject(
                src.read(i + 1),
                arr[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=src.crs,
                resampling=Resampling.average,
            )
        # Write to temp file
        tmp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        with rasterio.open(
            tmp_file.name,
            "w",
            driver="GTiff",
            height=new_height,
            width=new_width,
            count=src.count,
            dtype=arr.dtype,
            crs=src.crs,
            transform=new_transform,
            nodata=src.nodata,
        ) as dst:
            dst.write(arr)
        return tmp_file.name


class DEMToCardboardConverter:

    def __init__(self, fname, scale, thickness):
        self.scale = scale
        self.thickness = thickness

        self._load_dem(fname)

    def _load_dem(self, fname):
        with rasterio.open(fname) as src:
            self.elevation = src.read(1)
            self.pixel_size = src.transform[0]
            nodata = src.nodata

        if nodata is not None:
            self.elevation = np.where(self.elevation == nodata, np.nan, self.elevation)

    def calculate_layers(self):
        """Calculate elevation values for each layer."""
        elevation_interval = (self.thickness * self.scale) / 1000.0  # Convert to meters

        min_elev = np.nanmin(self.elevation)
        max_elev = np.nanmax(self.elevation)

        num_layers = int(np.ceil((max_elev - min_elev) / elevation_interval))
        self.layers = [min_elev + i * elevation_interval for i in range(num_layers + 1)]

        print(f"Elevation: {min_elev:.1f}m to {max_elev:.1f}m")
        print(f"Interval: {elevation_interval:.2f}m per layer")
        print(f"Layers: {num_layers}")

    def _extract_contour_layer(self, cont_gen, height):
        MAX_HEIGHT = 1_000_000
        contour = cont_gen.filled(height, MAX_HEIGHT)
        polygons = []
        for c in contour[0]:
            p = geometry.Polygon(c)
            if not p.is_valid:
                p = make_valid(p)
            polygons.append(p)
        return polygons

    def extract_contours(self):
        cont_gen = contour_generator(z=self.elevation)
        self.polygons = []
        for layer in self.layers:
            geoms = self._extract_contour_layer(cont_gen, layer)
            if len(geoms) > 0:
                geoms = [poly for geom in geoms for poly in extract_all_polygons(geom)]
            self.polygons.append(geoms)

    @staticmethod
    def scale_geometry(geom, pixel_size, scale):
        """Scale geometry from pixel coordinates to millimeters."""
        scale_factor = (pixel_size / scale) * 1000  # to mm
        return affinity.scale(
            geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)
        )

    def save_layer_as_svg(self, fname_base, layer_num, plot_inner=False):
        filename = f"{fname_base}_{layer_num:02d}.svg"

        this_polygon = self.polygons[layer_num]
        if len(this_polygon) == 0:
            return

        if layer_num < len(self.layers) - 1:
            next_polygon = self.polygons[layer_num + 1]
        else:
            next_polygon = None

        minx = min(
            DEMToCardboardConverter.scale_geometry(
                p, self.pixel_size, self.scale
            ).bounds[0]
            for p in this_polygon
        )
        miny = min(
            DEMToCardboardConverter.scale_geometry(
                p, self.pixel_size, self.scale
            ).bounds[1]
            for p in this_polygon
        )
        maxx = max(
            DEMToCardboardConverter.scale_geometry(
                p, self.pixel_size, self.scale
            ).bounds[2]
            for p in this_polygon
        )
        maxy = max(
            DEMToCardboardConverter.scale_geometry(
                p, self.pixel_size, self.scale
            ).bounds[3]
            for p in this_polygon
        )
        width = maxx - minx
        height = maxy - miny
        margin = 10

        dwg = svgwrite.Drawing(
            filename,
            size=(f"{width+2*margin}mm", f"{height+2*margin}mm"),
            viewBox=f"{minx-margin} {miny-margin} {width+2*margin} {height+2*margin}",
        )

        # Add label
        dwg.add(
            dwg.text(
                f"Layer {layer_num} - {self.layers[layer_num]:.1f}m",
                insert=(minx, miny - 5),
                font_size="6px",
            )
        )

        def add_polygon_to_svg(coords, fill_color, stroke_width, stroke_color):
            points = [(x, y) for x, y in list(coords)]

            dwg.add(
                dwg.polygon(
                    points=points,
                    fill=fill_color,
                    stroke=stroke_color,
                    stroke_width=stroke_width,
                )
            )

        # Add this layer:
        for p in this_polygon:
            if p.is_empty:
                continue
            scaled_p = DEMToCardboardConverter.scale_geometry(
                p, self.pixel_size, self.scale
            )
            # add outer contours:
            add_polygon_to_svg(
                scaled_p.exterior.coords,
                fill_color="grey",
                stroke_color="black",
                stroke_width="0.5",
            )

            if plot_inner:
                pass

        # add next layer:
        if next_polygon:
            for p in next_polygon:
                if p.is_empty:
                    continue
                scaled_p = DEMToCardboardConverter.scale_geometry(
                    p, self.pixel_size, self.scale
                )
                # add outer contours:
                add_polygon_to_svg(
                    scaled_p.exterior.coords,
                    fill_color="none",
                    stroke_color="red",
                    stroke_width="0.5",
                )

        # Save
        dwg.save()
        print(f"Layer {layer_num}: Saved to {filename} ({width:.1f} x {height:.1f} mm)")


class MultiDEMToCardboardConverter(DEMToCardboardConverter):
    def __init__(self, fnames, scale, thickness, downsampling):
        self.scale = scale
        self.thickness = thickness
        self.downsampling = downsampling

        self._load_multi(fnames)
        print(self.elevation.shape)

    def _load_multi(self, fnames):
        temp_files = []
        for f in fnames:
            tmp_path = downsample_and_save(f, self.downsampling)
            temp_files.append(tmp_path)

        srcs = [rasterio.open(f) for f in temp_files]
        self.elevation, out_transform = merge(srcs)
        self.elevation = self.elevation.squeeze()
        self.pixel_size = out_transform[0]
        print("pixel_size", self.pixel_size)


if __name__ == "__main__":

    ### SETTINGS ###
    SOURCE_FILE_PATH = "oberland/data"
    SVG_FILE_PATH = "oberland/svg"

    SCALE = 50_000  # scale 1:X
    CARDBOARD_THICKNESS = 2  # mm
    DOWNSAMPLING_FACTOR = 10

    print(f"model scale: 1:{SCALE}")
    print(f"cardboard thickness: {CARDBOARD_THICKNESS} mm")
    print(f"downsampling factor: {DOWNSAMPLING_FACTOR}")

    input_files = [
        os.path.join(SOURCE_FILE_PATH, f) for f in os.listdir(SOURCE_FILE_PATH)
    ]

    converter = MultiDEMToCardboardConverter(
        input_files, SCALE, CARDBOARD_THICKNESS, DOWNSAMPLING_FACTOR
    )
    print(f"processed {len(input_files)} tif-files")
    converter.calculate_layers()
    fig, ax = plt.subplots()
    im, _ = plot_preview(ax, converter)
    fig.colorbar(im)
    fig.savefig(os.path.join(SVG_FILE_PATH, "overview.svg"))
    converter.extract_contours()
    for i, p in enumerate(converter.polygons):
        # plot_level_polygon(ax, p, color="blue")
        converter.save_layer_as_svg(os.path.join(SVG_FILE_PATH, "layer"), i)
