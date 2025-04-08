import re
import xml.etree.ElementTree as xml
from threading import Lock

import logging

import numpy as np
from fastapi import HTTPException
import cooler
import h5py
import pandas as pd

from wsi_service.models.v3.slide import SlideLevel, SlideExtent, SlideInfo, SlidePixelSizeNm
from wsi_service.slide import Slide as BaseSlide
from wsi_service.utils.slide_utils import get_rgb_channel_list

logger = logging.getLogger("uvicorn")
TILE_SIZE = 256

class Slide(BaseSlide):
    async def open(self, filepath):
        self.filepath = filepath
        if not (cooler.fileops.is_cooler(filepath) or cooler.fileops.is_multires_file(filepath)):
            raise HTTPException(status_code=404, detail=f"Failed to open cool file.")
        self._get_info_mcool(filepath)

    async def close(self):
        return None

    async def get_info(self):
        return self.slide_info

    async def get_region(self, level, start_x, start_y, size_x, size_y, padding_color=None, z=0):
        # Flip level since the format assumes lowest level is lowest detail while highest level is highest detail
        level = self.max_zoom - level

        c = cooler.Cooler(self.filepath + "::" + str(level))
        # Resolution defines how many bases are per bin
        resolution = self.min_bin_size * (2 ** (self.max_zoom - level))

        tile = self._make_tile(c, resolution, start_x, start_y, size_x, size_y, "default")
        tile = np.array([self._assign_color(x) for x in tile])
        tile = tile.reshape(256, 256, 3).transpose(2, 0, 1)
        return tile

    async def get_thumbnail(self, max_x, max_y):
        return None

    async def get_tile(self, level, tile_x, tile_y, padding_color=None, z=0):
        if level > self.max_zoom:
            raise ValueError("Tile level too high")
        # Flip level since the format assumes lowest level is lowest detail while highest level is highest detail
        level = self.max_zoom - level

        c = cooler.Cooler(self.filepath + "::" + str(level))
        start_x = tile_x * self.slide_info.tile_extent.x
        start_y = tile_y * self.slide_info.tile_extent.y
        # Resolution defines how many bases are per bin
        resolution = self.min_bin_size * (2 ** (self.max_zoom - level))

        tile = self._make_tile(c, resolution, start_x, start_y, self.slide_info.tile_extent.x, self.slide_info.tile_extent.y, "default")
        tile = np.array([self._assign_color(x) for x in tile])

        tile = tile.reshape(256, 256, 3).transpose(2, 0, 1)
        return tile

    # private
    def _assign_color(self, val):
        if np.isnan(val):
            return np.array([200, 200, 200], dtype=np.uint8)
        mval = 0.1 if val > 0.1 else val
        if mval > 0.01:
            c_from = [0, 0, 0]
            c_to = [169, 3, 22]
            t = (mval - 0.01) / (0.1 - 0.01)
        elif mval > 0.001:
            c_from = [169, 3, 22]
            c_to = [245, 176, 54]
            t = (mval - 0.001) / (0.01 - 0.001)
        else:
            c_from = [245, 176, 54]
            c_to = [255, 255, 255]
            t = mval
        color = self._mix_colors(c_from, c_to, 1 - t)
        return color

    def _mix_colors(self, c1, c2, t):
        return np.array([
            c1[0] * (1 - t) + c2[0] * t,
            c1[1] * (1 - t) + c2[1] * t,
            c1[2] * (1 - t) + c2[2] * t,
        ], dtype=np.uint8)

    def _get_info_mcool(self, file_path):
        with h5py.File(file_path, "r") as f:
            self.max_zoom = int(f.attrs.get("max-zoom"))

            if self.max_zoom is None:
                raise ValueError("The `max_zoom` attribute is missing.")

            c = cooler.Cooler(f["0"])
            c_max = cooler.Cooler(f[str(self.max_zoom)])

            (chroms, chrom_sizes, chrom_cum_lengths) = self._get_chromosome_names_cumul_lengths(c)
            self.chromosomes = chroms
            self.chromosome_sizes = chrom_sizes
            self.chromosome_cumulative_lengths = chrom_cum_lengths
            self.genome_length = int(chrom_cum_lengths[-1])
            self.min_bin_size = int(f[str(self.max_zoom)].attrs["bin-size"])

            self.max_tile_bases_width = self.min_bin_size * TILE_SIZE * 2 ** self.max_zoom

            # the list of available data transforms
            self.transforms = {}

            for i in range(self.max_zoom):
                f_for_zoom = f[str(i)]["bins"]

                if "weight" in f_for_zoom:
                    self.transforms["weight"] = {"name": "ICE", "value": "weight"}
                if "KR" in f_for_zoom:
                    self.transforms["KR"] = {"name": "KR", "value": "KR"}
                if "VC" in f_for_zoom:
                    self.transforms["VC"] = {"name": "VC", "value": "VC"}
                if "VC_SQRT" in f_for_zoom:
                    self.transforms["VC_SQRT"] = {"name": "VC_SQRT", "value": "VC_SQRT"}

            self.slide_info = SlideInfo(
                id="",
                channels=get_rgb_channel_list(),
                channel_depth=8,
                extent=SlideExtent(x=c_max.info["nbins"], y=c_max.info["nbins"], z=1),
                num_levels=self.max_zoom + 1,
                pixel_size_nm=SlidePixelSizeNm(x=-1, y=-1),  # pixel size unknown
                tile_extent=SlideExtent(x=TILE_SIZE, y=TILE_SIZE, z=1),
                levels=self._get_mcool_levels(f),
            )

    def _get_mcool_levels(self, hdf_file):
        levels = []
        for i in range(self.max_zoom, -1, -1):
            c = cooler.Cooler(hdf_file[str(i)])
            levels.append(
                SlideLevel(
                    extent=SlideExtent(x=c.info["nbins"], y=c.info["nbins"], z=1),
                    downsample_factor=2,
                )
            )
        return levels

    def _get_chromosome_names_cumul_lengths(self, c):
        chrom_names = c.chromnames
        chrom_sizes = dict(c.chromsizes.astype(np.int64))
        chrom_cum_lengths = np.r_[0, np.cumsum(c.chromsizes.values)]
        return chrom_names, chrom_sizes, chrom_cum_lengths

    def _abs_coord_2_bin(self, c, abs_pos):
        try:
            chr_id = np.flatnonzero(self.chromosome_cumulative_lengths > abs_pos)[0] - 1
        except IndexError:
            return c.info["nbins"]

        chrom = self.chromosomes[chr_id]
        rel_pos = abs_pos - self.chromosome_cumulative_lengths[chr_id]
        return c.offset((chrom, rel_pos, self.chromosome_sizes[chrom]))


    def _get_data(
        self,
        c,
        resolution,
        start_pos_1,
        end_pos_1,
        start_pos_2,
        end_pos_2,
        transform="default",
    ):

        #i0 = self._abs_coord_2_bin(c, start_pos_1)
        i0 = start_pos_1 // resolution
        #i1 = self._abs_coord_2_bin(c, end_pos_1)
        i1 = end_pos_1 // resolution
        j0 = start_pos_2 // resolution
        j1 = end_pos_2 // resolution

        #j0 = self._abs_coord_2_bin(c, start_pos_2)
        #j1 = self._abs_coord_2_bin(c, end_pos_2)

        matrix = c.matrix(as_pixels=True, balance=False)

        if i0 >= matrix.shape[0] or j0 >= matrix.shape[1]:
            # query beyond the bounds of the matrix
            # return an empty matrix
            return (
                pd.DataFrame(columns=["genome_start1", "genome_start2", "balanced"]),
                (
                    pd.DataFrame({"genome_start": [], "genome_end": [], "weight": []}),
                    pd.DataFrame({"genome_start": [], "genome_end": [], "weight": []}),
                ),
            )

        # limit the range of the query to be within bounds
        i1 = min(i1, matrix.shape[0] - 1)
        j1 = min(j1, matrix.shape[1] - 1)
        pixels = matrix[i0: i1 + 1, j0: j1 + 1]

        # select bin columns to extract
        cols = ["chrom", "start", "end"]
        if (transform == "default" and "weight" in c.bins()) or transform == "weight":
            cols.append("weight")
        elif transform in ("KR", "VC", "VC_SQRT"):
            cols.append(transform)

        bins = c.bins(convert_enum=False)[cols]
        pixels = cooler.annotate(pixels, bins)

        pixels["genome_start1"] = self.chromosome_cumulative_lengths[pixels["chrom1"]] + pixels["start1"]
        pixels["genome_start2"] = self.chromosome_cumulative_lengths[pixels["chrom2"]] + pixels["start2"]

        bins1 = bins[i0 : i1 + 1]
        bins2 = bins[j0 : j1 + 1]

        bins1["genome_start"] = self.chromosome_cumulative_lengths[bins1["chrom"]] + bins1["start"]
        bins2["genome_start"] = self.chromosome_cumulative_lengths[bins2["chrom"]] + bins2["start"]

        bins1["genome_end"] = self.chromosome_cumulative_lengths[bins1["chrom"]] + bins1["end"]
        bins2["genome_end"] = self.chromosome_cumulative_lengths[bins2["chrom"]] + bins2["end"]

        # apply transform
        if (transform == "default" and "weight" in c.bins()) or transform == "weight":
            pixels["balanced"] = pixels["count"] * pixels["weight1"] * pixels["weight2"]

            return (pixels[["genome_start1", "genome_start2", "balanced"]], (bins1, bins2))
        elif transform in ("KR", "VC", "VC_SQRT"):
            pixels["balanced"] = (
                pixels["count"] / pixels[transform + "1"] / pixels[transform + "2"]
            )

            bins1["weight"] = bins1[transform]
            bins2["weight"] = bins2[transform]

            return (pixels[["genome_start1", "genome_start2", "balanced"]], (bins1, bins2))
        else:
            return (pixels[["genome_start1", "genome_start2", "count"]], (None, None))

    def _make_tile(
        self,
        c,
        resolution,
        start_x,
        start_y,
        size_x,
        size_y,
        transform_type="default"
    ):
        
        start1 = start_x * resolution
        end1 = (start_x + size_x) * resolution 
        start2 = start_y * resolution 
        end2 = (start_y + size_y) * resolution


        total_length = sum(self.chromosome_sizes.values())

        (data, (bins1, bins2)) = self._get_data(
            c,
            resolution,
            start1,
            end1,
            start2,
            end2,
            transform_type,
        )

        df = data[data["genome_start1"] >= start1]
        df = df[df["genome_start1"] < end1]

        df = df[df["genome_start2"] >= start2]
        df = df[df["genome_start2"] < end2]

        binsize = resolution

        j = ((df["genome_start1"].values - start1) // binsize).astype(int)
        i = ((df["genome_start2"].values - start2) // binsize).astype(int)

        if "balanced" in df:
            v = np.nan_to_num(df["balanced"].values)
        else:
            v = np.nan_to_num(df["count"].values)

        out = np.zeros((256, 256), dtype=np.float32)
        out[i, j] = v

        if bins1 is not None and bins2 is not None:
            sub_bins1 = bins1[bins1["genome_start"] >= start1]
            sub_bins2 = bins2[bins2["genome_start"] >= start2]

            sub_bins1 = sub_bins1[sub_bins1["genome_start"] < end1]
            sub_bins2 = sub_bins2[sub_bins2["genome_start"] < end2]

            # print("sub_bins1:", sub_bins1)

            nan_bins1 = sub_bins1[np.isnan(sub_bins1["weight"])]
            nan_bins2 = sub_bins2[np.isnan(sub_bins2["weight"])]

            bi = ((nan_bins1["genome_start"].values - start1) // binsize).astype(
                int
            )
            bj = ((nan_bins2["genome_start"].values - start2) // binsize).astype(
                int
            )

            bend1 = (
                (np.array(range(total_length, int(end1), int(resolution))) - start1)
                // binsize
            ).astype(int)
            bend2 = (
                (np.array(range(total_length, int(end2), int(resolution))) - start2)
                // binsize
            ).astype(int)

            bend1 = bend1[bend1 >= 0]
            bend2 = bend2[bend2 >= 0]

            out[:, bi] = np.nan
            out[bj, :] = np.nan

            out[:, bend1] = np.nan
            out[bend2, :] = np.nan

        # print('sum(isnan1)', isnan1-1)
        # print('out.ravel()', sum(np.isnan(out.ravel())), len(out.ravel()))
        result = out.ravel()
        return result

