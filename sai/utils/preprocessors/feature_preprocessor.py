# Copyright 2025 Xin Huang
#
# GNU General Public License v3.0
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, please see
#
#    https://www.gnu.org/licenses/gpl-3.0.en.html


import numpy as np
from typing import Any
from sai.stats.features import calc_u, calc_q, calc_fd, calc_df
from sai.utils.preprocessors import DataPreprocessor
import re

class FeaturePreprocessor(DataPreprocessor):
    """
    A preprocessor subclass for generating feature vectors from genomic data.

    This class extends DataPreprocessor to include additional functionality for creating
    feature vectors based on genomic variants, reference and target individual genotypes,
    and window-based genomic statistics.
    """

    def __init__(
        self,
        w: float,
        y: list[float],
        output_file: str,
        stat_type: str,
        anc_allele_available: bool = False,
    ):
        """
        Construct the FeaturePreprocessor.

        See full docstring in source for parameter details.
        """
        # ---------- basic attributes ----------
        self.w = w
        self.y = y
        self.output_file = output_file
        self.anc_allele_available = anc_allele_available
        self.stat_type = stat_type
        self.stat_prefix = stat_type[0]        # first letter

        # ---------- validate stat_type ----------
        if not (re.fullmatch(r"[UQ]\d{2}", stat_type) or stat_type in {"fd", "df"}):
            raise ValueError(
                f"Invalid stat_type: {stat_type}. "
                "Allowed: 'UXX', 'QXX', 'fd', or 'df'."
            )

        # ---------- threshold only for U / Q ----------
        if self.stat_prefix in {"U", "Q"}:
            # 'U50' -> 0.50 ; 'Q95' -> 0.95
            self.threshold = int(stat_type[1:]) / 100
        else:                                   # fd or df
            self.threshold = None
    def run(
        self,
        chr_name: str,
        ref_pop: str,
        tgt_pop: str,
        src_pop_list: list[str],
        start: int,
        end: int,
        pos: np.ndarray,
        ref_gts: np.ndarray,
        tgt_gts: np.ndarray,
        src_gts_list: list[np.ndarray],
        ploidy: int,
    ) -> list[dict[str, Any]]:
        """
        Generates feature vectors for a specified genomic window.

        Parameters
        ----------
        chr_name : str
            Chromosome name.
        ref_pop : str
            Reference population name.
        tgt_pop : str
            Target population name.
        src_pop_list : list[str]
            List of source population names.
        start : int
            Start position of the genomic window.
        end : int
            End position of the genomic window.
        pos : np.ndarray
            A 1D numpy array where each element represents the genomic position.
        ref_gts : np.ndarray
            Genotype data for the reference population.
        tgt_gts : np.ndarray
            Genotype data for the target population.
        src_gts_list : list[np.ndarray]
            List of genotype arrays for each source population.
        ploidy: int
            Ploidy of the genome.

        Returns
        -------
        list[dict[str, Any]]
            A list containing a dictionary of calculated feature vectors for the genomic window.
        """
        items = {
            "chr_name": chr_name,
            "start": start,
            "end": end,
            "ref_pop": ref_pop,
            "tgt_pop": tgt_pop,
            "src_pop_list": src_pop_list,
            "nsnps": len(pos),
        }

        if (
            (ref_gts is None)
            or (tgt_gts is None)
            or (src_gts_list is None)
            or (ploidy is None)
        ):
            items["statistic"] = np.nan
            items["candidates"] = np.array([])
        elif self.stat_type == "fd":
            # f_d hesapla
            items["statistic"] = calc_fd(
                ref_gts=ref_gts,
                tgt_gts=tgt_gts,
                src_gts=src_gts_list[0],
                ploidy=ploidy,
            )
            items["candidates"] = np.array([])
        elif self.stat_type == "df":
            # d_f hesapla
            items["statistic"] = calc_df(
                ref_gts=ref_gts,
                tgt_gts=tgt_gts,
                src_gts=src_gts_list[0],
                ploidy=ploidy,
            )
            items["candidates"] = np.array([])
        elif self.stat_prefix == "U":
            items["statistic"], items["candidates"] = calc_u(
                ref_gts=ref_gts,
                tgt_gts=tgt_gts,
                src_gts_list=src_gts_list,
                pos=pos,
                w=self.w,
                x=self.threshold,
                y_list=self.y,
                ploidy=ploidy,
                anc_allele_available=self.anc_allele_available,
            )
        elif self.stat_prefix == "Q":
            items["statistic"], items["candidates"] = calc_q(
                ref_gts=ref_gts,
                tgt_gts=tgt_gts,
                src_gts_list=src_gts_list,
                pos=pos,
                w=self.w,
                y_list=self.y,
                quantile=self.threshold,
                ploidy=ploidy,
                anc_allele_available=self.anc_allele_available,
            )
        else:
            raise ValueError(
                f"Invalid stat_type: {self.stat_type}. Must be 'U' or 'QXX' (e.g., 'Q95')."
            )

        return [items]

    def process_items(self, items: list[dict[str, Any]]) -> None:
        """
        Processes and writes a single dictionary of feature vectors to the output file.

        Parameters
        ----------
        items : dict[str, Any]
            A dictionary containing feature vectors for a genomic window.
        """
        with open(
            self.output_file, "a"
        ) as f:  # Open in append mode for continuous writing
            lines = []
            for item in items:
                src_pop_str = ",".join(item["src_pop_list"])
                candidates = (
                    "NA"
                    if item["candidates"].size == 0
                    else ",".join(
                        f"{item['chr_name']}:{pos}" for pos in item["candidates"]
                    )
                )

                line = (
                    f"{item['chr_name']}\t{item['start']}\t{item['end']}\t"
                    f"{item['ref_pop']}\t{item['tgt_pop']}\t{src_pop_str}\t"
                    f"{item['nsnps']}\t{item['statistic']}\t{candidates}\n"
                )
                lines.append(line)

            f.writelines(lines)
