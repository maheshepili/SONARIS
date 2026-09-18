import unittest

import numpy as np

from src.anomaly.extract_candidate_regions import regions_from_heatmap


class CandidateRegionTests(unittest.TestCase):
    def test_regions_include_stats_and_discard_tiny_components(self):
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[20:30, 20:30] = 0.01
        heatmap[40:43, 40:43] = 0.01

        regions = regions_from_heatmap(
            heatmap,
            threshold=0.003,
            min_region_area=10,
        )

        self.assertEqual(
            regions,
            [
                {
                    "bbox": [20, 20, 10, 10],
                    "area": 100,
                    "max_error": 0.01,
                    "mean_error": 0.01,
                }
            ],
        )

    def test_sonar_artifact_regions_are_excluded(self):
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[20:30, :5] = 0.01

        regions = regions_from_heatmap(
            heatmap,
            threshold=0.003,
            min_region_area=10,
        )

        self.assertEqual(regions, [])

    def test_narrow_regions_abutting_inner_edge_masks_are_excluded(self):
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[20:60, 5:7] = 0.01
        heatmap[20:60, 93:95] = 0.01

        regions = regions_from_heatmap(
            heatmap,
            threshold=0.003,
            min_region_area=10,
        )

        self.assertEqual(regions, [])

    def test_narrow_interior_region_is_preserved(self):
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[20:60, 20:22] = 0.01

        regions = regions_from_heatmap(
            heatmap,
            threshold=0.003,
            min_region_area=10,
        )

        self.assertEqual(regions[0]["bbox"], [20, 20, 2, 40])

    def test_regions_tighten_to_the_strongest_anomaly_concentration(self):
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[20:40, 20:40] = 0.004
        heatmap[26:34, 26:34] = 0.01

        regions = regions_from_heatmap(
            heatmap,
            threshold=0.003,
            min_region_area=10,
        )

        self.assertEqual(regions[0]["bbox"], [26, 26, 8, 8])

    def test_tightening_can_be_disabled_for_full_component_bounds(self):
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[20:40, 20:40] = 0.004
        heatmap[26:34, 26:34] = 0.01

        regions = regions_from_heatmap(
            heatmap,
            threshold=0.003,
            min_region_area=10,
            tighten=False,
        )

        self.assertEqual(regions[0]["bbox"], [20, 20, 20, 20])


if __name__ == "__main__":
    unittest.main()
