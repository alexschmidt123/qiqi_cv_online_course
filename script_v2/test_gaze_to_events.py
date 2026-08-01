import csv
import json
import tempfile
import unittest
from pathlib import Path

from gaze_to_events import build_events, label_gaze_rows, load_library


class PipelineTest(unittest.TestCase):
    def test_polygon_mapping_and_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = {
                "frame_size": {"width": 100, "height": 100},
                "states": [{"state_id": "s1", "slide_id": "slide1", "start_sec": 0, "end_sec": 10,
                    "elements": [
                        {"element_id": "t", "element_type": "slide_title", "label": "title", "polygons": [[[0, 0], [1, 0], [1, .5], [0, .5]]]},
                        {"element_id": "p", "element_type": "paragraph", "label": "paragraph", "polygons": [[[0, .5], [1, .5], [1, 1], [0, 1]]]},
                    ]}]
            }
            (root / "lib.json").write_text(json.dumps(library))
            with (root / "gaze.csv").open("w", newline="") as f:
                w = csv.writer(f); w.writerow(["timestamp_sec", "x_bl", "y_bl"])
                w.writerows([[0, 50, 90], [1, 50, 90], [2, 50, 10]])
            width, height, states = load_library(root / "lib.json")
            rows = label_gaze_rows(root / "gaze.csv", width, height, states, "bottom-left")
            events = build_events(rows, "u1")
            self.assertEqual([e["learning_element"] for e in events], ["title", "paragraph"])
            self.assertEqual([e["duration"] for e in events], [2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
