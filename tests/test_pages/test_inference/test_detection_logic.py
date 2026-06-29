"""Tests for inference detection — _draw_boxes, _detect_demo edge cases, _to_coco_format."""

import json
from PIL import Image


class TestDrawBoxes:
    def test_draws_on_image(self, mock_session_state):
        from flashstudio.pages.inference.run.detection import _draw_boxes
        img = Image.new("RGB", (640, 480), color="white")
        dets = [
            ["person", "0.95", "[100, 50, 300, 400]"],
            ["car", "0.80", "[200, 100, 500, 300]"],
        ]
        result = _draw_boxes(img, dets)
        assert isinstance(result, Image.Image)
        assert result.size == (640, 480)

    def test_empty_dets_returns_unchanged(self, mock_session_state):
        from flashstudio.pages.inference.run.detection import _draw_boxes
        img = Image.new("RGB", (100, 100), color="blue")
        result = _draw_boxes(img, [])
        assert result.size == (100, 100)

    def test_single_detection(self, mock_session_state):
        from flashstudio.pages.inference.run.detection import _draw_boxes
        img = Image.new("RGB", (200, 200), color="black")
        dets = [["cat", "0.99", "[10, 10, 100, 100]"]]
        result = _draw_boxes(img, dets)
        assert isinstance(result, Image.Image)


class TestDetectDemo:
    def test_respects_conf_threshold(self, mock_session_state):
        from flashstudio.pages.inference.run.detection import _detect_demo
        mock_session_state["infer_conf"] = 0.99
        img = Image.new("RGB", (640, 480), color="red")
        dets = _detect_demo(img)
        for det in dets:
            assert float(det[1]) >= 0.99

    def test_class_filter(self, mock_session_state):
        from flashstudio.pages.inference.run.detection import _detect_demo
        mock_session_state["infer_class_filter"] = ["person"]
        img = Image.new("RGB", (640, 480), color="green")
        dets = _detect_demo(img)
        for det in dets:
            assert det[0] == "person"

    def test_small_image(self, mock_session_state):
        from flashstudio.pages.inference.run.detection import _detect_demo
        img = Image.new("RGB", (10, 10), color="yellow")
        dets = _detect_demo(img)
        assert isinstance(dets, list)

    def test_bbox_within_image_bounds(self, mock_session_state):
        from flashstudio.pages.inference.run.detection import _detect_demo
        w, h = 200, 150
        img = Image.new("RGB", (w, h), color="blue")
        dets = _detect_demo(img)
        for det in dets:
            bbox = json.loads(det[2])
            assert bbox[2] <= w
            assert bbox[3] <= h


class TestToCocoFormatEdgeCases:
    def test_unknown_class(self):
        from flashstudio.pages.inference.run.detection import _to_coco_format
        results = [{
            "name": "test.jpg",
            "dets": [["unknown_class_xyz", "0.90", "[10, 10, 100, 100]"]],
        }]
        coco = _to_coco_format(results)
        assert coco["annotations"][0]["category_id"] == 0

    def test_multiple_images(self):
        from flashstudio.pages.inference.run.detection import _to_coco_format
        results = [
            {"name": "a.jpg", "width": 640, "height": 480,
             "dets": [["person", "0.9", "[0,0,100,100]"]]},
            {"name": "b.jpg", "width": 320, "height": 240,
             "dets": [["car", "0.8", "[0,0,50,50]"], ["person", "0.7", "[10,10,60,60]"]]},
        ]
        coco = _to_coco_format(results)
        assert len(coco["images"]) == 2
        assert len(coco["annotations"]) == 3
        img_ids = {a["image_id"] for a in coco["annotations"]}
        assert len(img_ids) == 2

    def test_default_resolution_used(self):
        from flashstudio.pages.inference.run.detection import _to_coco_format
        results = [{"name": "test.jpg", "dets": []}]
        coco = _to_coco_format(results)
        assert coco["images"][0]["width"] > 0
        assert coco["images"][0]["height"] > 0

    def test_iscrowd_is_zero(self):
        from flashstudio.pages.inference.run.detection import _to_coco_format
        results = [{"name": "t.jpg", "dets": [["person", "0.9", "[0,0,100,100]"]]}]
        coco = _to_coco_format(results)
        assert coco["annotations"][0]["iscrowd"] == 0
