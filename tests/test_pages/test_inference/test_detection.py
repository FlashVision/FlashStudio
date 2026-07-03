"""Tests for flashstudio.pages.inference.run.detection — COCO format, draw helpers."""



class TestToCocoFormat:
    def test_empty_results(self):
        from flashstudio.pages.inference.run.detection import _to_coco_format

        coco = _to_coco_format([])
        assert coco["images"] == []
        assert coco["annotations"] == []
        assert len(coco["categories"]) == 80

    def test_single_result(self):
        from flashstudio.pages.inference.run.detection import _to_coco_format

        results = [{
            "name": "test.jpg",
            "width": 640,
            "height": 480,
            "dets": [["person", "0.95", "[100, 50, 300, 400]"]],
        }]
        coco = _to_coco_format(results)
        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 1
        assert coco["images"][0]["file_name"] == "test.jpg"
        assert coco["annotations"][0]["score"] == 0.95

    def test_multiple_detections(self):
        from flashstudio.pages.inference.run.detection import _to_coco_format

        results = [{
            "name": "test.jpg",
            "dets": [
                ["person", "0.95", "[100, 50, 300, 400]"],
                ["car", "0.80", "[200, 100, 500, 300]"],
            ],
        }]
        coco = _to_coco_format(results)
        assert len(coco["annotations"]) == 2

    def test_bbox_converted_to_xywh(self):
        from flashstudio.pages.inference.run.detection import _to_coco_format

        results = [{
            "name": "test.jpg",
            "dets": [["person", "0.90", "[100, 50, 300, 250]"]],
        }]
        coco = _to_coco_format(results)
        bbox = coco["annotations"][0]["bbox"]
        assert bbox == [100, 50, 200, 200]

    def test_area_calculated(self):
        from flashstudio.pages.inference.run.detection import _to_coco_format

        results = [{
            "name": "test.jpg",
            "dets": [["person", "0.90", "[100, 50, 300, 250]"]],
        }]
        coco = _to_coco_format(results)
        area = coco["annotations"][0]["area"]
        assert area == 200 * 200

    def test_categories_are_coco80(self):
        from flashstudio.pages.inference.run.detection import _to_coco_format

        coco = _to_coco_format([])
        assert len(coco["categories"]) == 80
        names = [c["name"] for c in coco["categories"]]
        assert "person" in names
        assert "car" in names

    def test_annotation_ids_unique(self):
        from flashstudio.pages.inference.run.detection import _to_coco_format

        results = [
            {"name": "a.jpg", "dets": [["person", "0.9", "[0,0,100,100]"]]},
            {"name": "b.jpg", "dets": [["car", "0.8", "[0,0,50,50]"]]},
        ]
        coco = _to_coco_format(results)
        ids = [a["id"] for a in coco["annotations"]]
        assert len(ids) == len(set(ids))


class TestDetectDemo:
    def test_returns_list(self, mock_session_state):
        from flashstudio.pages.inference.run.detection import _detect_demo
        from PIL import Image

        img = Image.new("RGB", (640, 480), color="blue")
        dets = _detect_demo(img)
        assert isinstance(dets, list)

    def test_detections_format(self, mock_session_state):
        from flashstudio.pages.inference.run.detection import _detect_demo
        from PIL import Image

        img = Image.new("RGB", (640, 480), color="red")
        dets = _detect_demo(img)
        for det in dets:
            assert len(det) == 3
            assert isinstance(det[0], str)
            assert isinstance(det[1], str)

    def test_deterministic_for_same_image(self, mock_session_state):
        from flashstudio.pages.inference.run.detection import _detect_demo
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="green")
        dets1 = _detect_demo(img)
        dets2 = _detect_demo(img)
        assert dets1 == dets2

    def test_sorted_by_confidence(self, mock_session_state):
        from flashstudio.pages.inference.run.detection import _detect_demo
        from PIL import Image

        img = Image.new("RGB", (640, 480), color="blue")
        dets = _detect_demo(img)
        if len(dets) > 1:
            confs = [float(d[1]) for d in dets]
            assert confs == sorted(confs, reverse=True)


class TestInferencePageImports:
    def test_render_inference_page(self):
        from flashstudio.pages.inference import render_inference_page
        assert callable(render_inference_page)

    def test_page_module(self):
        from flashstudio.pages.inference.page import render_inference_page
        assert callable(render_inference_page)

    def test_detect_function(self):
        from flashstudio.pages.inference.run.detection import _detect
        assert callable(_detect)

    def test_draw_boxes(self):
        from flashstudio.pages.inference.run.detection import _draw_boxes
        assert callable(_draw_boxes)
