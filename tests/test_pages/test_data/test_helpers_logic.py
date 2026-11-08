"""Tests for data helpers — _find_root edge cases, _find_images nested, _find_labels nested."""

import io
import os
import zipfile

from flashstudio.pages.data import helpers as data_helpers


class TestDatasetMirrors:
    def test_legacy_registry_is_patched_to_working_host(self, monkeypatch):
        urls = {
            "trainval": "https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            "test": "https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        }
        monkeypatch.setattr(data_helpers, "_dataset_registry_urls", lambda _id: urls)

        data_helpers._apply_dataset_mirror("voc2007", 0)

        assert urls["trainval"] == data_helpers._DATASET_URL_MIRRORS["voc2007"]["trainval"][0]
        assert urls["test"] == data_helpers._DATASET_URL_MIRRORS["voc2007"]["test"][0]

    def test_legacy_registry_can_switch_mirror(self, monkeypatch):
        urls = {"trainval": "https://host.robots.ox.ac.uk/x.tar"}
        monkeypatch.setattr(data_helpers, "_dataset_registry_urls", lambda _id: urls)

        data_helpers._apply_dataset_mirror("voc2007", 1)

        assert urls["trainval"] == data_helpers._DATASET_URL_MIRRORS["voc2007"]["trainval"][1]

    def test_new_flashdet_url_lists_are_left_alone(self, monkeypatch):
        urls = {
            "trainval": [
                "https://primary.example/x.tar",
                "https://mirror.example/x.tar",
            ]
        }
        monkeypatch.setattr(data_helpers, "_dataset_registry_urls", lambda _id: urls)

        data_helpers._apply_dataset_mirror("voc2007", 0)

        assert urls["trainval"] == [
            "https://primary.example/x.tar",
            "https://mirror.example/x.tar",
        ]

    def test_download_retries_next_mirror(self, monkeypatch):
        urls = {"trainval": "https://host.robots.ox.ac.uk/x.tar"}
        monkeypatch.setattr(data_helpers, "_dataset_registry_urls", lambda _id: urls)

        calls = []

        def fake_download(dataset_id, output_dir):
            calls.append(urls["trainval"])
            if len(calls) == 1:
                raise RuntimeError("Download failed: https://host.robots.ox.ac.uk/x.tar")
            return output_dir

        monkeypatch.setattr(data_helpers, "download_dataset", fake_download, raising=False)

        result = data_helpers._download_dataset_with_mirrors("voc2007", "/tmp/out")

        assert result == "/tmp/out"
        assert calls == data_helpers._DATASET_URL_MIRRORS["voc2007"]["trainval"]

    def test_project_page_is_shown_as_manual_link(self, monkeypatch):
        import urllib.request

        downloaded = []
        monkeypatch.setattr(
            urllib.request, "urlretrieve",
            lambda *args, **kwargs: downloaded.append(args),
        )
        messages = []
        monkeypatch.setattr(
            data_helpers.st, "info",
            lambda message: messages.append(message),
            raising=False,
        )

        data_helpers._run_external_download({
            "name": "SKU-110K",
            "url": "https://github.com/eg4000/SKU110K_CVPR19",
        })

        assert downloaded == []
        assert messages
        assert "Manual download" in messages[0]

    def test_download_does_not_retry_non_download_errors(self, monkeypatch):
        urls = {"trainval": "https://host.robots.ox.ac.uk/x.tar"}
        monkeypatch.setattr(data_helpers, "_dataset_registry_urls", lambda _id: urls)

        calls = []

        def fake_download(dataset_id, output_dir):
            calls.append(dataset_id)
            raise RuntimeError("Could not find VOC dataset layout")

        monkeypatch.setattr(data_helpers, "download_dataset", fake_download, raising=False)

        try:
            data_helpers._download_dataset_with_mirrors("voc2007", "/tmp/out")
        except RuntimeError as exc:
            assert "VOC dataset layout" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

        assert calls == ["voc2007"]



class TestDatasetLoadingFlow:
    @staticmethod
    def _make_split(root, split, name):
        images = root / split / "images"
        images.mkdir(parents=True)
        (images / "image.jpg").write_bytes(b"x")
        annotation = root / split / "_annotations.coco.json"
        annotation.write_text(
            '{"images":[{"id":1,"file_name":"image.jpg","width":10,"height":10}],'
            '"annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":[1,1,2,2]}],'
            '"categories":[{"id":1,"name":"' + name + '","supercategory":"object"}]}'
        )
        return images

    def test_find_split_base_prefers_requested_split(self, tmp_path):
        root = tmp_path / "dataset"
        (root / "train" / "images").mkdir(parents=True)
        (root / "valid" / "images").mkdir(parents=True)

        assert data_helpers._find_split_base(str(root), "train") == str(root / "train")
        assert data_helpers._find_split_base(str(root), "val") == str(root / "valid")

    def test_find_split_base_ignores_label_directory(self, tmp_path):
        root = tmp_path / "dataset"
        images = root / "images" / "train2017"
        labels = root / "labels" / "train2017"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / "a.jpg").write_bytes(b"x")
        (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        assert data_helpers._find_split_base(str(root), "train") == str(images)

    def test_image_dir_descends_into_images_folder(self, tmp_path):
        images = tmp_path / "dataset" / "train" / "images"
        images.mkdir(parents=True)
        (images / "a.jpg").write_bytes(b"x")

        assert data_helpers._image_dir(str(tmp_path / "dataset" / "train")) == str(images)

    def test_use_existing_finds_nested_validation_and_annotations(
        self, tmp_path, mock_session_state, monkeypatch
    ):
        root = tmp_path / "dataset"
        train_images = self._make_split(root, "train", "cat")
        val_images = self._make_split(root, "valid", "cat")
        monkeypatch.setattr(data_helpers.st, "rerun", lambda: None)

        data_helpers._use_existing(
            str(root), {"name": "Test", "classes": 1}
        )

        assert mock_session_state["train_img_path"] == str(train_images)
        assert mock_session_state["val_img_path"] == str(val_images)
        assert os.path.isfile(mock_session_state["train_ann_path"])
        assert os.path.isfile(mock_session_state["val_ann_path"])
        assert mock_session_state["class_names"] == "cat"
        assert mock_session_state["num_classes"] == 1

    def test_finalize_official_coco_layout(self, tmp_path, mock_session_state):
        root = tmp_path / "coco"
        train_images = root / "train2017"
        val_images = root / "val2017"
        train_images.mkdir(parents=True)
        val_images.mkdir(parents=True)
        (train_images / "a.jpg").write_bytes(b"x")
        (val_images / "b.jpg").write_bytes(b"x")
        ann_dir = root / "annotations"
        ann_dir.mkdir()
        for name, split in (("instances_train2017.json", "train"),
                            ("instances_val2017.json", "val")):
            (ann_dir / name).write_text(
                '{"images":[{"id":1,"file_name":"image.jpg","width":10,"height":10}],'
                '"annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":[1,1,2,2]}],'
                '"categories":[{"id":1,"name":"cat","supercategory":"object"}]}'
            )

        data_helpers._finalize_downloaded_dataset(str(root), "COCO")

        assert mock_session_state["train_img_path"] == str(train_images)
        assert mock_session_state["val_img_path"] == str(val_images)
        assert os.path.isfile(os.path.join(str(train_images), "_annotations.coco.json"))
        assert os.path.isfile(os.path.join(str(val_images), "_annotations.coco.json"))
        assert mock_session_state["class_names"] == "cat"
        assert mock_session_state["num_classes"] == 1

    def test_extract_upload_train_valid_nested_layout(
        self, tmp_path, mock_session_state, monkeypatch
    ):
        annotation = (
            '{"images":[{"id":1,"file_name":"image.jpg","width":10,"height":10}],'
            '"annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":[1,1,2,2]}],'
            '"categories":[{"id":1,"name":"cat","supercategory":"object"}]}'
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("dataset/train/images/a.jpg", b"x")
            zf.writestr("dataset/train/_annotations.coco.json", annotation)
            zf.writestr("dataset/valid/images/b.jpg", b"x")
            zf.writestr("dataset/valid/_annotations.coco.json", annotation)

        class Upload:
            name = "dataset.zip"

            def read(self):
                return buffer.getvalue()

            def seek(self, *_args):
                return None

        class Progress:
            def progress(self, *_args, **_kwargs):
                return None

        errors = []
        monkeypatch.setattr(data_helpers, "_get_data_root", lambda: str(tmp_path / "data"))
        monkeypatch.setattr(data_helpers.st, "progress", lambda *_a, **_k: Progress())
        monkeypatch.setattr(data_helpers.st, "error", lambda message: errors.append(message))
        monkeypatch.setattr(data_helpers.st, "rerun", lambda: None)

        data_helpers._extract_upload(Upload(), "train")

        assert errors == []
        root = tmp_path / "data" / "uploaded" / "train" / "dataset"
        assert mock_session_state["dataset_output_path"] == str(root)
        assert mock_session_state["train_img_path"] == str(root / "train" / "images")
        assert mock_session_state["val_img_path"] == str(root / "valid" / "images")
        assert os.path.isfile(str(root / "train" / "images" / "_annotations.coco.json"))
        assert os.path.isfile(str(root / "valid" / "images" / "_annotations.coco.json"))
        assert mock_session_state["class_names"] == "cat"

    def test_convert_yolo_layout_with_images_split_dirs(self, tmp_path):
        from PIL import Image

        root = tmp_path / "yolo"
        images = root / "images" / "train2017"
        labels = root / "labels" / "train2017"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (12, 12), "white").save(images / f"{i}.jpg")
            (labels / f"{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")

        out = tmp_path / "converted"
        stats = data_helpers._convert_yolo_layout(
            str(root), str(out), class_names=["cat"]
        )

        assert stats
        assert os.path.isfile(str(out / "train" / "_annotations.coco.json"))
        assert os.path.isfile(str(out / "valid" / "_annotations.coco.json"))

    def test_convert_yolo_layout_preserves_year_suffix_splits(self, tmp_path):
        from PIL import Image

        root = tmp_path / "yolo"
        for split, count in (("train2017", 4), ("val2017", 2)):
            images = root / "images" / split
            labels = root / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for i in range(count):
                Image.new("RGB", (12, 12), "white").save(images / f"{i}.jpg")
                (labels / f"{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")

        out = tmp_path / "converted"
        stats = data_helpers._convert_yolo_layout(
            str(root), str(out), class_names=["cat"]
        )

        assert stats
        assert stats["train"]["images"] == 4
        assert stats["valid"]["images"] == 2

    def test_extract_upload_voc_nested_layout(
        self, tmp_path, mock_session_state, monkeypatch
    ):
        from PIL import Image

        image_bytes = io.BytesIO()
        Image.new("RGB", (12, 12), "white").save(image_bytes, format="JPEG")
        xml = (
            "<annotation><filename>0001.jpg</filename>"
            "<object><name>cat</name><bndbox>"
            "<xmin>1</xmin><ymin>1</ymin><xmax>8</xmax><ymax>8</ymax>"
            "</bndbox></object></annotation>"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("VOCdevkit/VOC2007/Annotations/0001.xml", xml)
            zf.writestr("VOCdevkit/VOC2007/JPEGImages/0001.jpg", image_bytes.getvalue())

        class Upload:
            name = "voc.zip"

            def read(self):
                return buffer.getvalue()

            def seek(self, *_args):
                return None

        class Progress:
            def progress(self, *_args, **_kwargs):
                return None

        errors = []
        monkeypatch.setattr(data_helpers, "_get_data_root", lambda: str(tmp_path / "data"))
        monkeypatch.setattr(data_helpers.st, "progress", lambda *_a, **_k: Progress())
        monkeypatch.setattr(data_helpers.st, "error", lambda message: errors.append(message))
        monkeypatch.setattr(data_helpers.st, "rerun", lambda: None)

        data_helpers._extract_upload(Upload(), "train")

        from flashstudio.pages.data._common import _auto_detect_classes

        assert errors == []
        assert mock_session_state["train_img_path"]
        assert os.path.isfile(mock_session_state["train_ann_path"])
        assert _auto_detect_classes() == ["cat"]

    def test_load_native_downloaded_sets_name_and_paths(
        self, tmp_path, mock_session_state
    ):
        root = tmp_path / "voc2007"
        train_images = self._make_split(root, "train", "cat")
        val_images = self._make_split(root, "valid", "cat")

        ok = data_helpers._load_native_downloaded(
            "voc2007", str(root), display_name="VOC 2007", classes=20
        )

        assert ok
        assert mock_session_state["dataset_name"] == "VOC 2007"
        assert mock_session_state["dataset_id"] == "voc2007"
        assert mock_session_state["dataset_classes"] == 20
        assert mock_session_state["train_img_path"] == str(train_images)
        assert mock_session_state["val_img_path"] == str(val_images)

    def test_use_existing_clears_stale_validation_path(
        self, tmp_path, mock_session_state, monkeypatch
    ):
        root = tmp_path / "dataset"
        self._make_split(root, "train", "cat")
        mock_session_state["val_img_path"] = "/old/validation"
        monkeypatch.setattr(data_helpers.st, "rerun", lambda: None)

        data_helpers._use_existing(
            str(root), {"name": "Test", "classes": 1}
        )

        # The stale path is replaced and a real validation split is created.
        val_path = mock_session_state.get("val_img_path", "")
        assert val_path and val_path != "/old/validation"
        assert os.path.isdir(val_path)
        assert os.path.isdir(mock_session_state["train_img_path"])


class TestFindRootSplitSafety:
    def test_does_not_return_train_split_as_root(self, tmp_path):
        root = tmp_path / "upload"
        (root / "train").mkdir(parents=True)
        (root / "valid").mkdir()
        assert data_helpers._find_root(str(root)) == str(root)

    def test_descends_through_wrapper_folders(self, tmp_path):
        root = tmp_path / "upload"
        dataset = root / "VOCdevkit" / "VOC2007"
        dataset.mkdir(parents=True)
        (dataset / "Annotations").mkdir()
        (dataset / "JPEGImages").mkdir()
        assert data_helpers._find_root(str(root)) == str(dataset)

    def test_single_train_folder_is_not_descended(self, tmp_path):
        root = tmp_path / "upload"
        images = root / "train" / "images"
        images.mkdir(parents=True)
        (images / "a.jpg").write_bytes(b"x")
        assert data_helpers._find_root(str(root)) == str(root)

    def test_internal_temp_folder_is_not_descended(self, tmp_path):
        root = tmp_path / "upload"
        images = root / "_yolo_1yi7iw6y" / "valid" / "images"
        images.mkdir(parents=True)
        (images / "a.jpg").write_bytes(b"x")
        (images / "_annotations.coco.json").write_text("{}")
        assert data_helpers._find_root(str(root)) == str(root)


class TestFindRootEdgeCases:
    def test_empty_dir_returns_base(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_root
        assert _find_root(str(tmp_path)) == str(tmp_path)

    def test_single_file_returns_base(self, tmp_path):
        (tmp_path / "readme.txt").write_text("x")
        from flashstudio.pages.data.helpers import _find_root
        assert _find_root(str(tmp_path)) == str(tmp_path)

    def test_nested_single_subdir(self, tmp_path):
        inner = tmp_path / "dataset" / "images"
        inner.mkdir(parents=True)
        from flashstudio.pages.data.helpers import _find_root
        result = _find_root(str(tmp_path))
        assert result == str(tmp_path / "dataset")


class TestFindImagesDeepNesting:
    def test_finds_images_in_deeply_nested_dir(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_images
        deep = tmp_path / "a" / "b" / "c" / "images"
        deep.mkdir(parents=True)
        for name in ["1.jpg", "2.jpg", "3.png"]:
            (deep / name).write_bytes(b"\xff")
        result = _find_images(str(tmp_path))
        assert result is not None
        assert "images" in result

    def test_prefers_images_dir_name(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_images
        other = tmp_path / "other"
        other.mkdir()
        (other / "1.jpg").write_bytes(b"\xff")

        imgs = tmp_path / "images"
        imgs.mkdir()
        for name in ["a.jpg", "b.jpg"]:
            (imgs / name).write_bytes(b"\xff")

        result = _find_images(str(tmp_path))
        assert result is not None
        assert "images" in result

    def test_empty_images_dir_skipped(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_images
        (tmp_path / "images").mkdir()
        assert _find_images(str(tmp_path)) is None


class TestFindAnnotationEdgeCases:
    def test_prefers_coco_json_over_random(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_ann
        (tmp_path / "_annotations.coco.json").write_text("{}")
        (tmp_path / "something.json").write_text("{}")
        result = _find_ann(str(tmp_path))
        assert "annotation" in result.lower() or "coco" in result.lower()

    def test_finds_nested_annotation(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_ann
        nested = tmp_path / "data" / "annotations"
        nested.mkdir(parents=True)
        (nested / "instances_train.coco.json").write_text("{}")
        result = _find_ann(str(tmp_path))
        assert result is not None
        assert result.endswith(".json")


class TestFindLabelsEdgeCases:
    def test_nested_labels_dir(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_labels
        labels = tmp_path / "data" / "labels"
        labels.mkdir(parents=True)
        (labels / "001.txt").write_text("0 0.5 0.5 0.3 0.3")
        result = _find_labels(str(tmp_path))
        assert result is not None
        assert "labels" in result

    def test_labels_dir_case_insensitive(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_labels
        labels = tmp_path / "Labels"
        labels.mkdir()
        (labels / "001.txt").write_text("0 0.5 0.5 0.3 0.3")
        result = _find_labels(str(tmp_path))
        assert result is not None
