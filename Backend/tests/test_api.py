import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import Backend.main as backend


class StubClassifier:
    def classify(self, file_path: str, top_k: int = 3):
        return "chair", 0.8, [("chair", 0.8), ("stool", 0.15), ("table", 0.05)]


class BackendApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(backend.app)
        self.classifier_patch = patch.object(
            backend, "get_classifier", return_value=StubClassifier()
        )
        self.classifier_patch.start()
        self.addCleanup(self.classifier_patch.stop)
        self.operations_directory = tempfile.TemporaryDirectory()
        operations_patch = patch.object(
            backend, "operations_directory",
            return_value=Path(self.operations_directory.name),
        )
        operations_patch.start()
        self.addCleanup(operations_patch.stop)
        self.addCleanup(self.operations_directory.cleanup)

    def test_settings_cover_all_classifier_classes(self):
        response = self.client.get("/settings")
        self.assertEqual(response.status_code, 200)
        catalog = response.json()
        self.assertEqual(set(catalog["classes"]), set(backend.CLASSES))
        self.assertEqual(len(catalog["classes"]), 40)
        vase = self.client.get("/settings/vase").json()
        self.assertEqual(vase["совет"], "Печатайте вертикально без заполнения; спиральный режим подходит только для одной стенки.")
        self.assertEqual(vase["tips"], [vase["совет"]])
        self.assertEqual(vase["class_name_ru"], "Ваза")
        self.assertEqual(vase["Материал"], "PLA")
        self.assertEqual(vase["Температура сопла"], "210 °C")
        self.assertEqual(vase["Температура стола"], "60 °C")
        self.assertEqual(vase["Высота слоя"], "0,20 мм")
        self.assertEqual(vase["Заполнение"], "0 %")

    def test_every_class_has_concrete_settings_and_one_short_advice(self):
        catalog = self.client.get("/settings").json()
        common_fields = set(catalog["base_profile"])
        for class_data in catalog["classes"].values():
            self.assertTrue(class_data["совет"])
            effective_settings = {
                **catalog["base_profile"],
                **class_data["settings"],
            }
            self.assertTrue(common_fields.issubset(effective_settings))
            self.assertIn("Заполнение", effective_settings)
            self.assertIn("Поддержки", effective_settings)
            self.assertIn("Кайма", effective_settings)
            self.assertIn("мм/с", effective_settings["Скорость печати"])

    def test_organize_preview_does_not_change_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "models"
            output_directory = root / "sorted"
            source_directory.mkdir()
            model_file = source_directory / "sample.stl"
            model_file.write_text("test mesh", encoding="utf-8")

            response = self.client.post("/organize", json={
                "path": str(source_directory),
                "output_directory": str(output_directory),
            })

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["status"], "preview")
            self.assertTrue(model_file.is_file())
            self.assertFalse(output_directory.exists())

    def test_apply_copies_files_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "models"
            output_directory = root / "sorted"
            source_directory.mkdir()
            model_file = source_directory / "sample.stl"
            model_file.write_text("test mesh", encoding="utf-8")

            response = self.client.post("/organize/apply", json={
                "path": str(source_directory),
                "output_directory": str(output_directory),
            })

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["copied"], 1)
            self.assertTrue(model_file.is_file())
            self.assertTrue((output_directory / "chair" / "sample.stl").is_file())

    def test_in_place_organization_can_be_undone(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested_directory = root / "incoming"
            nested_directory.mkdir()
            model_file = nested_directory / "sample.stl"
            model_file.write_text("test mesh", encoding="utf-8")

            organize_response = self.client.post("/organize-in-place", json={
                "path": str(root),
            })

            self.assertEqual(organize_response.status_code, 200, organize_response.text)
            operation = organize_response.json()
            self.assertEqual(operation["moved"], 1)
            self.assertFalse(model_file.exists())
            organized_file = root / "chair" / "sample.stl"
            self.assertTrue(organized_file.is_file())

            undo_response = self.client.post(f"/undo/{operation['operation_id']}")

            self.assertEqual(undo_response.status_code, 200, undo_response.text)
            self.assertEqual(undo_response.json()["restored"], 1)
            self.assertTrue(model_file.is_file())
            self.assertEqual(model_file.read_text(encoding="utf-8"), "test mesh")
            self.assertFalse(organized_file.exists())
            self.assertEqual(self.client.get("/operations").json()["operations"][0]["status"], "undone")

    def test_undo_refuses_to_overwrite_recreated_original(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_file = root / "sample.stl"
            model_file.write_text("original", encoding="utf-8")

            operation = self.client.post("/organize-in-place", json={
                "path": str(root),
                "recursive": False,
            }).json()
            model_file.write_text("new file", encoding="utf-8")

            response = self.client.post(f"/undo/{operation['operation_id']}")

            self.assertEqual(response.status_code, 409)
            self.assertEqual(model_file.read_text(encoding="utf-8"), "new file")
            self.assertTrue((root / "chair" / "sample.stl").is_file())


if __name__ == "__main__":
    unittest.main()
