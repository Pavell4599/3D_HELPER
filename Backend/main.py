
import json
import os
import shutil
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = PROJECT_ROOT / "Backend" / "Ai"
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from pointnet_core import PointNetClassifier, SUPPORTED_FORMATS  # noqa: E402

CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
    "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
    "flower", "glass", "guitar", "keyboard", "lamp", "laptop", "mantel",
    "monitor", "night", "person", "piano", "plant", "radio", "range", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv", "vase",
    "wardrobe", "xbox",
]
WEIGHTS_PATH = Path(os.environ.get(
    "POINTNET_WEIGHTS", str(AI_ROOT / "best_pointnet_modelnet40.pth")
)).expanduser()
SETTINGS_PATH = PROJECT_ROOT / "Data" / "settings" / "modelnet40.json"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_BATCH_FILES = 100
MAX_BATCH_UPLOAD_BYTES = 500 * 1024 * 1024

_classifier: PointNetClassifier | None = None

app = FastAPI(
    title="3D Helper — API",
    description=(
        "Локальная классификация 3D-моделей PointNet на ModelNet40 и "
        "ориентировочные рекомендации для FDM-печати. Настройки не являются "
        "проверенными профилями слайсера."
    ),
    version="1.0.0",
)

cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5500,http://127.0.0.1:5500,"
        "http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PathRequest(BaseModel):
    path: str


class ScanRequest(BaseModel):
    path: str
    recursive: bool = True
    formats: list[str] = Field(default_factory=lambda: list(SUPPORTED_FORMATS))


class OrganizeRequest(ScanRequest):
    output_directory: str


class UndoRequest(BaseModel):
    operation_id: str


def operations_directory() -> Path:
    return Path(os.environ.get(
        "THREE_D_HELPER_OPERATIONS_DIR",
        str(Path.home() / ".3d_helper" / "operations"),
    )).expanduser()


def operation_file(operation_id: str) -> Path:
    try:
        normalized_id = str(uuid.UUID(operation_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Некорректный номер операции") from exc
    return operations_directory() / f"{normalized_id}.json"


def save_operation(operation: dict[str, Any]) -> None:
    directory = operations_directory()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{operation['operation_id']}.json"
    temporary = directory / f".{operation['operation_id']}.tmp"
    try:
        temporary.write_text(
            json.dumps(operation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_settings() -> dict[str, Any]:
    with SETTINGS_PATH.open(encoding="utf-8") as settings_file:
        return json.load(settings_file)


def get_classifier() -> PointNetClassifier:
    global _classifier
    if _classifier is None:
        if not WEIGHTS_PATH.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"Не найдены веса PointNet: {WEIGHTS_PATH}",
            )
        try:
            _classifier = PointNetClassifier(str(WEIGHTS_PATH), CLASSES)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Не удалось загрузить веса PointNet: {exc}",
            ) from exc
    return _classifier


def class_profile(class_name: str) -> dict[str, Any]:
    catalog = load_settings()
    normalized_name = class_name.casefold()
    matching_name = next(
        (name for name in catalog["classes"] if name.casefold() == normalized_name),
        None,
    )
    if matching_name is None:
        raise HTTPException(status_code=404, detail="Неизвестный класс ModelNet40")
    class_data = catalog["classes"][matching_name]
    advice = class_data["совет"]
    return {
        **catalog["base_profile"],
        **class_data["settings"],
        "class": matching_name,
        "class_name_ru": class_data["name_ru"],
        "совет": advice,
        "tips": [advice],
        "disclaimer": catalog["disclaimer"],
    }


def validate_directory(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Папка не найдена: {path}")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Это не папка: {path}")
    return path


def normalized_formats(formats: list[str]) -> set[str]:
    allowed = {extension.lower() for extension in SUPPORTED_FORMATS}
    requested = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in formats
    }
    unsupported = requested - allowed
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail=f"Неподдерживаемые форматы 3D-файлов: {', '.join(sorted(unsupported))}",
        )
    return requested


def find_model_files(
    directory: Path,
    recursive: bool,
    extensions: set[str],
    skip_class_folders: bool = False,
) -> list[Path]:
    files: list[Path] = []
    if recursive:
        for current_directory, child_directories, file_names in os.walk(
            directory, followlinks=False
        ):
            child_directories[:] = [
                child
                for child in child_directories
                if not (Path(current_directory) / child).is_symlink()
                and not (skip_class_folders and child.casefold() in {
                    class_name.casefold() for class_name in CLASSES
                })
            ]
            files.extend(
                Path(current_directory) / file_name
                for file_name in file_names
                if Path(file_name).suffix.lower() in extensions
            )
    else:
        files.extend(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
    return sorted(files, key=lambda path: str(path).casefold())


def classify_path(file_path: Path) -> dict[str, Any]:
    try:
        label, confidence, predictions = get_classifier().classify(
            str(file_path), top_k=3
        )
    except HTTPException:
        raise
    except (OSError, ImportError, RuntimeError, ValueError) as exc:
        return {"path": str(file_path), "error": str(exc)}
    return {
        "path": str(file_path),
        "class": label,
        "class_name_ru": class_profile(label)["class_name_ru"],
        "confidence": confidence,
        "top_predictions": [
            {
                "class": name,
                "class_name_ru": class_profile(name)["class_name_ru"],
                "confidence": probability,
            }
            for name, probability in predictions
        ],
        "print_settings": class_profile(label),
    }


def classify_directory(
    request: ScanRequest, skip_class_folders: bool = False
) -> list[dict[str, Any]]:
    source_directory = validate_directory(request.path)
    extensions = normalized_formats(request.formats)
    paths = find_model_files(
        source_directory,
        request.recursive,
        extensions,
        skip_class_folders=skip_class_folders,
    )
    return [classify_path(path) for path in paths]


def organize_in_place(request: ScanRequest) -> dict[str, Any]:
    root = validate_directory(request.path)
    results = classify_directory(request, skip_class_folders=True)
    classified = [result for result in results if "class" in result]
    moved: list[dict[str, str]] = []
    created_directories: set[str] = set()
    operation_id = str(uuid.uuid4())

    try:
        for result in classified:
            source = Path(result["path"])
            target_directory = root / result["class"]
            if not target_directory.exists():
                target_directory.mkdir()
                created_directories.add(str(target_directory))
            destination = unique_destination(target_directory, source)
            shutil.move(str(source), str(destination))
            moved.append({
                "source": str(source),
                "destination": str(destination),
                "class": result["class"],
            })

        operation = {
            "operation_id": operation_id,
            "root": str(root),
            "status": "applied",
            "files": moved,
            "created_directories": sorted(created_directories),
        }
        save_operation(operation)
    except (OSError, HTTPException) as exc:
        rollback_errors: list[str] = []
        for entry in reversed(moved):
            try:
                shutil.move(entry["destination"], entry["source"])
            except OSError as rollback_error:
                rollback_errors.append(str(rollback_error))
        for directory in sorted(created_directories, key=len, reverse=True):
            try:
                Path(directory).rmdir()
            except OSError:
                pass
        detail = f"Не удалось завершить раскладку: {exc}"
        if rollback_errors:
            detail += f". Ошибки возврата файлов: {'; '.join(rollback_errors)}"
        raise HTTPException(status_code=500, detail=detail) from exc

    return {
        "status": "completed",
        "operation_id": operation_id,
        "scanned": len(results),
        "moved": len(moved),
        "errors": [result for result in results if "error" in result],
        "files": moved,
        "message": "Файлы разложены по папкам классов. Операцию можно отменить.",
    }


def undo_organization(operation_id: str) -> dict[str, Any]:
    manifest_path = operation_file(operation_id)
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Операция не найдена")
    try:
        operation = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось прочитать журнал операции: {exc}",
        ) from exc
    if operation.get("status") != "applied":
        raise HTTPException(status_code=409, detail="Операция уже отменена")

    for entry in operation["files"]:
        if not Path(entry["destination"]).is_file():
            raise HTTPException(
                status_code=409,
                detail=f"Перемещённый файл не найден: {entry['destination']}",
            )
        if Path(entry["source"]).exists():
            raise HTTPException(
                status_code=409,
                detail=f"Исходный путь уже занят: {entry['source']}",
            )

    restored: list[dict[str, str]] = []
    try:
        for entry in reversed(operation["files"]):
            Path(entry["source"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(entry["destination"], entry["source"])
            restored.append(entry)
        operation["status"] = "undone"
        save_operation(operation)
    except OSError as exc:
        rollback_errors: list[str] = []
        for entry in reversed(restored):
            try:
                shutil.move(entry["source"], entry["destination"])
            except OSError as rollback_error:
                rollback_errors.append(str(rollback_error))
        detail = f"Не удалось завершить отмену: {exc}"
        if rollback_errors:
            detail += f". Ошибки восстановления раскладки: {'; '.join(rollback_errors)}"
        raise HTTPException(status_code=500, detail=detail) from exc

    for directory in operation.get("created_directories", []):
        try:
            Path(directory).rmdir()
        except OSError:
            pass

    return {
        "status": "undone",
        "operation_id": operation["operation_id"],
        "restored": len(restored),
        "message": "Файлы возвращены в исходные папки.",
    }


def list_operations() -> list[dict[str, Any]]:
    directory = operations_directory()
    if not directory.exists():
        return []
    operations: list[dict[str, Any]] = []
    for manifest_path in sorted(
        directory.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        try:
            operation = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Не удалось прочитать журнал {manifest_path.name}: {exc}",
            ) from exc
        operations.append({
            "operation_id": operation["operation_id"],
            "root": operation["root"],
            "status": operation["status"],
            "file_count": len(operation["files"]),
        })
    return operations


def ensure_disjoint_directories(source: Path, destination: Path) -> None:
    if source == destination or source in destination.parents or destination in source.parents:
        raise HTTPException(
            status_code=400,
            detail="Папки источника и назначения не должны пересекаться.",
        )


def unique_destination(directory: Path, source: Path) -> Path:
    destination = directory / source.name
    index = 1
    while destination.exists():
        destination = directory / f"{source.stem}_{index}{source.suffix}"
        index += 1
    return destination


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "3D Helper — API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_weights_available": WEIGHTS_PATH.is_file(),
        "model_loaded": _classifier is not None,
        "class_count": len(CLASSES),
    }


@app.get("/classes")
def list_classes() -> dict[str, Any]:
    return {
        "dataset": "ModelNet40",
        "classes": CLASSES,
        "class_names_ru": {
            class_name: class_profile(class_name)["class_name_ru"]
            for class_name in CLASSES
        },
        "supported_formats": SUPPORTED_FORMATS,
    }


@app.get("/settings")
def list_settings() -> dict[str, Any]:
    return load_settings()


@app.get("/settings/{class_name}")
def get_settings(class_name: str) -> dict[str, Any]:
    return class_profile(class_name)


@app.post("/classify-file")
def classify_local_file(request: PathRequest) -> dict[str, Any]:
    file_path = Path(request.path).expanduser().resolve()
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=422, detail="Неподдерживаемый формат 3D-файла")
    return classify_path(file_path)


@app.post("/scan")
def scan_directory(request: ScanRequest) -> dict[str, Any]:
    results = classify_directory(request)
    return {
        "directory": str(Path(request.path).expanduser().resolve()),
        "scanned": len(results),
        "results": results,
    }


@app.post("/organize")
def preview_organization(request: OrganizeRequest) -> dict[str, Any]:
    source_directory = validate_directory(request.path)
    output_directory = Path(request.output_directory).expanduser().resolve()
    ensure_disjoint_directories(source_directory, output_directory)
    results = classify_directory(request)
    plan = [
        {
            "source": result["path"],
            "destination": str(output_directory / result["class"] / Path(result["path"]).name),
            "class": result["class"],
            "confidence": result["confidence"],
        }
        for result in results
        if "class" in result
    ]
    return {
        "status": "preview",
        "source_directory": str(source_directory),
        "output_directory": str(output_directory),
        "planned": len(plan),
        "errors": [result for result in results if "error" in result],
        "plan": plan,
        "message": "Файлы не изменены. Отправьте POST /organize/apply, чтобы скопировать их.",
    }


@app.post("/organize-in-place")
def organize_selected_folder(request: ScanRequest) -> dict[str, Any]:
    return organize_in_place(request)


@app.get("/operations")
def get_operations() -> dict[str, Any]:
    return {"operations": list_operations()}


@app.post("/undo/{operation_id}")
def undo_selected_organization(operation_id: str) -> dict[str, Any]:
    return undo_organization(operation_id)


@app.post("/organize/apply")
def apply_organization(request: OrganizeRequest) -> dict[str, Any]:
    source_directory = validate_directory(request.path)
    output_directory = Path(request.output_directory).expanduser().resolve()
    ensure_disjoint_directories(source_directory, output_directory)
    results = classify_directory(request)

    copied: list[dict[str, str]] = []
    errors = [result for result in results if "error" in result]
    try:
        for result in results:
            if "class" not in result:
                continue
            source = Path(result["path"])
            destination_directory = output_directory / result["class"]
            destination_directory.mkdir(parents=True, exist_ok=True)
            destination = unique_destination(destination_directory, source)
            shutil.copy2(source, destination)
            copied.append({
                "source": str(source),
                "destination": str(destination),
                "class": result["class"],
            })
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Копирование остановлено после {len(copied)} файла(ов): {exc}",
        ) from exc

    return {
        "status": "completed",
        "copied": len(copied),
        "errors": errors,
        "files": copied,
        "message": "Исходные файлы оставлены без изменений.",
    }


@app.post("/analyze-model")
async def analyze_model(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=422, detail="Неподдерживаемый формат 3D-файла")

    temporary_path: Path | None = None
    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Размер загруженного файла превышает 100 МБ")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(contents)
            temporary_path = Path(temporary_file.name)

        result = await run_in_threadpool(classify_path, temporary_path)
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        return {
            "model_name": file.filename,
            "class": result["class"],
            "confidence": result["confidence"],
            "top_predictions": result["top_predictions"],
            "settings": result["print_settings"],
        }
    finally:
        await file.close()
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


@app.post("/organize-upload")
async def organize_uploaded_models(files: list[UploadFile] = File(...)) -> FileResponse:
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"За один раз можно отсортировать не более {MAX_BATCH_FILES} файлов.",
        )

    temporary_directory = Path(tempfile.mkdtemp(prefix="3d-helper-sort-"))
    archive_path = temporary_directory / "sorted_models.zip"
    total_bytes = 0
    archived_files = 0
    errors: list[str] = []
    used_archive_names: set[str] = set()
    response_ready = False

    try:
        with zipfile.ZipFile(
            archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for upload in files:
                file_name = (upload.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
                suffix = Path(file_name).suffix.lower()
                if suffix not in SUPPORTED_FORMATS:
                    continue

                temporary_model = temporary_directory / f"{uuid.uuid4()}{suffix}"
                file_bytes = 0
                with temporary_model.open("wb") as model_file:
                    while True:
                        chunk = await upload.read(1024 * 1024)
                        if not chunk:
                            break
                        file_bytes += len(chunk)
                        total_bytes += len(chunk)
                        if file_bytes > MAX_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"Файл {file_name} превышает ограничение 100 МБ.",
                            )
                        if total_bytes > MAX_BATCH_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail="Общий размер файлов превышает ограничение 500 МБ.",
                            )
                        model_file.write(chunk)

                result = await run_in_threadpool(classify_path, temporary_model)
                if "error" in result:
                    errors.append(f"{file_name}: {result['error']}")
                    temporary_model.unlink()
                    continue

                class_name = result["class"]
                archive_name = f"{class_name}/{file_name}"
                stem = Path(file_name).stem
                index = 1
                while archive_name.casefold() in used_archive_names:
                    archive_name = f"{class_name}/{stem}_{index}{suffix}"
                    index += 1
                used_archive_names.add(archive_name.casefold())
                archive.write(temporary_model, archive_name)
                temporary_model.unlink()
                archived_files += 1

            if archived_files == 0:
                raise HTTPException(
                    status_code=422,
                    detail="Не удалось классифицировать ни одной модели из выбранной папки.",
                )
            if errors:
                archive.writestr("_errors.txt", "\n".join(errors))

        response = FileResponse(
            path=archive_path,
            media_type="application/zip",
            filename="3d-helper-sorted.zip",
            background=BackgroundTask(
                shutil.rmtree, temporary_directory, ignore_errors=True
            ),
        )
        response_ready = True
        return response
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Не удалось создать архив: {exc}"
        ) from exc
    finally:
        for upload in files:
            await upload.close()
        if not response_ready:
            shutil.rmtree(temporary_directory, ignore_errors=True)
