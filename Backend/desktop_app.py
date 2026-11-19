import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from fastapi import HTTPException

from Backend.main import (
    ScanRequest,
    classify_directory,
    list_operations,
    organize_in_place,
    undo_organization,
)


class OrganizerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("3D Helper — классификация моделей")
        self.root.geometry("940x620")
        self.root.minsize(720, 440)
        self.folder = tk.StringVar()
        self.status = tk.StringVar(value="Выберите папку с 3D-моделями.")
        self.operation_id = tk.StringVar()
        self.results: queue.Queue = queue.Queue()
        self.busy = False

        self._build_ui()
        self._refresh_operations()
        self.root.after(100, self._poll_results)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)

        folder_row = ttk.Frame(container)
        folder_row.pack(fill="x")
        ttk.Entry(folder_row, textvariable=self.folder).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(folder_row, text="Выбрать папку…", command=self._choose_folder).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(folder_row, text="Сканировать", command=self._scan).pack(
            side="left", padx=(8, 0)
        )

        columns = ("file", "class", "confidence")
        self.table = ttk.Treeview(
            container, columns=columns, show="headings", selectmode="browse"
        )
        self.table.heading("file", text="Файл")
        self.table.heading("class", text="Распознанный класс")
        self.table.heading("confidence", text="Уверенность")
        self.table.column("file", width=500)
        self.table.column("class", width=210)
        self.table.column("confidence", width=110, anchor="center")
        self.table.pack(fill="both", expand=True, pady=10)

        actions = ttk.Frame(container)
        actions.pack(fill="x")
        self.organize_button = ttk.Button(
            actions,
            text="Разложить по папкам классов",
            command=self._organize,
        )
        self.organize_button.pack(side="left")
        ttk.Label(actions, text="Операция для отмены:").pack(side="left", padx=(18, 5))
        self.operations = ttk.Combobox(
            actions,
            textvariable=self.operation_id,
            state="readonly",
            width=38,
        )
        self.operations.pack(side="left")
        self.undo_button = ttk.Button(
            actions,
            text="Вернуть файлы",
            command=self._undo,
        )
        self.undo_button.pack(side="left", padx=(8, 0))

        ttk.Label(
            container,
            textvariable=self.status,
            wraplength=900,
            justify="left",
        ).pack(fill="x", pady=(10, 0))

    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Выберите папку с 3D-моделями")
        if folder:
            self.folder.set(folder)

    def _request(self) -> ScanRequest:
        folder = self.folder.get().strip()
        if not folder:
            raise ValueError("Сначала выберите папку.")
        return ScanRequest(path=folder, recursive=True)

    def _run_async(self, work) -> None:
        if self.busy:
            return
        self.busy = True
        self.organize_button.configure(state="disabled")
        self.undo_button.configure(state="disabled")
        self.status.set("Обработка…")

        def run() -> None:
            try:
                self.results.put(("success", work()))
            except (HTTPException, OSError, ValueError) as exc:
                detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                self.results.put(("error", str(detail)))
            except (ImportError, KeyError, RuntimeError, TypeError) as exc:
                self.results.put(("error", f"Ошибка обработки: {exc}"))

        threading.Thread(target=run, daemon=True).start()

    def _scan(self) -> None:
        try:
            request = self._request()
        except ValueError as exc:
            messagebox.showwarning("Папка не выбрана", str(exc))
            return

        def scan():
            return classify_directory(request, skip_class_folders=True)

        self._run_async(scan)

    def _organize(self) -> None:
        try:
            request = self._request()
        except ValueError as exc:
            messagebox.showwarning("Папка не выбрана", str(exc))
            return
        if not messagebox.askyesno(
            "Подтвердите раскладку",
            "Файлы будут перемещены из выбранной папки в подпапки по классам. "
            "Их можно будет вернуть кнопкой «Вернуть файлы». Продолжить?",
        ):
            return
        self._run_async(lambda: organize_in_place(request))

    def _undo(self) -> None:
        operation_id = self.operation_id.get()
        if not operation_id:
            messagebox.showwarning("Нет операции", "Выберите операцию для отмены.")
            return
        if not messagebox.askyesno(
            "Подтвердите отмену",
            "Переместить файлы этой операции обратно в исходные папки?",
        ):
            return
        self._run_async(lambda: undo_organization(operation_id))

    def _refresh_operations(self) -> None:
        available = [
            item["operation_id"]
            for item in list_operations()
            if item["status"] == "applied"
        ]
        self.operations.configure(values=available)
        if self.operation_id.get() not in available:
            self.operation_id.set(available[0] if available else "")

    def _show_scan_results(self, results: list[dict]) -> None:
        self.table.delete(*self.table.get_children())
        for result in results:
            if "error" in result:
                category = f"Ошибка: {result['error']}"
                confidence = ""
            else:
                category = result["class_name_ru"]
                confidence = f"{result['confidence']:.1%}"
            self.table.insert(
                "",
                "end",
                values=(Path(result["path"]).name, category, confidence),
            )
        failed = sum("error" in result for result in results)
        self.status.set(
            f"Найдено моделей: {len(results)}. Ошибок классификации: {failed}."
        )

    def _poll_results(self) -> None:
        try:
            result_type, value = self.results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_results)
            return

        self.busy = False
        self.organize_button.configure(state="normal")
        self.undo_button.configure(state="normal")
        if result_type == "error":
            self.status.set(f"Ошибка: {value}")
            messagebox.showerror("Не удалось выполнить операцию", value)
        elif isinstance(value, list):
            self._show_scan_results(value)
        else:
            self.status.set(value["message"])
            if value.get("operation_id"):
                self._refresh_operations()
                self.operation_id.set(value["operation_id"])
            elif value.get("status") == "undone":
                self._refresh_operations()
        self.root.after(100, self._poll_results)


def main() -> None:
    root = tk.Tk()
    OrganizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
