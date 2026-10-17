
import os
import csv
import shutil
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Опциональный drag&drop — если установлен tkinterdnd2.
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

from pointnet_core import PointNetClassifier, SUPPORTED_FORMATS


# ---------- Конфиг по умолчанию ----------
WEIGHTS_DEFAULT = "best_pointnet_modelnet10.pth"
CSV_DEFAULT     = "metadata_modelnet10.csv"


# =============================================================================
#  Утилиты
# =============================================================================
def load_classes_from_csv(csv_path):
    """
    Достаёт отсортированный список классов — ровно как при обучении.
    Порядок sorted() критичен: он должен совпадать с class_to_idx из обучения.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[~df['object_path'].astype(str).str.contains('.DS', na=True)]
    df = df[~df['class'].astype(str).str.contains('.DS', na=True)]
    df = df.dropna(subset=['object_path'])
    return sorted(df['class'].astype(str).unique())


def safe_copy(src, dst_dir):
    """
    Копирует src в dst_dir с защитой от дубликатов.

    Правила:
      • файла с таким именем нет          -> копируем;
      • есть, размеры совпадают           -> это тот же файл, ПРОПУСКАЕМ;
      • есть, размеры разные              -> добавляем _1, _2, ... пока не найдём
                                             свободное имя или файл с тем же размером.
    Возвращает ('copied' | 'skipped', итоговый_путь).
    """
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src)
    dst = os.path.join(dst_dir, base)
    src_size = os.path.getsize(src)

    if os.path.exists(dst):
        if os.path.getsize(dst) == src_size:
            return 'skipped', dst

        stem, ext = os.path.splitext(base)
        i = 1
        while True:
            candidate = os.path.join(dst_dir, f"{stem}_{i}{ext}")
            if not os.path.exists(candidate):
                dst = candidate
                break
            if os.path.getsize(candidate) == src_size:
                return 'skipped', candidate
            i += 1

    shutil.copy2(src, dst)
    return 'copied', dst


# =============================================================================
#  Приложение
# =============================================================================
class ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PointNet 3D Classifier — ModelNet10")
        self.root.geometry("1020x760")
        self.root.minsize(900, 640)

        self.classifier = None
        self.classes = None
        self.file_paths = []          # список абсолютных путей добавленных файлов
        self._worker = None

        # ---- Переменные интерфейса ----
        self.var_weights   = tk.StringVar(value=WEIGHTS_DEFAULT)
        self.var_csv       = tk.StringVar(value=CSV_DEFAULT)
        self.var_outdir    = tk.StringVar(value=os.path.abspath("classified_output"))
        self.var_recursive = tk.BooleanVar(value=True)
        self.var_use_undo  = tk.BooleanVar(value=True)
        self.var_status    = tk.StringVar(value="Модель не загружена")

        self._build_ui()
        self._register_dnd()
        self.root.after(150, self._try_autoload)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        pad = {'padx': 8, 'pady': 4}

        # --- 1. Модель ---
        f1 = ttk.LabelFrame(self.root, text="1. Обученная модель")
        f1.pack(fill='x', **pad)

        ttk.Label(f1, text="Веса (.pth):").grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(f1, textvariable=self.var_weights, width=64).grid(row=0, column=1, sticky='we', padx=4)
        ttk.Button(f1, text="...", width=3, command=self._pick_weights).grid(row=0, column=2, padx=4)

        ttk.Label(f1, text="CSV с классами:").grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(f1, textvariable=self.var_csv, width=64).grid(row=1, column=1, sticky='we', padx=4)
        ttk.Button(f1, text="...", width=3, command=self._pick_csv).grid(row=1, column=2, padx=4)

        ttk.Button(f1, text="Загрузить модель", command=self._load_model).grid(
            row=2, column=1, sticky='e', padx=4, pady=(2, 6))
        ttk.Label(f1, textvariable=self.var_status, foreground='#0a6').grid(
            row=2, column=0, sticky='w', padx=6)
        f1.columnconfigure(1, weight=1)

        # --- 2. Файлы ---
        f2 = ttk.LabelFrame(self.root, text="2. 3D-файлы для классификации")
        f2.pack(fill='both', expand=True, **pad)

        btns = ttk.Frame(f2)
        btns.pack(fill='x', padx=6, pady=4)
        ttk.Button(btns, text="Добавить файлы...", command=self._add_files).pack(side='left', padx=4)
        ttk.Button(btns, text="Добавить папку...", command=self._add_folder).pack(side='left', padx=4)
        ttk.Button(btns, text="Удалить выбранные", command=self._remove_selected).pack(side='left', padx=4)
        ttk.Button(btns, text="Очистить список",   command=self._clear_files).pack(side='left', padx=4)
        ttk.Checkbutton(btns, text="Рекурсивно в подпапках",
                        variable=self.var_recursive).pack(side='right', padx=6)

        # Таблица файлов
        cols = ('file', 'size')
        self.tree = ttk.Treeview(f2, columns=cols, show='headings',
                                 height=8, selectmode='extended')
        self.tree.heading('file', text='Файл')
        self.tree.heading('size', text='Размер, КБ')
        self.tree.column('file', width=720, anchor='w')
        self.tree.column('size', width=100, anchor='e')
        self.tree.pack(fill='both', expand=True, padx=6, pady=4)

        # Скроллбар для таблицы
        sb = ttk.Scrollbar(f2, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.place(relx=1.0, rely=0.25, relheight=0.6, anchor='ne')

        if DND_AVAILABLE:
            hint = ("Перетащите 3D-файлы прямо в окно.  Поддерживаются: "
                    + ", ".join(SUPPORTED_FORMATS))
        else:
            hint = ("Поддерживаются: " + ", ".join(SUPPORTED_FORMATS)
                    + "   |   Совет: pip install tkinterdnd2 — включит drag&drop")
        ttk.Label(f2, text=hint, foreground='#666', wraplength=980,
                  justify='left').pack(anchor='w', padx=8, pady=(0, 6))

        # --- 3. Выход ---
        f3 = ttk.LabelFrame(self.root, text="3. Куда складывать результат")
        f3.pack(fill='x', **pad)

        ttk.Label(f3, text="Папка вывода:").grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(f3, textvariable=self.var_outdir, width=64).grid(row=0, column=1, sticky='we', padx=4)
        ttk.Button(f3, text="...", width=3, command=self._pick_outdir).grid(row=0, column=2, padx=4)
        f3.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            f3,
            text="Писать undo-лог (CSV) — чтобы можно было откатить раскладку",
            variable=self.var_use_undo
        ).grid(row=1, column=1, sticky='w', padx=4, pady=(0, 6))

        # --- 4. Запуск + лог ---
        f4 = ttk.Frame(self.root)
        f4.pack(fill='both', expand=True, **pad)

        run_bar = ttk.Frame(f4)
        run_bar.pack(fill='x', padx=8, pady=6)
        self.btn_run = ttk.Button(run_bar, text="Классифицировать и разложить",
                                  command=self._run)
        self.btn_run.pack(side='left', padx=4)
        self.progress = ttk.Progressbar(run_bar, mode='determinate')
        self.progress.pack(side='left', fill='x', expand=True, padx=8)

        log_frame = ttk.Frame(f4)
        log_frame.pack(fill='both', expand=True, padx=8, pady=(0, 8))
        self.log = tk.Text(log_frame, height=14, wrap='word')
        self.log.pack(side='left', fill='both', expand=True)
        log_sb = ttk.Scrollbar(log_frame, orient='vertical', command=self.log.yview)
        log_sb.pack(side='right', fill='y')
        self.log.configure(yscrollcommand=log_sb.set)

        self._log("Готово. Укажите веса + CSV и добавьте 3D-файлы (или перетащите их в окно).")

    # ------------------------------------------------------ drag&drop -------
    def _register_dnd(self):
        if not DND_AVAILABLE:
            return
        try:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self._on_drop)
        except Exception as e:
            self._log(f"Drag&drop не активировался: {e}")

    def _on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        added = 0
        for p in paths:
            if os.path.isfile(p) and self._is_supported(p):
                added += self._append_file(p)
            elif os.path.isdir(p):
                added += self._scan_dir(p)
        self._log(f"Drag&drop: добавлено {added} файл(ов).")

    # --------------------------------------------------- пикеры путей -------
    def _pick_weights(self):
        p = filedialog.askopenfilename(
            title="Веса модели",
            filetypes=[("PyTorch weights", "*.pth"), ("Все файлы", "*.*")]
        )
        if p:
            self.var_weights.set(p)

    def _pick_csv(self):
        p = filedialog.askopenfilename(
            title="CSV с классами",
            filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")]
        )
        if p:
            self.var_csv.set(p)

    def _pick_outdir(self):
        p = filedialog.askdirectory(title="Папка вывода")
        if p:
            self.var_outdir.set(p)

    # ------------------------------------------------- работа со списком ----
    def _is_supported(self, path):
        return os.path.splitext(path)[1].lower() in SUPPORTED_FORMATS

    def _append_file(self, path):
        """Добавляет один файл в список. Возвращает 1 если добавлен, иначе 0."""
        abs_path = os.path.abspath(path)
        if abs_path in self.file_paths:
            return 0
        self.file_paths.append(abs_path)
        try:
            size_kb = os.path.getsize(abs_path) / 1024.0
        except OSError:
            size_kb = 0.0
        self.tree.insert('', 'end', iid=abs_path, values=(abs_path, f"{size_kb:.1f}"))
        return 1

    def _scan_dir(self, folder):
        recursive = self.var_recursive.get()
        n = 0
        if recursive:
            for root, _, files in os.walk(folder):
                for f in files:
                    if self._is_supported(f):
                        n += self._append_file(os.path.join(root, f))
        else:
            for f in os.listdir(folder):
                fp = os.path.join(folder, f)
                if os.path.isfile(fp) and self._is_supported(f):
                    n += self._append_file(fp)
        return n

    def _add_files(self):
        patterns = " ".join(f"*{e}" for e in SUPPORTED_FORMATS)
        paths = filedialog.askopenfilenames(
            title="Выберите 3D-модели",
            filetypes=[
                ("Все поддерживаемые 3D", patterns),
                ("Mesh (.off .obj .stl .ply)", "*.off *.obj *.stl *.ply"),
                ("glTF / FBX / Collada (.glb .gltf .fbx .dae)", "*.glb *.gltf *.fbx *.dae"),
                ("Все файлы", "*.*"),
            ]
        )
        n = sum(self._append_file(p) for p in paths)
        self._log(f"Добавлено файлов: {n}  (пропущено дубликатов: {len(paths) - n})")

    def _add_folder(self):
        d = filedialog.askdirectory(title="Папка с 3D-файлами")
        if not d:
            return
        n = self._scan_dir(d)
        self._log(f"Добавлено из папки: {n}")

    def _remove_selected(self):
        selected = self.tree.selection()
        if not selected:
            return
        for iid in selected:
            if iid in self.file_paths:
                self.file_paths.remove(iid)
            self.tree.delete(iid)
        self._log(f"Удалено из списка: {len(selected)}")

    def _clear_files(self):
        self.file_paths.clear()
        for i in self.tree.get_children():
            self.tree.delete(i)
        self._log("Список файлов очищен.")

    # ------------------------------------------------- работа с моделью -----
    def _try_autoload(self):
        if os.path.isfile(self.var_weights.get()) and os.path.isfile(self.var_csv.get()):
            self._load_model()

    def _load_model(self):
        weights = self.var_weights.get().strip()
        csvp    = self.var_csv.get().strip()

        if not os.path.isfile(weights):
            messagebox.showerror("Ошибка", f"Не найден файл весов:\n{weights}")
            return
        if not os.path.isfile(csvp):
            messagebox.showerror("Ошибка", f"Не найден CSV:\n{csvp}")
            return

        try:
            self.classes = load_classes_from_csv(csvp)
        except Exception as e:
            messagebox.showerror("Ошибка CSV", f"Не удалось прочитать классы:\n{e}")
            return

        try:
            self.classifier = PointNetClassifier(weights, self.classes)
        except Exception as e:
            messagebox.showerror("Ошибка модели", f"Не удалось загрузить веса:\n{e}")
            return

        dev = self.classifier.device
        self.var_status.set(f"OK | классов: {len(self.classes)} | устройство: {dev}")
        self._log(f"Модель загружена. Классы ({len(self.classes)}): {', '.join(self.classes)}")
        self._log(f"Устройство: {dev}")

    # ------------------------------------------------- запуск обработки ----
    def _run(self):
        if self.classifier is None:
            messagebox.showwarning("Модель не загружена",
                                   "Сначала нажмите 'Загрузить модель'.")
            return
        if not self.file_paths:
            messagebox.showwarning("Нет файлов", "Добавьте 3D-файлы или папку.")
            return

        outdir = self.var_outdir.get().strip()
        if not outdir:
            messagebox.showwarning("Нет папки", "Укажите папку вывода.")
            return

        self.btn_run.config(state='disabled')
        self.progress['value'] = 0
        self.progress['maximum'] = len(self.file_paths)

        self._worker = threading.Thread(
            target=self._process_files,
            args=(list(self.file_paths), outdir),
            daemon=True
        )
        self._worker.start()

    def _process_files(self, files, outdir):
        os.makedirs(outdir, exist_ok=True)

        # Подготовка undo-лога
        log_path = None
        log_file = None
        log_writer = None
        if self.var_use_undo.get():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(outdir, f"_undo_log_{stamp}.csv")
            log_file = open(log_path, 'w', newline='', encoding='utf-8')
            log_writer = csv.writer(log_file)
            log_writer.writerow(['source', 'destination', 'status', 'label', 'confidence'])

        stats = {'copied': 0, 'skipped': 0, 'error': 0}
        total = len(files)

        for i, fp in enumerate(files, 1):
            try:
                label, conf, topk = self.classifier.classify(fp, top_k=3)
                dst_dir = os.path.join(outdir, label)
                status, dst = safe_copy(fp, dst_dir)
                stats[status] += 1

                top_str = ", ".join(f"{c}:{p:.1%}" for c, p in topk)
                self._log(f"[{i}/{total}] {os.path.basename(fp)} → {label} "
                          f"({conf:.1%})  [{status}]")
                self._log(f"        ТОП-3: {top_str}")
                self._log(f"        → {dst}")

                if log_writer is not None:
                    log_writer.writerow([fp, dst, status, label, f"{conf:.6f}"])
                    log_file.flush()

            except Exception as e:
                stats['error'] += 1
                self._log(f"[{i}/{total}] ОШИБКА {os.path.basename(fp)}: {e}")
                if log_writer is not None:
                    log_writer.writerow([fp, '', 'error', '', str(e)])
                    log_file.flush()

            # Обновление прогресса в главном потоке
            self.root.after(0, lambda v=i: self.progress.configure(value=v))

        if log_file is not None:
            log_file.close()
            self._log(f"Undo-лог: {log_path}")

        self._log("")
        self._log(f"Готово! Всего: {total}  |  Скопировано: {stats['copied']}  |  "
                  f"Пропущено (уже было): {stats['skipped']}  |  Ошибок: {stats['error']}")
        self._log(f"Результат: {outdir}")

        # Возвращаем кнопку и показываем итог
        self.root.after(0, lambda: self.btn_run.config(state='normal'))
        self.root.after(0, lambda: messagebox.showinfo(
            "Готово",
            f"Обработано: {total}\n"
            f"Скопировано: {stats['copied']}\n"
            f"Пропущено (дубликаты): {stats['skipped']}\n"
            f"Ошибок: {stats['error']}"
        ))

    # ------------------------------------------------------- лог -----------
    def _log(self, text):
        def _append():
            self.log.insert('end', text + "\n")
            self.log.see('end')
        self.root.after(0, _append)


# =============================================================================
#  Точка входа
# =============================================================================
def main():
    root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    ClassifierApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()