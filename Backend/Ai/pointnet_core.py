# pointnet_core.py
"""
Ядро классификатора: архитектура PointNet + загрузка мешей разных форматов
+ препроцессинг. Импортируется приложением classifier_app.py.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#  Архитектура (1:1 с обучением!)
# =============================================================================
class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, dim=2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# =============================================================================
#  Утилита: триангуляция полигона «веером»
# =============================================================================
def _fan_triangulate(indices):
    """[(v0,v1,v2,v3)] -> [(v0,v1,v2), (v0,v2,v3)]"""
    for k in range(1, len(indices) - 1):
        yield indices[0], indices[k], indices[k + 1]


def _filter_valid_faces(faces, n_verts):
    """Убираем граней с индексами вне диапазона [0, n_verts) — защита от битых файлов."""
    if faces.size == 0:
        return faces
    mask = (faces >= 0).all(axis=1) & (faces < n_verts).all(axis=1)
    return faces[mask]


# =============================================================================
#  Парсеры конкретных форматов (нативные, без зависимостей)
# =============================================================================
def read_off(file_path):
    """Читает .off -> (verts[V,3], faces[F,3])."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip()
    if header == 'OFF':
        n_verts, n_faces, _ = map(int, lines[1].strip().split())
        start = 2
    elif header.startswith('OFF'):
        parts = header.split()
        n_verts, n_faces = int(parts[1]), int(parts[2])
        start = 1
    else:
        raise ValueError(f"Не .off файл: {file_path}")

    verts = np.array(
        [list(map(float, lines[i].strip().split()[:3]))
         for i in range(start, start + n_verts)],
        dtype=np.float32
    )

    faces = []
    for i in range(start + n_verts, start + n_verts + n_faces):
        toks = lines[i].strip().split()
        idxs = list(map(int, toks[1:]))
        faces.extend(_fan_triangulate(idxs))
    return verts, np.array(faces, dtype=np.int64)


def read_obj(file_path):
    """
    Читает .obj -> (verts[V,3], faces[F,3]).
    Поддерживает грани 'f a/b/c', 'f a//c', 'f a', отрицательные индексы.
    Полигоны триангулируются веером.
    """
    verts = []
    faces = []
    with open(file_path, 'r', errors='ignore') as f:
        for line in f:
            if not line or line[0] == '#':
                continue
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == 'v':
                if len(parts) < 4:
                    continue
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == 'f':
                idxs = []
                for tok in parts[1:]:
                    # Отбрасываем /vt и /vn — нам нужен только индекс вершины.
                    vi = tok.split('/')[0]
                    if vi == '':
                        continue
                    i = int(vi)
                    if i < 0:
                        i = len(verts) + i   # отрицательные индексы — с конца
                    else:
                        i = i - 1            # .obj нумерует с 1
                    idxs.append(i)
                if len(idxs) >= 3:
                    faces.extend(_fan_triangulate(idxs))

    verts = np.asarray(verts, dtype=np.float32) if verts else np.zeros((0, 3), dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)
    faces = _filter_valid_faces(faces, len(verts))
    return verts, faces


def read_stl(file_path):
    """
    Читает .stl -> (verts[V,3], faces[F,3]).
    Автоматически определяет ASCII / binary. Работает даже если у меша нет
    вершин в явном виде (STL хранит только треугольники — объединяем их через unique).
    """
    with open(file_path, 'rb') as f:
        head = f.read(80)
        rest = f.read(4)
        is_ascii = False
        try:
            # Иногда ASCII-STL начинается с 'solid', но и binary тоже может.
            text_probe = (head + rest).decode('ascii', errors='ignore').lower()
            if 'facet' in text_probe or text_probe.strip().startswith('solid'):
                # Дополнительная проверка: в ASCII-STL есть ключевое слово 'facet'.
                with open(file_path, 'r', errors='ignore') as ftxt:
                    probe = ftxt.read(2048).lower()
                    is_ascii = 'facet' in probe
        except Exception:
            is_ascii = False

    if is_ascii:
        verts_list = []
        with open(file_path, 'r', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4 and parts[0].lower() == 'vertex':
                    verts_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
        # Каждые 3 вершины — треугольник. Схлопываем дубликаты через unique.
        arr = np.asarray(verts_list, dtype=np.float32)
        if arr.shape[0] < 3:
            return arr, np.zeros((0, 3), dtype=np.int64)
        arr = arr[: (arr.shape[0] // 3) * 3].reshape(-1, 3, 3)  # [F, 3, 3]
        uniq, inv = np.unique(arr.reshape(-1, 3), axis=0, return_inverse=True)
        faces = inv.reshape(-1, 3).astype(np.int64)
        return uniq.astype(np.float32), faces

    # Binary STL
    with open(file_path, 'rb') as f:
        header = f.read(80)
        n_tri = int.from_bytes(f.read(4), 'little')
        # Каждый треугольник: 12 float (нормаль + 3 вершины) + 2 байта атрибутов = 50 байт.
        buf = np.fromfile(f, dtype=np.uint8)
    expected = n_tri * 50
    if expected > buf.size:
        # Файл короче, чем заявлено — режем по факту.
        n_tri = buf.size // 50
        expected = n_tri * 50
    buf = buf[:expected]
    # Собираем только вершины (пропускаем нормаль и атрибуты).
    dt = np.dtype([('normal', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    rec = buf.view(dt).reshape(-1)
    tri = rec['v'].astype(np.float32)                     # [F, 3, 3]
    uniq, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3).astype(np.int64)
    return uniq.astype(np.float32), faces


def read_ply(file_path):
    """
    Читает .ply -> (verts[V,3], faces[F,3]).
    Поддерживает ASCII и binary_little_endian. Игнорирует цвета/нормали,
    берёт только x,y,z и грани (треугольники или полигоны).
    """
    with open(file_path, 'rb') as f:
        first = f.readline().strip()
        if not first.startswith(b'ply'):
            raise ValueError("Не .ply файл")

        fmt = None
        n_verts = 0
        n_faces = 0
        v_props = []     # [(name, type)]
        f_props = []
        current = None
        header_lines = [first]

        while True:
            line = f.readline()
            if not line:
                break
            header_lines.append(line)
            s = line.decode('ascii', errors='ignore').strip()
            if s.startswith('format'):
                fmt = s.split()[1]
            elif s.startswith('element vertex'):
                n_verts = int(s.split()[2])
                current = 'vertex'
            elif s.startswith('element face'):
                n_faces = int(s.split()[2])
                current = 'face'
            elif s.startswith('property'):
                toks = s.split()
                if current == 'vertex' and len(toks) >= 3 and toks[1] != 'list':
                    v_props.append((toks[2], toks[1]))
                elif current == 'face':
                    f_props.append(s)
            elif s.startswith('end_header'):
                break

        if fmt is None:
            raise ValueError("Не найден заголовок 'format'")

        # --- ASCII ---
        if fmt == 'ascii':
            data_text = f.read().decode('ascii', errors='ignore').split('\n')
            vx = vy = vz = None
            names = [p[0] for p in v_props]
            for i in ('x', 'y', 'z'):
                if i not in names:
                    raise ValueError("В .ply нет x/y/z у вершин")
            ix, iy, iz = names.index('x'), names.index('y'), names.index('z')

            verts = np.zeros((n_verts, 3), dtype=np.float32)
            it = iter(data_text)
            read = 0
            while read < n_verts:
                line = next(it, None)
                if line is None:
                    break
                line = line.strip()
                if not line:
                    continue
                vals = line.split()
                verts[read] = [float(vals[ix]), float(vals[iy]), float(vals[iz])]
                read += 1
            verts = verts[:read]

            faces = []
            read_f = 0
            while read_f < n_faces:
                line = next(it, None)
                if line is None:
                    break
                line = line.strip()
                if not line:
                    continue
                vals = line.split()
                n = int(vals[0])
                idxs = [int(v) for v in vals[1:1 + n]]
                if len(idxs) >= 3:
                    faces.extend(_fan_triangulate(idxs))
                read_f += 1
            faces = np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)
            faces = _filter_valid_faces(faces, len(verts))
            return verts, faces

        # --- Binary little endian ---
        if fmt == 'binary_little_endian':
            np_types = {
                'char': 'i1', 'uchar': 'u1', 'int8': 'i1', 'uint8': 'u1',
                'short': 'i2', 'ushort': 'u2', 'int16': 'i2', 'uint16': 'u2',
                'int': 'i4', 'uint': 'u4', 'int32': 'i4', 'uint32': 'u4',
                'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
            }
            dtype_list = [(name, np_types.get(t, 'f4')) for name, t in v_props]
            vdt = np.dtype(dtype_list)
            vraw = np.frombuffer(f.read(vdt.itemsize * n_verts), dtype=vdt, count=n_verts)
            verts = np.stack([vraw['x'], vraw['y'], vraw['z']], axis=1).astype(np.float32)

            # Грани: большинство .ply binary пишут 'property list uchar int vertex_indices'
            faces = []
            # Тип длины и индексов берём из заголовка, но для простоты считаем uchar+int32.
            face_len_dt = np.dtype([('n', 'u1')])
            for _ in range(n_faces):
                nb = f.read(1)
                if not nb:
                    break
                n = int.from_bytes(nb, 'little')
                idxs = np.frombuffer(f.read(4 * n), dtype='<i4', count=n).tolist()
                if len(idxs) >= 3:
                    faces.extend(_fan_triangulate(idxs))
            faces = np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)
            faces = _filter_valid_faces(faces, len(verts))
            return verts, faces

        # --- binary_big_endian ---
        if fmt == 'binary_big_endian':
            np_types = {
                'char': 'i1', 'uchar': 'u1',
                'short': '>i2', 'ushort': '>u2',
                'int': '>i4', 'uint': '>u4',
                'float': '>f4', 'double': '>f8',
            }
            dtype_list = [(name, np_types.get(t, '>f4')) for name, t in v_props]
            vdt = np.dtype(dtype_list)
            vraw = np.frombuffer(f.read(vdt.itemsize * n_verts), dtype=vdt, count=n_verts)
            verts = np.stack([vraw['x'], vraw['y'], vraw['z']], axis=1).astype(np.float32)
            faces = []
            for _ in range(n_faces):
                nb = f.read(1)
                if not nb:
                    break
                n = int.from_bytes(nb, 'big')
                idxs = np.frombuffer(f.read(4 * n), dtype='>i4', count=n).tolist()
                if len(idxs) >= 3:
                    faces.extend(_fan_triangulate(idxs))
            faces = np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)
            faces = _filter_valid_faces(faces, len(verts))
            return verts, faces

        raise ValueError(f"Неподдерживаемый формат .ply: {fmt}")


# =============================================================================
#  Универсальная загрузка через trimesh (для glb/gltf/fbx/dae/3ds/...)
# =============================================================================
def read_with_trimesh(file_path):
    """Пробует загрузить через trimesh. Возвращает (verts, faces)."""
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "Для форматов .glb/.gltf/.fbx/.dae/.3ds/.x3d нужен пакет trimesh.\n"
            "Установите:  pip install trimesh  (или  pip install trimesh[easy] для glTF/FBX)"
        )

    mesh = trimesh.load(file_path, force='mesh', process=False)
    if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
        # Это Scene — сшиваем все меши в один.
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values()]
        )
    if mesh is None or not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
        raise ValueError(f"trimesh не смог извлечь геометрию: {file_path}")

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Если граней нет (точки), но есть vertices — сделаем фиктивные грани,
    # чтобы sampler мог работать (просто проигнорирует площади).
    if faces.size == 0:
        # Псевдо-грани: считаем, что все вершины соединены в один «треугольник» с повтором.
        # Это редкий случай; сэмплер отработает через равномерный fallback.
        raise ValueError("В файле нет граней — облако точек не поддерживается.")

    faces = _filter_valid_faces(faces, len(verts))
    if faces.size == 0:
        raise ValueError("Все грани меша некорректны.")
    return verts, faces


# =============================================================================
#  Реестр форматов и универсальный вход
# =============================================================================
_NATIVE_PARSERS = {
    '.off': read_off,
    '.obj': read_obj,
    '.stl': read_stl,
    '.ply': read_ply,
}

# Форматы, которые пробуем через trimesh (если он установлен).
_TRIMESH_EXT = {
    '.glb', '.gltf', '.fbx', '.dae', '.3ds', '.x3d', '.3mf', '.wrl',
    '.xyz', '.pcd', '.msh', '.gmsh', '.vrml', '.amf',
}

SUPPORTED_FORMATS = sorted(set(_NATIVE_PARSERS) | _TRIMESH_EXT)


def load_mesh(file_path):
    """
    Универсальная загрузка. Автоматически выбирает парсер по расширению.
    Возвращает (verts[V,3] float32, faces[F,3] int64).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    if ext in _NATIVE_PARSERS:
        verts, faces = _NATIVE_PARSERS[ext](file_path)
    elif ext in _TRIMESH_EXT:
        verts, faces = read_with_trimesh(file_path)
    else:
        # Незнакомое расширение — пробуем через trimesh, если он есть.
        verts, faces = read_with_trimesh(file_path)

    if verts is None or len(verts) == 0:
        raise ValueError(f"В файле нет вершин: {file_path}")
    if faces is None or len(faces) == 0:
        raise ValueError(f"В файле нет граней: {file_path}")
    return verts, faces


# =============================================================================
#  Сэмплирование точек с поверхности
# =============================================================================
def sample_points_vectorized(verts, faces, n):
    """Векторно сэмплирует n точек с поверхности меша (как при обучении)."""
    v = verts[faces]
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    if areas.sum() == 0:
        areas = np.ones_like(areas)
    probs = areas / areas.sum()

    sel = v[np.random.choice(len(faces), n, p=probs)]

    r1 = np.random.rand(n, 1)
    r2 = np.random.rand(n, 1)
    s, t = np.minimum(r1, r2), np.maximum(r1, r2)
    c0, c1, c2 = s, (t - s), (1.0 - t)
    pts = c0 * sel[:, 0] + c1 * sel[:, 1] + c2 * sel[:, 2]
    return pts.astype(np.float32)


class Normalize:
    """Центр в 0 и масштаб в единичную сферу (как при обучении)."""
    def __call__(self, pc):
        pc = pc - pc.mean(axis=0)
        m = np.max(np.linalg.norm(pc, axis=1))
        if m > 0:
            pc = pc / m
        return pc


# =============================================================================
#  Готовый классификатор
# =============================================================================
class PointNetClassifier:
    def __init__(self, weights_path, classes, num_points=2048, device=None):
        self.classes = list(classes)
        self.num_points = num_points
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.model = PointNet(num_classes=len(self.classes)).to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v
                 for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()

        self.normalize = Normalize()

    def classify(self, file_path, top_k=3):
        """
        Принимает ЛЮБОЙ поддерживаемый 3D-формат.
        Возвращает (label, conf, [(class, prob), ...]).
        """
        verts, faces = load_mesh(file_path)
        pts = sample_points_vectorized(verts, faces, self.num_points)
        pts = self.normalize(pts)

        tensor = torch.from_numpy(pts).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs_t = F.softmax(logits, dim=1)[0].cpu().numpy()

        top_idx = np.argsort(probs_t)[::-1][:top_k]
        probs = [(self.classes[i], float(probs_t[i])) for i in top_idx]
        label = self.classes[int(top_idx[0])]
        conf = float(probs_t[top_idx[0]])
        return label, conf, probs