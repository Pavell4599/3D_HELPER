import json
from pathlib import Path

settings_path = Path(__file__).resolve().parents[1] / "Data" / "settings" / "modelnet40.json"
with settings_path.open(encoding="utf-8") as settings_file:
    catalog = json.load(settings_file)

class_name = "vase"
profile = {
    **catalog["base_profile"],
    **catalog["classes"][class_name]["settings"],
    "tips": catalog["classes"][class_name]["tips"],
}
for key, value in profile.items():
    print(f"{key} ---> {value}")





