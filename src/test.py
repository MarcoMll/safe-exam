from pathlib import Path
from ultralytics import YOLO
from utils.paths_initializer import get_paths

project_paths = get_paths()
path_to_test_image: Path = project_paths.RAW_DIR / "student_with_phone_test_3.png"
model_path = project_paths.MODELS_DIR / "yolo26s.pt"

model = YOLO(model_path)

results = model.predict(
    source=path_to_test_image,
    classes=[67],
)

for r in results:
    #print(r.boxes)
    r.save(f"{project_paths.PROCESSED_DIR}/phone_result.png")