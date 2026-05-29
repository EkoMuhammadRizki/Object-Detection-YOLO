import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import time

import cv2
from ultralytics import YOLO


DEFAULT_MODEL = "yolov8n.pt"
PERSON_CLASS_NAME = "person"


def normalize_class_name(name):
    return str(name).strip().lower().replace("_", " ")


def get_model_names(model):
    names = model.names
    if isinstance(names, dict):
        return {int(class_id): str(label) for class_id, label in names.items()}
    return {class_id: str(label) for class_id, label in enumerate(names)}


def parse_source(source):
    source = str(source).strip()
    if source.isdigit():
        return int(source)
    return source


def parse_target_classes(raw_classes, names, include_person):
    class_ids = set(names.keys())
    lookup = {normalize_class_name(label): class_id for class_id, label in names.items()}

    if raw_classes:
        class_ids = set()
        for token in raw_classes.split(","):
            token = token.strip()
            if not token:
                continue

            if token.isdigit():
                class_id = int(token)
                if class_id not in names:
                    raise ValueError(f"Class id {class_id} tidak ada pada model.")
                class_ids.add(class_id)
                continue

            class_id = lookup.get(normalize_class_name(token))
            if class_id is None:
                sample = ", ".join(list(names.values())[:15])
                raise ValueError(
                    f"Kelas '{token}' tidak ditemukan. Contoh kelas COCO: {sample}"
                )
            class_ids.add(class_id)

    if not include_person:
        person_id = lookup.get(PERSON_CLASS_NAME)
        if person_id in class_ids:
            class_ids.remove(person_id)

    if not class_ids:
        raise ValueError("Tidak ada kelas objek yang dipilih setelah person dikecualikan.")

    return sorted(class_ids)


def count_objects(result, names):
    counts = Counter()
    if result.boxes is None or result.boxes.cls is None:
        return counts

    for class_id in result.boxes.cls.cpu().numpy().astype(int):
        counts[names.get(int(class_id), str(class_id))] += 1
    return counts


def draw_counter_panel(frame, counts, fps=None):
    total = sum(counts.values())
    lines = [f"Total objek: {total}"]

    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")

    for label, amount in counts.most_common(8):
        lines.append(f"{label}: {amount}")

    x, y = 16, 18
    line_height = 28
    width = 330
    height = 18 + (line_height * len(lines))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x + 12, y + 28 + (index * line_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def create_writer(output_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)


def process_image(model, source_path, class_ids, conf, iou, output_dir, show_window):
    image = cv2.imread(str(source_path))
    if image is None:
        raise RuntimeError(f"Gambar tidak dapat dibaca: {source_path}")

    names = get_model_names(model)
    result = model.predict(
        image,
        conf=conf,
        iou=iou,
        classes=class_ids,
        verbose=False,
    )[0]
    annotated = result.plot()
    counts = count_objects(result, names)
    draw_counter_panel(annotated, counts)

    output_path = output_dir / f"hasil_deteksi_{Path(source_path).stem}.jpg"
    cv2.imwrite(str(output_path), annotated)
    if show_window:
        cv2.imshow("YOLO Object Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return output_path


def process_stream(model, source, class_ids, conf, iou, output_dir, show_window):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Sumber video/kamera tidak dapat dibuka: {source}")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Frame pertama tidak dapat dibaca.")

    frame_height, frame_width = frame.shape[:2]
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = input_fps if input_fps and input_fps > 1 else 20.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"hasil_deteksi_objek_{timestamp}.mp4"
    writer = create_writer(output_path, output_fps, (frame_width, frame_height))

    names = get_model_names(model)
    prev_time = time.time()

    while ret:
        start_time = time.time()
        result = model.predict(
            frame,
            conf=conf,
            iou=iou,
            classes=class_ids,
            verbose=False,
        )[0]

        annotated = result.plot()
        counts = count_objects(result, names)
        fps = 1.0 / max(start_time - prev_time, 1e-6)
        prev_time = start_time
        draw_counter_panel(annotated, counts, fps=fps)

        writer.write(annotated)

        if show_window:
            cv2.imshow("YOLO Object Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        ret, frame = cap.read()

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    return output_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Object detection YOLO untuk tugas selain people counting."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Sumber input: 0 untuk kamera utama, path video, atau path gambar.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model YOLO, contoh: yolov8n.pt, yolov8s.pt, atau model custom .pt.",
    )
    parser.add_argument(
        "--classes",
        default="",
        help="Kelas target dipisah koma, contoh: car,motorcycle,bottle. Kosong berarti semua objek selain person.",
    )
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence minimum.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold NMS.")
    parser.add_argument(
        "--include-person",
        action="store_true",
        help="Aktifkan hanya jika ingin memasukkan kelas person. Default: person dikecualikan.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Jangan tampilkan jendela preview, hanya simpan output.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Folder penyimpanan video/gambar hasil deteksi.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = get_model_names(model)
    class_ids = parse_target_classes(args.classes, names, args.include_person)

    source = parse_source(args.source)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if isinstance(source, str) and Path(source).suffix.lower() in image_extensions:
        output_path = process_image(
            model=model,
            source_path=source,
            class_ids=class_ids,
            conf=args.conf,
            iou=args.iou,
            output_dir=output_dir,
            show_window=not args.no_window,
        )
    else:
        output_path = process_stream(
            model=model,
            source=source,
            class_ids=class_ids,
            conf=args.conf,
            iou=args.iou,
            output_dir=output_dir,
            show_window=not args.no_window,
        )

    selected = ", ".join(names[class_id] for class_id in class_ids)
    print(f"Kelas terdeteksi: {selected}")
    print(f"Hasil deteksi disimpan di: {output_path}")


if __name__ == "__main__":
    main()
