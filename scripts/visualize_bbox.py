import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def draw_bounding_boxes(image_path: str, detections: list, output_path: str):
    """
    Vẽ các bounding box lên ảnh và lưu kết quả.

    Args:
        image_path (str): Đường dẫn tới ảnh gốc.
        detections (list): Danh sách các phát hiện chứa bbox, class_name, confidence.
        output_path (str): Đường dẫn lưu ảnh kết quả.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Thử nải font chữ mặc định hoặc font hệ thống nếu có
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    # Màu sắc cố định cho các lớp nhận diện
    colors = [
        "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A",
        "#92E7C9", "#30B350", "#00D4BB", "#2C99A8", "#00C2FF", "#344593",
        "#6473E4", "#0018EC", "#8438FF", "#520085", "#CB38FF", "#FF9DAE",
        "#C33764", "#7B1FA2"
    ]

    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        xmin, ymin, xmax, ymax = bbox
        class_name = det.get("class_name", "object")
        conf = det.get("confidence", 0.0)
        class_id = det.get("class_id", 0)

        color = colors[class_id % len(colors)]
        label = f"{class_name} {conf:.2f}"

        # Vẽ hình chữ nhật bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        # Tính kích thước label để vẽ nền cho chữ
        text_bbox = draw.textbbox((xmin, ymin), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Vị trí nhãn chữ (ở trên bbox hoặc ở dưới nếu chạm viền trên)
        text_ymin = max(0, ymin - text_height - 4)
        draw.rectangle(
            [xmin, text_ymin, xmin + text_width + 8, text_ymin + text_height + 4],
            fill=color
        )
        draw.text((xmin + 4, text_ymin + 2), label, fill="white", font=font)

    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"[Thành công] Đã lưu ảnh đã vẽ bounding box tại: {output_path}")


def visualize_from_json(json_path: str, target_image_name: str, output_path: str = None):
    """
    Trực quan hóa bounding box của ảnh được chỉ định từ file JSON.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File JSON không tồn tại: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_entry = None
    for res in data.get("results", []):
        if res.get("image_name") == target_image_name:
            target_entry = res
            break

    if not target_entry:
        print(f"[Lỗi] Không tìm thấy ảnh '{target_image_name}' trong file JSON.")
        return

    img_abs_path = target_entry.get("absolute_path")
    detections = target_entry.get("detections", [])

    print(f"Ảnh: {target_image_name}")
    print(f"Đường dẫn tuyệt đối: {img_abs_path}")
    print(f"Số lượng vật thể phát hiện: {len(detections)}")
    for i, d in enumerate(detections, 1):
        print(f"  {i}. {d['class_name']} (conf: {d['confidence']}): bbox = {d['bbox']}")

    if not output_path:
        output_dir = json_path.parent / "annotated"
        output_path = output_dir / target_image_name

    draw_bounding_boxes(img_abs_path, detections, str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Vẽ bounding box lên ảnh từ file kết quả YOLO JSON.")
    parser.add_argument("-j", "--json", required=True, type=str, help="Đường dẫn file JSON kết quả YOLO")
    parser.add_argument("-i", "--image-name", required=True, type=str, help="Tên file ảnh (vd: L21_V001_transnetv2_0001_00.jpg)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Đường dẫn file ảnh đầu ra (tùy chọn)")

    args = parser.parse_args()
    visualize_from_json(args.json, args.image_name, args.output)


if __name__ == "__main__":
    main()
