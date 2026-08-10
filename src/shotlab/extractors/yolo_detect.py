import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm
from ultralytics import YOLO


# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Các định dạng ảnh được hỗ trợ
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class YOLOv11Detector:
    """
    Lớp xử lý nhận diện vật thể trong thư mục ảnh sử dụng mô hình YOLOv11
    và xuất kết quả chi tiết ra file JSON.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """
        Khởi tạo mô hình YOLOv11.

        Args:
            model_path (str): Đường dẫn hoặc tên file weights YOLOv11 (vd: 'yolo11n.pt', 'yolo11x.pt').
            conf_threshold (float): Ngưỡng độ tin cậy (Confidence threshold).
            iou_threshold (float): Ngưỡng NMS IoU threshold.
            device (Optional[str]): Thiết bị chạy mô hình ('cpu', 'cuda', '0', v.v.). Mặc định tự động chọn.
        """
        if YOLO is None:
            raise ImportError(
                "Thư viện 'ultralytics' chưa được cài đặt. "
                "Vui lòng cài đặt bằng lệnh: pip install ultralytics"
            )

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        logger.info(f"Đang tải mô hình YOLOv11 từ: {model_path}")
        self.model = YOLO(model_path)

    def find_images(self, input_dir: Path, recursive: bool = False) -> List[Path]:
        """
        Tìm tất cả các file ảnh trong thư mục đầu vào.

        Args:
            input_dir (Path): Thư mục chứa ảnh.
            recursive (bool): Nếu True, tìm kiếm đệ quy trong tất cả các thư mục con.

        Returns:
            List[Path]: Danh sách các đường dẫn ảnh tìm thấy.
        """
        pattern = "**/*" if recursive else "*"
        image_files = [
            file_path
            for file_path in input_dir.glob(pattern)
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        image_files.sort()
        return image_files

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_json_path: Union[str, Path],
        batch_size: int = 16,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        """
        Xử lý toàn bộ các ảnh trong thư mục đầu vào và lưu kết quả ra file JSON.

        Args:
            input_dir (Union[str, Path]): Đường dẫn thư mục ảnh đầu vào.
            output_json_path (Union[str, Path]): Đường dẫn file JSON đầu ra để lưu kết quả.
            batch_size (int): Kích thước batch khi suy luận (Inference batch size).
            recursive (bool): Tìm kiếm ảnh đệ quy trong thư mục con hay không.

        Returns:
            Dict[str, Any]: Dữ liệu kết quả nhận diện đã lưu vào file JSON.
        """
        input_dir = Path(input_dir)
        output_json_path = Path(output_json_path)

        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"Thư mục đầu vào không tồn tại hoặc không phải thư mục: {input_dir}")

        image_paths = self.find_images(input_dir, recursive=recursive)
        logger.info(f"Tìm thấy {len(image_paths)} ảnh trong thư mục: {input_dir}")

        if not image_paths:
            logger.warning("Không tìm thấy file ảnh hợp lệ nào để xử lý.")

        output_data = {
            "model_path": self.model_path,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "total_images": len(image_paths),
            "results": [],
        }

        if image_paths:
            # Chia danh sách ảnh thành các batch nhỏ
            batches = [image_paths[i : i + batch_size] for i in range(0, len(image_paths), batch_size)]
            iterator = tqdm(batches, desc="Phát hiện vật thể") if HAS_TQDM else batches

            for batch in iterator:
                str_batch = [str(p) for p in batch]
                results = self.model.predict(
                    source=str_batch,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False,
                )

                for img_path, res in zip(batch, results):
                    detections = []
                    boxes = res.boxes

                    if boxes is not None and len(boxes) > 0:
                        names = res.names
                        for box in boxes:
                            xyxy = box.xyxy[0].tolist()
                            conf = float(box.conf[0].item())
                            cls_id = int(box.cls[0].item())
                            cls_name = names.get(cls_id, str(cls_id))

                            detections.append({
                                "class_id": cls_id,
                                "class_name": cls_name,
                                "confidence": round(conf, 4),
                                "bbox": [round(coord, 2) for coord in xyxy],  # [xmin, ymin, xmax, ymax]
                            })

                    try:
                        rel_path = str(img_path.relative_to(input_dir))
                    except ValueError:
                        rel_path = img_path.name

                    output_data["results"].append({
                        "image_name": img_path.name,
                        "relative_path": rel_path,
                        "absolute_path": str(img_path.resolve()),
                        "detections_count": len(detections),
                        "detections": detections,
                    })

        # Tạo thư mục chứa file JSON nếu chưa tồn tại
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        # Ghi kết quả ra file JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Đã lưu thành công kết quả phát hiện vào: {output_json_path}")
        return output_data


def detect_directory(
    input_dir: str,
    output_json_path: str,
    model_path: str = "yolo11n.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    batch_size: int = 16,
    device: Optional[str] = None,
    recursive: bool = False,
) -> Dict[str, Any]:
    """
    Hàm tiện ích phát hiện vật thể trong thư mục ảnh bằng mô hình YOLOv11.

    Args:
        input_dir (str): Đường dẫn thư mục ảnh đầu vào.
        output_json_path (str): Đường dẫn file JSON đầu ra.
        model_path (str): File weights của YOLOv11 (mặc định: 'yolo11n.pt').
        conf_threshold (float): Ngưỡng tin cậy (mặc định: 0.25).
        iou_threshold (float): Ngưỡng IoU cho NMS (mặc định: 0.45).
        batch_size (int): Kích thước batch xử lý ảnh (mặc định: 16).
        device (Optional[str]): Thiết bị tính toán ('cpu', 'cuda', v.v.).
        recursive (bool): Tìm kiếm đệ quy ảnh trong thư mục con.

    Returns:
        Dict[str, Any]: Kết quả chi tiết thu được từ quá trình nhận diện.
    """
    detector = YOLOv11Detector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )
    return detector.process_directory(
        input_dir=input_dir,
        output_json_path=output_json_path,
        batch_size=batch_size,
        recursive=recursive,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Nhận diện vật thể trong thư mục ảnh bằng YOLOv11 và xuất kết quả ra file JSON."
    )
    parser.add_argument(
        "-i", "--input-dir", required=True, type=str, help="Đường dẫn thư mục chứa ảnh đầu vào"
    )
    parser.add_argument(
        "-o", "--output-json", required=True, type=str, help="Đường dẫn file JSON đầu ra"
    )
    parser.add_argument(
        "-m", "--model", default="yolo11n.pt", type=str, help="Mô hình YOLOv11 (mặc định: yolo11n.pt)"
    )
    parser.add_argument(
        "-c", "--conf", default=0.25, type=float, help="Ngưỡng độ tin cậy confidence threshold (mặc định: 0.25)"
    )
    parser.add_argument(
        "--iou", default=0.45, type=float, help="Ngưỡng NMS IoU threshold (mặc định: 0.45)"
    )
    parser.add_argument(
        "-b", "--batch-size", default=16, type=int, help="Kích thước batch (mặc định: 16)"
    )
    parser.add_argument(
        "-d", "--device", default=None, type=str, help="Thiết bị chạy ('cpu', 'cuda', '0', v.v.)"
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Tìm ảnh đệ quy trong thư mục con"
    )

    args = parser.parse_args()

    detect_directory(
        input_dir=args.input_dir,
        output_json_path=args.output_json,
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        batch_size=args.batch_size,
        device=args.device,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
