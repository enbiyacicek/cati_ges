"""YOLO11-seg çatı modelini eğitir ve uygulamanın kullanacağı ONNX'e dönüştürür.

GPU'lu bir makinede (ör. Google Colab, Kaggle) çalıştırın:
    pip install -r requirements-train.txt
    python training/train.py --data dataset/data.yaml --model yolo11s-seg.pt --epochs 100

Çıktı: models/roof_yolo11.onnx  → uygulama otomatik olarak YOLO moduna geçer.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="yolo11s-seg.pt", help="yolo11n/s/m-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0, cpu ...")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "models" / "roof_yolo11.onnx"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project="runs", name="roof", exist_ok=True, patience=25,
        # Uydu görüntüleri için uygun artırmalar: her yöne döndürme, dikey çevirme
        degrees=90, flipud=0.5, fliplr=0.5, mosaic=1.0, scale=0.3,
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,
    )
    best = Path("runs/roof/weights/best.pt")
    metrics = YOLO(best).val(data=args.data, imgsz=args.imgsz)
    print("Maske mAP50-95:", metrics.seg.map, " mAP50:", metrics.seg.map50)

    onnx_path = YOLO(best).export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, args.out)
    print("ONNX model kaydedildi:", args.out)


if __name__ == "__main__":
    main()
