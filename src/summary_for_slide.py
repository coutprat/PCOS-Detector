from pathlib import Path, json
OUT = Path(r"D:\PCOS\pcos-detection\data\processed\results_image_only")
m = json.load(open(OUT/"metrics.json"))
print("Best threshold (from training):", m.get("threshold"))
print("TEST metrics (uncalibrated):", m.get("test", {}))
if (OUT/"calibration.json").exists():
    c = json.load(open(OUT/"calibration.json"))
    print("\nCalibration T*:", c["temperature"])
    print("Raw  (TEST):", c["raw"])
    print("Calib(TEST):", c["calibrated"])
