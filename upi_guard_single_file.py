"""
UPI Guard Pro v3 — Multi-User Fake UPI Payment Screenshot Detector (< 3s)
=========================================================================
Features:
- Per-Phone / Per-Shop User Accounts (isolated history, KPIs, UTR ledger & references)
- Tuned Forensics (supports Light & Dark mode GPay, PhonePe, Paytm without false-flagging real receipts)
- Strict Live Mode toggle (catches old/stale screenshots at the counter; turn off to test with older genuine receipts)
- 1-Tap "Clear My History" inside the app (no need to delete .db files manually)
- Custom 512x512 Shield App Icon + PWA Manifest
"""

import base64
import io
import json
import re
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageDraw

# ============================================================================
# 1. REFERENCE LAYOUT PROFILES (Google Pay, PhonePe, Paytm — Light & Dark)
# ============================================================================

UPI_PROFILES: Dict[str, Dict[str, Any]] = {
    "phonepe": {
        "display_name": "PhonePe",
        "keywords": [
            "transaction successful",
            "phonepe",
            "powered by",
            "debited from",
            "transfer details",
            "view details",
            "check balance",
            "split expense",
            "paid to",
        ],
        "header_zone": (0.0, 0.34),
        "accent_hsv_ranges": [((45, 65, 55), (92, 255, 255))],
        "min_accent_ratio": 0.10,
        "required_phrases": ["successful", "paid to"],
        "utr_regex": r"\b(?:utr|upi\s*ref)[^0-9]{0,12}(\d{4}\s?\d{4}\s?\d{4})\b",
        "tx_id_regex": r"\b(T\d{18,24})\b",
    },
    "gpay": {
        "display_name": "Google Pay (GPay)",
        "keywords": [
            "google pay",
            "gpay",
            "upi transaction id",
            "completed",
            "paid to",
            "pay again",
            "having issues",
            "split with friends",
            "google transaction id",
        ],
        "header_zone": (0.05, 0.45),
        "accent_hsv_ranges": [
            ((95, 60, 65), (125, 255, 255)),
            ((48, 55, 55), (90, 255, 255)),
        ],
        "min_accent_ratio": 0.012,
        "required_phrases": ["completed", "upi transaction id"],
        "utr_regex": r"\b(?:upi\s*transaction\s*id|utr)[^0-9]{0,15}(\d{4}\s?\d{4}\s?\d{4})\b",
        "tx_id_regex": r"\b(CICAg[A-Za-z0-9_-]{8,})\b",
    },
    "paytm": {
        "display_name": "Paytm",
        "keywords": [
            "paytm",
            "paid successfully",
            "payment successful",
            "upi ref",
            "order id",
            "pay again",
            "share receipt",
            "from:",
            "to:",
        ],
        "header_zone": (0.0, 0.42),
        "accent_hsv_ranges": [
            ((85, 45, 65), (128, 255, 255)),
            ((48, 55, 55), (90, 255, 255)),
        ],
        "min_accent_ratio": 0.025,
        "required_phrases": ["paid", "upi ref"],
        "utr_regex": r"\b(?:upi\s*ref(?:\.?\s*no\.?)?|utr)[^0-9]{0,15}(\d{4}\s?\d{4}\s?\d{4})\b",
        "tx_id_regex": r"\b(\d{12})\b",
    },
}


def compute_layout_signature(bgr_img: np.ndarray) -> List[float]:
    resized = cv2.resize(bgr_img, (240, 480), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    rows, cols = 4, 3
    cell_h, cell_w = 480 // rows, 240 // cols
    features: List[float] = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            ph, pg, pe = hsv[y0:y1, x0:x1], gray[y0:y1, x0:x1], edges[y0:y1, x0:x1]
            features.extend(
                [
                    float(np.mean(ph[:, :, 0])) / 180.0,
                    float(np.mean(ph[:, :, 1])) / 255.0,
                    float(np.mean(ph[:, :, 2])) / 255.0,
                    float(np.std(pg)) / 128.0,
                    float(np.mean(pe > 0)),
                    float(np.mean(pg < 60)),
                ]
            )
    return [round(v, 5) for v in features]


def compare_layout_signatures(sig_a: List[float], sig_b: List[float]) -> float:
    a, b = np.array(sig_a, dtype=np.float32), np.array(sig_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom < 1e-8 else float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def detect_upi_app(ocr_text: str, bgr_img: np.ndarray) -> Dict[str, Any]:
    text_lower = ocr_text.lower()
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h, _ = bgr_img.shape[:2]

    scores: Dict[str, float] = {}
    matched_kw: Dict[str, List[str]] = {}
    for app_key, prof in UPI_PROFILES.items():
        hits = [kw for kw in prof["keywords"] if kw in text_lower]
        matched_kw[app_key] = hits
        score = len(hits) * 2.5

        y0 = int(h * prof["header_zone"][0])
        y1 = max(y0 + 1, int(h * prof["header_zone"][1]))
        zone = hsv[y0:y1, :]
        mask = np.zeros(zone.shape[:2], dtype=np.uint8)
        for lo, hi in prof["accent_hsv_ranges"]:
            mask = cv2.bitwise_or(mask, cv2.inRange(zone, np.array(lo), np.array(hi)))
        accent_ratio = float(np.mean(mask > 0))
        if accent_ratio >= prof["min_accent_ratio"]:
            score += 1.5
        if app_key == "phonepe" and accent_ratio > 0.20:
            score += 1.5
        scores[app_key] = score

    best = max(scores, key=lambda k: scores[k])
    if scores[best] < 1.5:
        return {"app_key": "unknown", "display_name": "Unknown UPI App", "confidence": 0.0, "matched_keywords": []}
    return {
        "app_key": best,
        "display_name": UPI_PROFILES[best]["display_name"],
        "confidence": round(min(1.0, scores[best] / 8.0), 2),
        "matched_keywords": matched_kw[best],
    }


# ============================================================================
# 2. MULTI-USER SQLITE DATABASE (Isolated per Shop / User ID)
# ============================================================================

DB_PATH = Path(__file__).parent / "upi_guard_v3.db"


def _clean_user_id(user_id: Optional[str]) -> str:
    raw = (user_id or "default").strip().lower()
    clean = re.sub(r"[^a-z0-9_-]", "_", raw)[:40]
    return clean or "default"


def _conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    return c


def init_db(db_path: Path = DB_PATH) -> None:
    with _conn(db_path) as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                app_key TEXT NOT NULL,
                verdict TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                utr TEXT,
                tx_id TEXT,
                amount TEXT,
                dhash TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                latency_ms INTEGER NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS reference_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                app_key TEXT NOT NULL,
                label TEXT NOT NULL,
                signature_json TEXT NOT NULL
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_scans_user_utr ON scans(user_id, utr)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_scans_user_tx ON scans(user_id, tx_id)")
        c.commit()


def hamming_distance(hex_a: str, hex_b: str) -> int:
    if not hex_a or not hex_b or len(hex_a) != len(hex_b):
        return 64
    return bin(int(hex_a, 16) ^ int(hex_b, 16)).count("1")


def find_reused_transaction(user_id: str, utr: Optional[str], tx_id: Optional[str], dhash: str, db_path: Path = DB_PATH) -> Dict[str, Any]:
    uid = _clean_user_id(user_id)
    with _conn(db_path) as c:
        if utr:
            row = c.execute("SELECT id, created_at FROM scans WHERE user_id = ? AND utr = ? ORDER BY id ASC LIMIT 1", (uid, utr)).fetchone()
            if row:
                return {"reused": True, "detail": f"UTR {utr} was already scanned at {row['created_at']} (Scan #{row['id']})."}
        if tx_id:
            row = c.execute("SELECT id, created_at FROM scans WHERE user_id = ? AND tx_id = ? ORDER BY id ASC LIMIT 1", (uid, tx_id)).fetchone()
            if row:
                return {"reused": True, "detail": f"Transaction ID {tx_id} was already scanned at {row['created_at']} (Scan #{row['id']})."}
        for row in c.execute("SELECT id, created_at, dhash FROM scans WHERE user_id = ? ORDER BY id DESC LIMIT 200", (uid,)).fetchall():
            dist = hamming_distance(dhash, row["dhash"])
            if dist <= 5:
                return {"reused": True, "detail": f"Near-identical screenshot already scanned at {row['created_at']} (visual distance {dist}/64)."}
    return {"reused": False, "detail": None}


def record_scan(user_id: str, app_key: str, verdict: str, risk_score: int, utr: Optional[str], tx_id: Optional[str],
                amount: Optional[str], dhash: str, reasons: List[Dict[str, Any]], latency_ms: int, db_path: Path = DB_PATH) -> int:
    uid = _clean_user_id(user_id)
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _conn(db_path) as c:
        cur = c.execute(
            "INSERT INTO scans (user_id, created_at, app_key, verdict, risk_score, utr, tx_id, amount, dhash, reasons_json, latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (uid, now_iso, app_key, verdict, risk_score, utr, tx_id, amount, dhash, json.dumps(reasons), latency_ms),
        )
        c.commit()
        return int(cur.lastrowid)


def get_recent_scans(user_id: str, limit: int = 25, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    uid = _clean_user_id(user_id)
    with _conn(db_path) as c:
        rows = c.execute("SELECT * FROM scans WHERE user_id = ? ORDER BY id DESC LIMIT ?", (uid, limit)).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        item["reasons"] = json.loads(item.pop("reasons_json"))
        out.append(item)
    return out


def clear_user_scans(user_id: str, db_path: Path = DB_PATH) -> int:
    uid = _clean_user_id(user_id)
    with _conn(db_path) as c:
        cur = c.execute("DELETE FROM scans WHERE user_id = ?", (uid,))
        c.commit()
        return cur.rowcount


def get_stats(user_id: str, db_path: Path = DB_PATH) -> Dict[str, Any]:
    uid = _clean_user_id(user_id)
    with _conn(db_path) as c:
        total = c.execute("SELECT COUNT(*) FROM scans WHERE user_id = ?", (uid,)).fetchone()[0]
        flagged = c.execute("SELECT COUNT(*) FROM scans WHERE user_id = ? AND verdict != 'REAL'", (uid,)).fetchone()[0]
        avg_ms = c.execute("SELECT AVG(latency_ms) FROM scans WHERE user_id = ?", (uid,)).fetchone()[0]
        refs = c.execute("SELECT COUNT(*) FROM reference_templates WHERE user_id = ?", (uid,)).fetchone()[0]
    return {
        "total_scans": int(total or 0),
        "flagged_count": int(flagged or 0),
        "avg_latency_ms": int(avg_ms or 0),
        "total_references": int(refs or 0),
    }


def save_reference_template(user_id: str, app_key: str, label: str, signature: List[float], db_path: Path = DB_PATH) -> int:
    uid = _clean_user_id(user_id)
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _conn(db_path) as c:
        cur = c.execute(
            "INSERT INTO reference_templates (user_id, created_at, app_key, label, signature_json) VALUES (?,?,?,?,?)",
            (uid, now_iso, app_key, label, json.dumps(signature)),
        )
        c.commit()
        return int(cur.lastrowid)


def get_reference_templates(user_id: str, app_key: Optional[str] = None, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    uid = _clean_user_id(user_id)
    with _conn(db_path) as c:
        rows = (
            c.execute("SELECT * FROM reference_templates WHERE user_id = ? AND app_key = ?", (uid, app_key)).fetchall()
            if app_key
            else c.execute("SELECT * FROM reference_templates WHERE user_id = ?", (uid,)).fetchall()
        )
    out = []
    for r in rows:
        item = dict(r)
        item["signature"] = json.loads(item.pop("signature_json"))
        out.append(item)
    return out


# ============================================================================
# 3. TUNED FORENSICS ENGINE (Low False Positives on Real Screenshots)
# ============================================================================

_OCR_ENGINE = None
_OCR_TRIED = False
_OCR_ERROR = None


def _get_ocr():
    global _OCR_ENGINE, _OCR_TRIED, _OCR_ERROR
    if not _OCR_TRIED:
        _OCR_TRIED = True
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
        except Exception as e:
            _OCR_ENGINE = None
            _OCR_ERROR = str(e)
    return _OCR_ENGINE


def compute_dhash(bgr_img: np.ndarray, hash_size: int = 8) -> str:
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = "".join("1" if b else "0" for b in (resized[:, 1:] > resized[:, :-1]).flatten())
    return f"{int(bits, 2):016x}"


def run_ocr(bgr_img: np.ndarray) -> Tuple[str, List[Dict[str, Any]]]:
    ocr = _get_ocr()
    if ocr is None:
        return "", []
    h, w = bgr_img.shape[:2]
    scale = 1.0
    ocr_img = bgr_img
    if h > 760:
        scale = 760.0 / h
        ocr_img = cv2.resize(bgr_img, (int(w * scale), 760), interpolation=cv2.INTER_AREA)

    res, _ = ocr(ocr_img)
    if not res:
        return "", []
    lines, boxes = [], []
    inv = 1.0 / scale
    for pts, text, conf in res:
        xs, ys = [p[0] * inv for p in pts], [p[1] * inv for p in pts]
        lines.append(str(text))
        boxes.append({"text": str(text), "conf": float(conf), "x0": int(min(xs)), "y0": int(min(ys)), "x1": int(max(xs)), "y1": int(max(ys))})
    return "\n".join(lines), boxes


def extract_payment_fields(ocr_text: str, app_key: str) -> Dict[str, Optional[str]]:
    text = ocr_text or ""
    amount = None
    m_amt = re.search(r"(?:₹|rs\.?|inr)\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)", text, re.IGNORECASE)
    if m_amt:
        amount = m_amt.group(1).replace(",", "")
    else:
        m_fb = re.search(r"(?m)^\s*[₹rRfF?]?\s*([0-9]{1,2},[0-9]{2},[0-9]{3}|[0-9]{1,3},[0-9]{3}(?:\.[0-9]{1,2})?)\s*$", text)
        if m_fb:
            amount = m_fb.group(1).replace(",", "")

    utr, tx_id = None, None
    prof = UPI_PROFILES.get(app_key)
    if prof:
        m = re.search(prof["utr_regex"], text, re.IGNORECASE)
        if m:
            utr = re.sub(r"\s+", "", m.group(1))
        mt = re.search(prof["tx_id_regex"], text)
        if mt:
            tx_id = mt.group(1)
    if not utr:
        m = re.search(r"(?<!\d)(\d{4}\s?\d{4}\s?\d{4})(?!\d)", text)
        if m:
            utr = re.sub(r"\s+", "", m.group(1))
    return {"amount": amount, "utr": utr, "tx_id": tx_id}


def check_metadata_and_editor_tags(raw_bytes: bytes) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []
    editors = ["photoshop", "picsart", "canva", "snapseed", "gimp", "pixellab", "lightroom", "figma", "inshot", "fake", "prank", "spoof"]
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        blob = " ".join(f"{k}:{v}" for k, v in (pil_img.info or {}).items())
        exif = pil_img.getexif()
        if exif:
            blob += " " + " ".join(f"{ExifTags.TAGS.get(k, k)}:{v}" for k, v in exif.items())
        low = blob.lower()
        for ed in editors:
            if ed in low:
                flags.append({"category": "Metadata / Editor", "pillar": "pixel", "severity": "critical", "points": 55, "message": f"Image metadata contains editing software signature: '{ed}'."})
                break
    except Exception:
        pass
    return flags


def check_layout(user_id: str, bgr_img: np.ndarray, app_info: Dict[str, Any], ocr_text: str, db_path: Path = DB_PATH) -> Tuple[List[Dict[str, Any]], float]:
    flags: List[Dict[str, Any]] = []
    h, w = bgr_img.shape[:2]
    aspect = h / max(1, w)
    app_key = app_info["app_key"]

    # Phone screenshots (including tall 21:9 phones or slightly cropped status bars)
    if aspect < 1.35 or aspect > 2.75:
        flags.append({"category": "Layout Mismatch", "pillar": "layout", "severity": "medium", "points": 20, "message": f"Unusual aspect ratio ({aspect:.2f}). Genuine phone screenshots are typically 1.6–2.4."})

    if app_key == "unknown":
        flags.append({"category": "Layout Mismatch", "pillar": "layout", "severity": "high", "points": 35, "message": "Does not match Google Pay, PhonePe, or Paytm payment screen layout."})
        return flags, 0.0

    prof = UPI_PROFILES[app_key]
    # Only enforce header brand color if OCR didn't strongly confirm the app (prevents Dark Mode false positives)
    if len(app_info.get("matched_keywords", [])) < 2:
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        y0, y1 = int(h * prof["header_zone"][0]), max(1, int(h * prof["header_zone"][1]))
        zone = hsv[y0:y1, :]
        mask = np.zeros(zone.shape[:2], dtype=np.uint8)
        for lo, hi in prof["accent_hsv_ranges"]:
            mask = cv2.bitwise_or(mask, cv2.inRange(zone, np.array(lo), np.array(hi)))
        accent_ratio = float(np.mean(mask > 0))
        if accent_ratio < prof["min_accent_ratio"] * 0.4:
            flags.append({"category": "Layout Mismatch", "pillar": "layout", "severity": "medium", "points": 20, "message": f"{prof['display_name']} expected brand header missing ({accent_ratio*100:.1f}%)."})

    # Compare against THIS user's calibrated genuine references
    sig = compute_layout_signature(bgr_img)
    refs = get_reference_templates(user_id=user_id, app_key=app_key, db_path=db_path)
    best_sim = -1.0
    if refs:
        best_sim = max(compare_layout_signatures(sig, r["signature"]) for r in refs)
        if best_sim < 0.78:
            flags.append({"category": "Reference Mismatch", "pillar": "layout", "severity": "high", "points": 30, "message": f"Layout deviates from your calibrated {prof['display_name']} references (similarity {best_sim*100:.1f}%)."})
    return flags, round(best_sim, 3) if best_sim >= 0 else -1.0


def check_pixel_and_font_forensics(bgr_img: np.ndarray, ocr_boxes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    flags: List[Dict[str, Any]] = []
    h, w = bgr_img.shape[:2]

    # A. Error Level Analysis (ELA) — tuned so crisp native UI text does not false-trigger
    ok, enc = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    ela_b64 = ""
    if ok:
        recomp = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        ela_gray = np.mean(cv2.absdiff(bgr_img, recomp).astype(np.float32), axis=2)
        ela_heat = cv2.applyColorMap(np.clip(ela_gray * 14.0, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        ela_small = cv2.resize(ela_heat, (min(360, w), int(min(360, w) * h / max(1, w))))
        _, heat_buf = cv2.imencode(".jpg", ela_small, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        ela_b64 = "data:image/jpeg;base64," + base64.b64encode(heat_buf.tobytes()).decode("ascii")

        bh, bw = max(1, h // 8), max(1, w // 4)
        means = [float(np.mean(ela_gray[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw])) for r in range(8) for c in range(4)]
        med_ela, max_ela = float(np.median(means)), float(np.max(means))
        # Genuine screenshots have low ELA everywhere; spliced JPEG edits spike > 18 AND > 6x median
        if max_ela > max(18.0, med_ela * 6.0):
            flags.append({"category": "Pixel Tampering (ELA)", "pillar": "pixel", "severity": "high", "points": 32, "message": f"Strong localized compression splice detected (peak ELA {max_ela:.1f} vs baseline {med_ela:.1f})."})

    # B. Rectangular Cover-Up Patch Detection (excludes full-width UI cards to prevent false positives)
    y0, y1 = int(h * 0.14), int(h * 0.52)
    roi = bgr_img[y0:y1, :]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3))
    subtle = (((gx >= 10) & (gx <= 140)) | ((gy >= 10) & (gy <= 140))).astype(np.uint8) * 255
    subtle = cv2.morphologyEx(subtle, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(subtle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Only small-to-medium boxes (amount/timestamp size), NOT full-width app cards
        if 0.14 * w <= cw <= 0.56 * w and 0.03 * h <= ch <= 0.11 * h:
            if cv2.contourArea(cnt) / max(1.0, float(cw * ch)) >= 0.68:
                pad = 5
                if y - pad >= 0 and y + ch + pad < roi.shape[0] and x - pad >= 0 and x + cw + pad < roi.shape[1]:
                    inside_top = roi[y + 2 : y + 5, x + 6 : x + cw - 6]
                    outside_top = roi[y - pad : y - 2, x + 6 : x + cw - 6]
                    outside_bot = roi[y + ch + 2 : y + ch + pad, x + 6 : x + cw - 6]
                    if inside_top.size > 0 and outside_top.size > 0 and outside_bot.size > 0:
                        diff_top = float(np.linalg.norm(np.mean(inside_top, axis=(0, 1)) - np.mean(outside_top, axis=(0, 1))))
                        bg_match = float(np.linalg.norm(np.mean(outside_top, axis=(0, 1)) - np.mean(outside_bot, axis=(0, 1))))
                        # A real cover-up patch sits inside a uniform surrounding background (bg_match < 6)
                        if 4.0 <= diff_top <= 36.0 and bg_match < 12.0:
                            flags.append({"category": "Pixel Inconsistency", "pillar": "pixel", "severity": "high", "points": 32, "message": "Detected a rectangular cover-up patch with mismatched background shade over the payment details."})
                            break

    # C. Font Stroke-Width Consistency Across OCR Boxes
    if len(ocr_boxes) >= 4:
        strokes, amt_stroke = [], None
        for b in ocr_boxes:
            bw_b, bh_b = b["x1"] - b["x0"], b["y1"] - b["y0"]
            if bw_b < 18 or bh_b < 12:
                continue
            crop = bgr_img[max(0, b["y0"]) : min(h, b["y1"]), max(0, b["x0"]) : min(w, b["x1"])]
            if crop.size == 0:
                continue
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(th > 0) > 0.5:
                th = 255 - th
            fg = cv2.distanceTransform(th, cv2.DIST_L2, 3)
            fg = fg[fg > 0]
            if fg.size > 15:
                norm_sw = float(np.median(fg)) / max(1.0, float(bh_b))
                strokes.append(norm_sw)
                if re.search(r"(?:₹|rs\.?|\d{2,})", b["text"], re.IGNORECASE) and bh_b > h * 0.03:
                    amt_stroke = norm_sw
        if len(strokes) >= 4 and amt_stroke is not None:
            med_sw = float(np.median(strokes))
            if med_sw > 0 and (amt_stroke / med_sw > 2.8 or amt_stroke / med_sw < 0.28):
                flags.append({"category": "Font Inconsistency", "pillar": "pixel", "severity": "medium", "points": 22, "message": f"Amount text stroke weight ({amt_stroke:.3f}) differs sharply from surrounding app UI font ({med_sw:.3f})."})

    return flags, ela_b64


def check_timestamp(ocr_text: str, now: Optional[datetime] = None, strict_live_mode: bool = False, max_age_minutes: int = 15) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    flags: List[Dict[str, Any]] = []
    if not ocr_text:
        return flags, None
    current = now or datetime.now()

    bad_ampm = re.search(r"\b(1[3-9]|2[0-9]|00):([0-5]\d)\s*(?:am|pm)\b", ocr_text, re.IGNORECASE)
    if bad_ampm:
        flags.append({"category": "Impossible Timestamp", "pillar": "timestamp", "severity": "critical", "points": 55, "message": f"Impossible 12-hour clock time detected: '{bad_ampm.group(0)}'."})
        return flags, bad_ampm.group(0)

    bad_min = re.search(r"\b([0-2]?\d):([6-9]\d)\b", ocr_text)
    if bad_min:
        flags.append({"category": "Impossible Timestamp", "pillar": "timestamp", "severity": "critical", "points": 55, "message": f"Impossible minute value in timestamp: '{bad_min.group(0)}'."})
        return flags, bad_min.group(0)

    months = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
    p1 = rf"(\d{{1,2}})\s+({months})[a-z]*\s*,?\s*(\d{{4}})\s*(?:,|at|on)?\s*(\d{{1,2}}:\d{{2}})\s*(am|pm)?"
    p2 = rf"(\d{{1,2}}:\d{{2}})\s*(am|pm)?\s*(?:on|,)?\s*(\d{{1,2}})\s+({months})[a-z]*\s*,?\s*(\d{{4}})"
    p3 = rf"({months})[a-z]*\s+(\d{{1,2}})\s*,?\s*(\d{{4}})\s*(?:,|at)?\s*(\d{{1,2}}:\d{{2}})\s*(am|pm)?"

    parsed_dt, raw_str = None, None
    m1 = re.search(p1, ocr_text, re.IGNORECASE)
    m2 = None if m1 else re.search(p2, ocr_text, re.IGNORECASE)
    m3 = None if (m1 or m2) else re.search(p3, ocr_text, re.IGNORECASE)

    if m1 or m2 or m3:
        if m1:
            day_s, mon_s, yr_s, hm_s, ampm_s = m1.groups()
            raw_str = m1.group(0)
        elif m2:
            hm_s, ampm_s, day_s, mon_s, yr_s = m2.groups()
            raw_str = m2.group(0)
        else:
            mon_s, day_s, yr_s, hm_s, ampm_s = m3.groups()
            raw_str = m3.group(0)
        try:
            fmt = "%d %b %Y %I:%M %p" if ampm_s else "%d %b %Y %H:%M"
            val = f"{day_s} {mon_s[:3]} {yr_s} {hm_s} {ampm_s.upper()}" if ampm_s else f"{day_s} {mon_s[:3]} {yr_s} {hm_s}"
            parsed_dt = datetime.strptime(val, fmt)
        except ValueError:
            flags.append({"category": "Impossible Timestamp", "pillar": "timestamp", "severity": "critical", "points": 55, "message": f"Calendar date or time does not exist: '{raw_str}'."})

    if parsed_dt is not None:
        diff_min = (current - parsed_dt).total_seconds() / 60.0
        if diff_min < -5.0:
            flags.append({"category": "Impossible Timestamp", "pillar": "timestamp", "severity": "critical", "points": 55, "message": f"Timestamp '{raw_str}' is {abs(diff_min):.0f} minutes in the FUTURE."})
        elif strict_live_mode and diff_min > max_age_minutes:
            age = f"{diff_min/60:.1f} hours" if diff_min >= 90 else f"{diff_min:.0f} minutes"
            flags.append({"category": "Stale Timestamp", "pillar": "timestamp", "severity": "high", "points": 35, "message": f"Payment timestamp '{raw_str}' is {age} old (Strict Live Mode limit: {max_age_minutes} min)."})

    return flags, raw_str


def analyze_screenshot(raw_bytes: bytes, user_id: str = "default", expected_amount: Optional[str] = None,
                       strict_live_mode: bool = False, override_ocr_text: Optional[str] = None,
                       now: Optional[datetime] = None, db_path: Path = DB_PATH, save_to_db: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    bgr = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image. Please upload a valid PNG or JPEG screenshot.")

    h, w = bgr.shape[:2]
    if h > 1100:
        bgr = cv2.resize(bgr, (int(w * (1100.0 / h)), 1100), interpolation=cv2.INTER_AREA)

    ocr_text, ocr_boxes = (override_ocr_text, []) if override_ocr_text is not None else run_ocr(bgr)
    app_info = detect_upi_app(ocr_text, bgr)
    fields = extract_payment_fields(ocr_text, app_info["app_key"])
    dhash = compute_dhash(bgr)

    reasons: List[Dict[str, Any]] = []
    reasons.extend(check_metadata_and_editor_tags(raw_bytes))
    layout_flags, ref_sim = check_layout(user_id, bgr, app_info, ocr_text, db_path=db_path)
    reasons.extend(layout_flags)
    pixel_flags, ela_b64 = check_pixel_and_font_forensics(bgr, ocr_boxes)
    reasons.extend(pixel_flags)
    ts_flags, parsed_ts = check_timestamp(ocr_text, now=now, strict_live_mode=strict_live_mode)
    reasons.extend(ts_flags)

    reuse = find_reused_transaction(user_id, fields["utr"], fields["tx_id"], dhash, db_path=db_path)
    if reuse["reused"]:
        reasons.append({"category": "Reused Transaction ID", "pillar": "utr", "severity": "critical", "points": 60, "message": reuse["detail"]})

    if expected_amount and expected_amount.strip():
        clean_exp = re.sub(r"[^0-9.]", "", expected_amount)
        if clean_exp and fields["amount"]:
            try:
                if abs(float(clean_exp) - float(fields["amount"])) > 0.01:
                    reasons.append({"category": "Amount Mismatch", "pillar": "utr", "severity": "critical", "points": 60, "message": f"Screenshot shows ₹{fields['amount']}, but cashier expected ₹{clean_exp}."})
            except ValueError:
                pass

    def _pillar_status(key: str) -> str:
        p_flags = [r for r in reasons if r.get("pillar") == key]
        if any(r["severity"] in ("critical", "high") for r in p_flags):
            return "FAIL"
        if p_flags:
            return "WARN"
        return "PASS"

    pillars = {
        "layout": _pillar_status("layout"),
        "pixel": _pillar_status("pixel"),
        "timestamp": _pillar_status("timestamp"),
        "utr": _pillar_status("utr"),
    }

    risk = min(100, sum(r["points"] for r in reasons))
    verdict = "LIKELY FAKE" if risk >= 55 else ("SUSPICIOUS" if risk >= 25 else "REAL")
    if not reasons:
        reasons.append({"category": "All Checks Passed", "pillar": "ok", "severity": "ok", "points": 0, "message": f"Layout matches {app_info['display_name']}, no pixel tampering found, timestamp is valid, and Transaction ID has not been reused."})

    latency_ms = int((time.perf_counter() - t0) * 1000)
    scan_id = record_scan(user_id, app_info["app_key"], verdict, risk, fields["utr"], fields["tx_id"], fields["amount"], dhash, reasons, latency_ms, db_path=db_path) if save_to_db else None
    return {
        "scan_id": scan_id,
        "verdict": verdict,
        "risk_score": risk,
        "latency_ms": latency_ms,
        "app": app_info,
        "extracted": {"amount": fields["amount"], "utr": fields["utr"], "tx_id": fields["tx_id"], "timestamp": parsed_ts},
        "reference_similarity": ref_sim,
        "pillars": pillars,
        "reasons": reasons,
        "ela_heatmap": ela_b64,
        "ocr_available": _OCR_ENGINE is not None or override_ocr_text is not None,
        "ocr_preview": (ocr_text[:450] + "…") if len(ocr_text) > 450 else ocr_text,
    }


# ============================================================================
# 4. MULTI-USER MOBILE-FIRST UI (Per-Shop Login + Strict Mode Toggle)
# ============================================================================

MOBILE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <meta name="theme-color" content="#070b14" />
  <link rel="icon" type="image/png" sizes="512x512" href="/icon.png" />
  <link rel="apple-touch-icon" sizes="512x512" href="/icon.png" />
  <link rel="manifest" href="/manifest.json" />
  <title>UPI Guard Pro — 3-Second Payment Forensics</title>
  <style>
    :root {
      --bg:#070b14; --card:rgba(20,29,48,0.88); --border:rgba(148,163,184,0.16);
      --text:#f8fafc; --muted:#94a3b8; --accent:#38bdf8;
      --green:#22c55e; --green-bg:rgba(22,163,74,0.14);
      --amber:#f59e0b; --amber-bg:rgba(245,158,11,0.14);
      --red:#ef4444; --red-bg:rgba(239,68,68,0.14);
    }
    * { box-sizing:border-box; margin:0; padding:0; font-family:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
    body {
      background:
        radial-gradient(circle at 12% 0%, rgba(56,189,248,0.14), transparent 36%),
        radial-gradient(circle at 88% 10%, rgba(99,102,241,0.16), transparent 40%),
        var(--bg);
      color:var(--text); min-height:100vh; padding-bottom:92px;
    }

    /* Login Overlay */
    .login-overlay {
      position:fixed; inset:0; z-index:100;
      background:rgba(7,11,20,0.96); backdrop-filter:blur(14px);
      display:none; align-items:center; justify-content:center; padding:20px;
    }
    .login-overlay.show { display:flex; }
    .login-box {
      width:100%; max-width:410px; background:#0f172a; border:1px solid var(--border);
      border-radius:22px; padding:26px 22px; box-shadow:0 20px 50px rgba(0,0,0,0.5);
    }
    .login-logo {
      width:54px; height:54px; border-radius:14px; margin:0 auto 12px;
      background:linear-gradient(135deg,#0ea5e9,#6366f1); display:grid; place-items:center; font-size:1.6rem;
    }
    .login-box h2 { text-align:center; font-size:1.25rem; font-weight:800; }
    .login-box p { text-align:center; font-size:0.84rem; color:var(--muted); margin:6px 0 18px; line-height:1.4; }

    /* Header */
    header {
      position:sticky; top:0; z-index:20; backdrop-filter:blur(14px);
      background:rgba(7,11,20,0.84); border-bottom:1px solid var(--border); padding:12px 16px;
    }
    .header-inner { max-width:980px; margin:0 auto; display:flex; justify-content:space-between; align-items:center; gap:10px; }
    .brand { display:flex; align-items:center; gap:10px; }
    .brand-icon {
      width:34px; height:34px; border-radius:10px;
      background:linear-gradient(135deg,#0ea5e9,#6366f1); display:grid; place-items:center; font-size:1rem;
    }
    .brand h1 { font-size:1.06rem; font-weight:800; }
    .brand small { display:block; font-size:0.7rem; color:var(--muted); }
    .user-pill {
      display:inline-flex; align-items:center; gap:6px; cursor:pointer;
      font-size:0.76rem; font-weight:700; color:#e2e8f0;
      background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3);
      padding:6px 11px; border-radius:999px;
    }

    main { max-width:980px; margin:0 auto; padding:16px; }
    .kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:16px; }
    @media (max-width:640px) { .kpi-grid { grid-template-columns:repeat(2,1fr); } }
    .kpi { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:12px 14px; }
    .kpi-label { font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em; }
    .kpi-val { font-size:1.3rem; font-weight:800; margin-top:3px; }

    .split { display:grid; grid-template-columns:1fr 1.15fr; gap:16px; align-items:start; }
    @media (max-width:820px) { .split { grid-template-columns:1fr; } }

    .card { background:var(--card); border:1px solid var(--border); border-radius:18px; padding:18px; box-shadow:0 12px 32px rgba(0,0,0,0.28); }
    .card-title { font-size:0.96rem; font-weight:700; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center; }
    .app-chips { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:12px; }
    .chip { font-size:0.72rem; font-weight:600; padding:4px 10px; border-radius:999px; background:rgba(148,163,184,0.12); color:#cbd5e1; border:1px solid var(--border); }

    .label { font-size:0.78rem; color:var(--muted); margin-bottom:6px; display:block; font-weight:600; }
    input[type="number"], input[type="text"], select {
      width:100%; padding:12px 14px; border-radius:12px;
      border:1px solid var(--border); background:rgba(7,11,20,0.75);
      color:var(--text); font-size:0.96rem; margin-bottom:12px; outline:none;
    }
    input:focus, select:focus { border-color:var(--accent); }

    .toggle-row {
      display:flex; align-items:center; justify-content:space-between; gap:10px;
      background:rgba(7,11,20,0.5); border:1px solid var(--border);
      padding:10px 12px; border-radius:12px; margin-bottom:14px; font-size:0.8rem;
    }
    .toggle-row input { width:18px; height:18px; accent-color:#38bdf8; }

    .dropzone {
      border:2px dashed rgba(56,189,248,0.35); background:rgba(56,189,248,0.04);
      border-radius:14px; padding:18px 14px; text-align:center; margin-bottom:14px; cursor:pointer;
    }
    .dropzone p { font-size:0.82rem; color:var(--muted); margin-top:5px; }
    .preview-thumb { max-height:170px; border-radius:10px; margin-top:10px; border:1px solid var(--border); display:none; }

    .btn-row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .btn {
      display:flex; align-items:center; justify-content:center; gap:8px;
      padding:13px 12px; border-radius:12px; border:none; font-size:0.93rem; font-weight:700; cursor:pointer;
    }
    .btn-primary { background:linear-gradient(135deg,#0ea5e9,#3b82f6); color:#fff; }
    .btn-secondary { background:rgba(51,65,85,0.85); color:#f8fafc; border:1px solid var(--border); }
    .btn-danger { background:rgba(239,68,68,0.16); color:#fca5a5; border:1px solid rgba(239,68,68,0.35); padding:7px 12px; font-size:0.78rem; border-radius:10px; }
    .btn-full { width:100%; }
    input[type="file"] { display:none; }

    .loader { display:none; text-align:center; padding:34px 16px; }
    .spinner { width:40px; height:40px; border:4px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.7s linear infinite; margin:0 auto 12px; }
    @keyframes spin { to { transform:rotate(360deg); } }

    .empty-state { text-align:center; padding:36px 16px; color:var(--muted); }
    .empty-state h3 { color:var(--text); font-size:1.02rem; margin:10px 0 6px; }
    .empty-state p { font-size:0.85rem; line-height:1.5; }

    .verdict-banner { border-radius:16px; padding:16px; margin-bottom:14px; display:flex; justify-content:space-between; align-items:center; gap:12px; border:1.5px solid transparent; }
    .v-REAL { background:var(--green-bg); border-color:var(--green); }
    .v-SUSPICIOUS { background:var(--amber-bg); border-color:var(--amber); }
    .v-FAKE { background:var(--red-bg); border-color:var(--red); }
    .v-title { font-size:1.45rem; font-weight:900; }
    .v-sub { font-size:0.82rem; color:#e2e8f0; margin-top:4px; }

    .gauge { position:relative; width:66px; height:66px; flex-shrink:0; }
    .gauge svg { transform:rotate(-90deg); width:66px; height:66px; }
    .gauge-num { position:absolute; inset:0; display:grid; place-items:center; font-size:0.92rem; font-weight:800; }

    .pillars { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin-bottom:14px; }
    .pillar { background:rgba(7,11,20,0.55); border:1px solid var(--border); border-radius:12px; padding:10px 12px; display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; font-weight:600; }
    .tag { font-size:0.68rem; font-weight:800; padding:3px 8px; border-radius:999px; }
    .tag-PASS { background:rgba(34,197,94,0.18); color:#4ade80; }
    .tag-WARN { background:rgba(245,158,11,0.2); color:#fbbf24; }
    .tag-FAIL { background:rgba(239,68,68,0.22); color:#f87171; }

    .meta-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin-bottom:14px; }
    .meta-item { background:rgba(7,11,20,0.55); border:1px solid var(--border); padding:10px 12px; border-radius:12px; }
    .meta-k { font-size:0.68rem; color:var(--muted); text-transform:uppercase; }
    .meta-v { font-size:0.9rem; font-weight:700; margin-top:3px; word-break:break-all; }

    .reason-list { list-style:none; display:flex; flex-direction:column; gap:8px; margin-bottom:14px; }
    .reason-item { background:rgba(7,11,20,0.6); border-left:4px solid var(--amber); padding:10px 12px; border-radius:10px; font-size:0.85rem; line-height:1.4; }
    .reason-critical, .reason-high { border-left-color:var(--red); }
    .reason-ok { border-left-color:var(--green); }
    .reason-cat { font-size:0.7rem; font-weight:800; text-transform:uppercase; color:var(--muted); display:block; margin-bottom:2px; }

    .compare-grid { display:none; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px; }
    .compare-grid img { width:100%; border-radius:10px; border:1px solid var(--border); }
    .compare-cap { font-size:0.72rem; color:var(--muted); text-align:center; margin-top:4px; }

    details { margin-top:10px; font-size:0.78rem; color:var(--muted); }
    details pre { margin-top:6px; padding:10px; border-radius:10px; background:rgba(7,11,20,0.8); border:1px solid var(--border); white-space:pre-wrap; color:#cbd5e1; max-height:130px; overflow:auto; }

    .tabs {
      position:fixed; bottom:0; left:0; right:0; z-index:20; backdrop-filter:blur(14px);
      background:rgba(7,11,20,0.92); border-top:1px solid var(--border);
      display:flex; justify-content:center; gap:10px; padding:10px 14px;
    }
    .tab-btn { flex:1; max-width:190px; background:none; border:1px solid transparent; color:var(--muted); font-size:0.85rem; font-weight:700; padding:9px 10px; cursor:pointer; border-radius:12px; }
    .tab-btn.active { color:var(--accent); background:rgba(56,189,248,0.12); border-color:rgba(56,189,248,0.3); }
    .view { display:none; } .view.active { display:block; }
    .history-row { display:flex; justify-content:space-between; align-items:center; padding:12px 0; border-bottom:1px solid var(--border); font-size:0.88rem; }
    .badge-REAL { color:#4ade80; font-weight:800; } .badge-SUSPICIOUS { color:#fbbf24; font-weight:800; } .badge-FAKE { color:#f87171; font-weight:800; }
  </style>
</head>
<body>
  <!-- Per-Phone / Per-Shop User Login Screen -->
  <div id="loginOverlay" class="login-overlay">
    <div class="login-box">
      <div class="login-logo">🛡️</div>
      <h2>Sign In to UPI Guard</h2>
      <p>Each shop or phone gets its own private scan history, UTR ledger, and calibrated references.</p>
      <label class="label" for="shopNameInput">Shop or Staff Name</label>
      <input type="text" id="shopNameInput" placeholder="e.g. Campus Canteen / Prince" />
      <label class="label" for="shopPinInput">4-Digit Shop Code (keeps your data separate)</label>
      <input type="number" id="shopPinInput" placeholder="e.g. 1234" inputmode="numeric" />
      <button class="btn btn-primary btn-full" onclick="saveUserLogin()">Continue to Dashboard →</button>
    </div>
  </div>

  <header>
    <div class="header-inner">
      <div class="brand">
        <div class="brand-icon">🛡️</div>
        <div>
          <h1>UPI Guard Pro</h1>
          <small>Campus Merchant Forensics</small>
        </div>
      </div>
      <button class="user-pill" onclick="openLogin()" title="Switch Shop / User">
        👤 <span id="activeUserLabel">Sign In</span> ▾
      </button>
    </div>
  </header>

  <main>
    <div class="kpi-grid">
      <div class="kpi"><div class="kpi-label">My Scans</div><div class="kpi-val" id="kpiTotal">0</div></div>
      <div class="kpi"><div class="kpi-label">Fakes Caught</div><div class="kpi-val" id="kpiFlagged" style="color:#f87171;">0</div></div>
      <div class="kpi"><div class="kpi-label">Avg Speed</div><div class="kpi-val" id="kpiSpeed" style="color:#38bdf8;">—</div></div>
      <div class="kpi"><div class="kpi-label">References</div><div class="kpi-val" id="kpiRefs" style="color:#4ade80;">0</div></div>
    </div>

    <!-- TAB 1: SCAN -->
    <section id="view-scan" class="view active">
      <div class="split">
        <div class="card">
          <div class="card-title"><span>Verify Payment Screenshot</span></div>
          <div class="app-chips"><span class="chip">Google Pay</span><span class="chip">PhonePe</span><span class="chip">Paytm</span></div>

          <label class="label" for="expectedAmount">Expected Amount ₹ (optional — catches edited amounts)</label>
          <input type="number" id="expectedAmount" placeholder="e.g. 120" inputmode="decimal" />

          <label class="toggle-row">
            <span><strong>Strict Live Counter Mode</strong><br><small style="color:var(--muted);">Flag screenshots older than 15 minutes</small></span>
            <input type="checkbox" id="strictLive" />
          </label>

          <div class="dropzone" id="dropzone" onclick="document.getElementById('galleryInput').click()">
            <div style="font-size:1.6rem;">📤</div>
            <div style="font-weight:700; margin-top:4px;">Drop screenshot here or tap to browse</div>
            <p>Supports PNG &amp; JPG • Auto-compressed for &lt;3s verdict</p>
            <img id="previewThumb" class="preview-thumb" alt="Preview" />
          </div>

          <div class="btn-row">
            <button class="btn btn-primary" onclick="document.getElementById('cameraInput').click()">📷 Snap Photo</button>
            <button class="btn btn-secondary" onclick="document.getElementById('galleryInput').click()">🖼️ Upload File</button>
          </div>
          <input id="cameraInput" type="file" accept="image/*" capture="environment" onchange="handleFile(event)" />
          <input id="galleryInput" type="file" accept="image/*" onchange="handleFile(event)" />
        </div>

        <div class="card">
          <div id="loader" class="loader">
            <div class="spinner"></div>
            <div style="font-weight:700;">Running 4-Layer Forensic Scan…</div>
            <div style="font-size:0.8rem; color:var(--muted); margin-top:4px;">Layout • ELA Pixels • Timestamp • UTR History</div>
          </div>
          <div id="resultArea">
            <div class="empty-state">
              <div style="font-size:2.3rem;">⚡</div>
              <h3>Ready for 3-Second Verification</h3>
              <p>Snap or upload a customer's UPI payment screenshot. Your scans, UTR ledger, and references stay private to your shop account.</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 2: HISTORY -->
    <section id="view-history" class="view">
      <div class="card">
        <div class="card-title">
          <span>My Recent Scans &amp; UTR Ledger</span>
          <button class="btn btn-danger" onclick="clearMyHistory()">🗑️ Clear My History</button>
        </div>
        <div id="historyList" style="color:var(--muted);">Loading...</div>
      </div>
    </section>

    <!-- TAB 3: CALIBRATE -->
    <section id="view-calibrate" class="view">
      <div class="card" style="max-width:560px; margin:0 auto;">
        <div class="card-title"><span>My Genuine Reference Dataset</span></div>
        <p style="font-size:0.85rem; color:var(--muted); margin-bottom:14px; line-height:1.45;">
          Upload 3–5 real payment screenshots per UPI app so UPI Guard learns your local layout baselines.
        </p>
        <div id="refCounts" style="font-size:0.86rem; margin-bottom:14px; color:var(--accent); font-weight:600;"></div>
        <label class="label" for="calApp">Select UPI App</label>
        <select id="calApp"><option value="gpay">Google Pay (GPay)</option><option value="phonepe">PhonePe</option><option value="paytm">Paytm</option></select>
        <label class="label" for="calLabel">Device / Theme Label</label>
        <input type="text" id="calLabel" placeholder="e.g. GPay Dark Mode" />
        <button class="btn btn-primary btn-full" onclick="document.getElementById('calFile').click()">+ Upload Genuine Reference Screenshot</button>
        <input id="calFile" type="file" accept="image/*" onchange="handleCalibrate(event)" />
        <div id="calStatus" style="margin-top:12px; font-size:0.88rem;"></div>
      </div>
    </section>
  </main>

  <nav class="tabs">
    <button class="tab-btn active" onclick="switchTab('scan', this)">🔍 Scan</button>
    <button class="tab-btn" onclick="switchTab('history', this)">📋 History</button>
    <button class="tab-btn" onclick="switchTab('calibrate', this)">⚙️ Calibrate</button>
  </nav>

  <script>
    let lastPreviewUrl = '';

    function getUserId() { return localStorage.getItem('upi_guard_uid') || ''; }
    function getUserLabel() { return localStorage.getItem('upi_guard_name') || 'Guest'; }

    function openLogin() {
      document.getElementById('shopNameInput').value = localStorage.getItem('upi_guard_name') || '';
      document.getElementById('shopPinInput').value = localStorage.getItem('upi_guard_pin') || '';
      document.getElementById('loginOverlay').classList.add('show');
    }

    function saveUserLogin() {
      const name = document.getElementById('shopNameInput').value.trim();
      const pin = document.getElementById('shopPinInput').value.trim() || '0000';
      if (!name) { alert('Please enter your Shop or Staff name.'); return; }
      const uid = name.toLowerCase().replace(/[^a-z0-9]/g, '_') + '_' + pin;
      localStorage.setItem('upi_guard_name', name);
      localStorage.setItem('upi_guard_pin', pin);
      localStorage.setItem('upi_guard_uid', uid);
      document.getElementById('activeUserLabel').textContent = name;
      document.getElementById('loginOverlay').classList.remove('show');
      document.getElementById('resultArea').innerHTML = '<div class="empty-state"><div style="font-size:2.3rem;">⚡</div><h3>Ready for 3-Second Verification</h3><p>Signed in as <strong>' + name + '</strong>. Your history and references are now isolated to this account.</p></div>';
      loadStats(); loadHistory(); loadRefCounts();
    }

    function initUser() {
      if (!getUserId()) { openLogin(); }
      else {
        document.getElementById('activeUserLabel').textContent = getUserLabel();
        loadStats();
      }
    }

    async function loadStats() {
      if (!getUserId()) return;
      try {
        const res = await fetch('/api/stats?user_id=' + encodeURIComponent(getUserId()));
        const s = await res.json();
        document.getElementById('kpiTotal').textContent = s.total_scans;
        document.getElementById('kpiFlagged').textContent = s.flagged_count;
        document.getElementById('kpiSpeed').textContent = s.avg_latency_ms ? s.avg_latency_ms + ' ms' : '—';
        document.getElementById('kpiRefs').textContent = s.total_references;
      } catch (e) {}
    }

    function switchTab(name, btn) {
      document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('view-' + name).classList.add('active');
      btn.classList.add('active');
      if (name === 'history') loadHistory();
      if (name === 'calibrate') loadRefCounts();
    }

    const dz = document.getElementById('dropzone');
    dz.addEventListener('dragover', (e) => { e.preventDefault(); });
    dz.addEventListener('drop', (e) => {
      e.preventDefault();
      if (e.dataTransfer.files && e.dataTransfer.files[0]) processSelectedFile(e.dataTransfer.files[0]);
    });

    function compressImage(file, maxH = 1080) {
      return new Promise((resolve) => {
        const img = new Image(), url = URL.createObjectURL(file);
        img.onload = () => {
          URL.revokeObjectURL(url);
          let w = img.width, h = img.height;
          if (h > maxH) { w = Math.round(w * (maxH / h)); h = maxH; }
          const c = document.createElement('canvas'); c.width = w; c.height = h;
          c.getContext('2d').drawImage(img, 0, 0, w, h);
          c.toBlob((b) => resolve(b || file), 'image/jpeg', 0.90);
        };
        img.onerror = () => resolve(file);
        img.src = url;
      });
    }

    function handleFile(event) {
      const file = event.target.files[0]; event.target.value = '';
      if (file) processSelectedFile(file);
    }

    async function processSelectedFile(file) {
      if (!getUserId()) { openLogin(); return; }
      if (lastPreviewUrl) URL.revokeObjectURL(lastPreviewUrl);
      lastPreviewUrl = URL.createObjectURL(file);
      const thumb = document.getElementById('previewThumb');
      thumb.src = lastPreviewUrl; thumb.style.display = 'inline-block';

      const loader = document.getElementById('loader'), area = document.getElementById('resultArea');
      loader.style.display = 'block'; area.innerHTML = '';
      const t0 = performance.now();
      try {
        const blob = await compressImage(file, 1080);
        const form = new FormData();
        form.append('file', blob, 'screenshot.jpg');
        form.append('user_id', getUserId());
        form.append('strict_live_mode', document.getElementById('strictLive').checked ? 'true' : 'false');
        const exp = document.getElementById('expectedAmount').value.trim();
        if (exp) form.append('expected_amount', exp);
        const res = await fetch('/api/analyze', { method: 'POST', body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Server error');
        renderVerdict(data, Math.round(performance.now() - t0));
        loadStats();
      } catch (e) {
        area.innerHTML = `<div style="padding:16px; border-radius:12px; background:var(--red-bg); border:1px solid var(--red); color:#fca5a5;">Error: ${e.message}</div>`;
      } finally { loader.style.display = 'none'; }
    }

    function renderVerdict(data, totalMs) {
      const cls = data.verdict === 'REAL' ? 'REAL' : (data.verdict === 'SUSPICIOUS' ? 'SUSPICIOUS' : 'FAKE');
      const icon = cls === 'REAL' ? '✅' : (cls === 'SUSPICIOUS' ? '⚠️' : '🚨');
      const strokeColor = cls === 'REAL' ? '#22c55e' : (cls === 'SUSPICIOUS' ? '#f59e0b' : '#ef4444');
      const circ = 2 * Math.PI * 28, offset = circ - (data.risk_score / 100) * circ;
      const p = data.pillars || { layout: 'PASS', pixel: 'PASS', timestamp: 'PASS', utr: 'PASS' };
      const reasonsHtml = (data.reasons || []).map(r => `<li class="reason-item reason-${r.severity}"><span class="reason-cat">${r.category} ${r.points > 0 ? '(+'+r.points+' risk)' : ''}</span>${r.message}</li>`).join('');

      document.getElementById('resultArea').innerHTML = `
        <div class="verdict-banner v-${cls}">
          <div>
            <div class="v-title">${icon} ${data.verdict}</div>
            <div class="v-sub">App: <strong>${data.app.display_name}</strong> &middot; ⚡ <strong>${totalMs} ms</strong></div>
          </div>
          <div class="gauge">
            <svg viewBox="0 0 68 68">
              <circle cx="34" cy="34" r="28" fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="6" />
              <circle cx="34" cy="34" r="28" fill="none" stroke="${strokeColor}" stroke-width="6" stroke-dasharray="${circ}" stroke-dashoffset="${offset}" stroke-linecap="round" />
            </svg>
            <div class="gauge-num">${data.risk_score}</div>
          </div>
        </div>
        <div class="pillars">
          <div class="pillar"><span>Layout Match</span><span class="tag tag-${p.layout}">${p.layout}</span></div>
          <div class="pillar"><span>Pixel &amp; Font</span><span class="tag tag-${p.pixel}">${p.pixel}</span></div>
          <div class="pillar"><span>Timestamp</span><span class="tag tag-${p.timestamp}">${p.timestamp}</span></div>
          <div class="pillar"><span>UTR / Reuse</span><span class="tag tag-${p.utr}">${p.utr}</span></div>
        </div>
        <div class="meta-grid">
          <div class="meta-item"><div class="meta-k">Amount</div><div class="meta-v">${data.extracted.amount ? '₹' + data.extracted.amount : 'Not detected'}</div></div>
          <div class="meta-item"><div class="meta-k">12-Digit UTR</div><div class="meta-v">${data.extracted.utr || 'Not detected'}</div></div>
          <div class="meta-item"><div class="meta-k">Timestamp</div><div class="meta-v">${data.extracted.timestamp || 'Not detected'}</div></div>
          <div class="meta-item"><div class="meta-k">Ref Match</div><div class="meta-v">${data.reference_similarity >= 0 ? Math.round(data.reference_similarity * 100) + '%' : 'Not calibrated'}</div></div>
        </div>
        <ul class="reason-list">${reasonsHtml}</ul>
        ${data.ela_heatmap ? `
          <button class="btn btn-secondary btn-full" style="padding:11px; font-size:0.86rem;" onclick="const g=document.getElementById('cmpGrid'); g.style.display=g.style.display==='grid'?'none':'grid';">🔬 Inspect Side-by-Side ELA Tamper Heatmap</button>
          <div id="cmpGrid" class="compare-grid">
            <div><img src="${lastPreviewUrl}" alt="Original" /><div class="compare-cap">Uploaded Screenshot</div></div>
            <div><img src="${data.ela_heatmap}" alt="ELA Heatmap" /><div class="compare-cap">ELA Tamper Heatmap</div></div>
          </div>` : ''}
        <details><summary style="cursor:pointer;">View Extracted OCR Text</summary><pre>${data.ocr_preview || 'No text extracted.'}</pre></details>`;
    }

    async function loadHistory() {
      if (!getUserId()) return;
      const box = document.getElementById('historyList');
      const res = await fetch('/api/history?user_id=' + encodeURIComponent(getUserId()));
      const data = await res.json();
      box.innerHTML = data.scans.length ? data.scans.map(s => {
        const cls = s.verdict === 'REAL' ? 'REAL' : (s.verdict === 'SUSPICIOUS' ? 'SUSPICIOUS' : 'FAKE');
        return `<div class="history-row"><div><div style="font-weight:700; color:var(--text);">${s.amount ? '₹'+s.amount : '₹?'} &middot; ${s.app_key.toUpperCase()}</div><div style="font-size:0.76rem; color:var(--muted);">UTR: ${s.utr || 'N/A'} &middot; ${s.created_at}</div></div><div style="text-align:right;"><div class="badge-${cls}">${s.verdict} (${s.risk_score})</div><div style="font-size:0.74rem; color:var(--muted);">⚡ ${s.latency_ms} ms</div></div></div>`;
      }).join('') : 'No scans yet for ' + getUserLabel() + '.';
    }

    async function clearMyHistory() {
      if (!confirm('Clear all scan history for ' + getUserLabel() + '?')) return;
      await fetch('/api/history?user_id=' + encodeURIComponent(getUserId()), { method: 'DELETE' });
      loadHistory(); loadStats();
    }

    async function loadRefCounts() {
      if (!getUserId()) return;
      const res = await fetch('/api/references?user_id=' + encodeURIComponent(getUserId()));
      const c = (await res.json()).counts;
      document.getElementById('refCounts').innerHTML = `Saved References — GPay: <strong>${c.gpay}</strong> &middot; PhonePe: <strong>${c.phonepe}</strong> &middot; Paytm: <strong>${c.paytm}</strong>`;
    }

    async function handleCalibrate(event) {
      const file = event.target.files[0]; if (!file) return; event.target.value = '';
      const status = document.getElementById('calStatus'); status.textContent = 'Saving reference...';
      const form = new FormData();
      form.append('user_id', getUserId());
      form.append('app_key', document.getElementById('calApp').value);
      form.append('label', document.getElementById('calLabel').value || 'Genuine Reference');
      form.append('file', file);
      const res = await fetch('/api/calibrate', { method: 'POST', body: form });
      const data = await res.json();
      status.innerHTML = res.ok ? `<span style="color:#4ade80;">Saved! Total for ${data.app_key.toUpperCase()}: ${data.total_references_for_app}</span>` : `<span style="color:#f87171;">${data.detail}</span>`;
      loadRefCounts(); loadStats();
    }

    initUser();
  </script>
</body>
</html>
"""


# ============================================================================
# 5. FASTAPI APP & ROUTES (Per-User Isolation + Shield App Icon)
# ============================================================================

def create_app():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    api = FastAPI(title="UPI Guard Pro v3", version="3.0.0")
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    init_db()

    def _build_icon_png(size: int = 512) -> bytes:
        y_idx, x_idx = np.mgrid[0:size, 0:size].astype(np.float32) / size
        t = np.clip(0.55 * x_idx + 0.45 * y_idx, 0.0, 1.0)
        r = (14 * (1 - t) + 79 * t).astype(np.uint8)
        g = (165 * (1 - t) + 70 * t).astype(np.uint8)
        b = (233 * (1 - t) + 229 * t).astype(np.uint8)
        img = Image.fromarray(np.stack([r, g, b], axis=2), mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(img)
        pad = int(size * 0.10)
        draw.rounded_rectangle([pad, pad, size - pad, size - pad], radius=int(size * 0.20), fill=(10, 18, 36, 215), outline=(56, 189, 248, 220), width=max(4, size // 64))
        cx, cy, sw, sh = size // 2, int(size * 0.50), int(size * 0.26), int(size * 0.30)
        shield = [(cx, cy - sh), (cx + sw, cy - int(sh * 0.62)), (cx + int(sw * 0.88), cy + int(sh * 0.18)), (cx, cy + sh), (cx - int(sw * 0.88), cy + int(sh * 0.18)), (cx - sw, cy - int(sh * 0.62))]
        draw.polygon(shield, fill=(16, 185, 129, 255), outline=(236, 253, 245, 255))
        draw.line([(cx - int(sw * 0.38), cy + int(sh * 0.02)), (cx - int(sw * 0.08), cy + int(sh * 0.32)), (cx + int(sw * 0.42), cy - int(sh * 0.26))], fill=(255, 255, 255, 255), width=max(10, size // 22), joint="curve")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    _ICON_BYTES = _build_icon_png(512)

    print("[UPI Guard] Pre-warming OCR engine...")
    _get_ocr()
    print("[UPI Guard] OCR status:", "READY" if _OCR_ENGINE is not None else f"UNAVAILABLE ({_OCR_ERROR})")

    @api.get("/", response_class=HTMLResponse)
    def home():
        return MOBILE_HTML

    @api.get("/icon.png")
    def app_icon():
        return Response(content=_ICON_BYTES, media_type="image/png")

    @api.get("/manifest.json")
    def app_manifest():
        return JSONResponse({
            "name": "UPI Guard Pro",
            "short_name": "UPI Guard",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#070b14",
            "theme_color": "#070b14",
            "icons": [{"src": "/icon.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}],
        })

    @api.get("/api/stats")
    def api_stats(user_id: str = "default"):
        return get_stats(user_id)

    @api.post("/api/analyze")
    async def api_analyze(
        file: UploadFile = File(...),
        user_id: str = Form(default="default"),
        expected_amount: Optional[str] = Form(default=None),
        strict_live_mode: str = Form(default="false"),
    ):
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file.")
        try:
            strict = strict_live_mode.strip().lower() == "true"
            return analyze_screenshot(raw, user_id=user_id, expected_amount=expected_amount, strict_live_mode=strict)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @api.get("/api/history")
    def api_history(user_id: str = "default", limit: int = 25):
        return {"scans": get_recent_scans(user_id, limit=limit)}

    @api.delete("/api/history")
    def api_clear_history(user_id: str = "default"):
        deleted = clear_user_scans(user_id)
        return {"deleted": deleted}

    @api.post("/api/calibrate")
    async def api_calibrate(
        user_id: str = Form(default="default"),
        app_key: str = Form(...),
        label: str = Form(default="Genuine Reference"),
        file: UploadFile = File(...),
    ):
        key = app_key.strip().lower()
        if key not in UPI_PROFILES:
            raise HTTPException(status_code=400, detail=f"Choose from {list(UPI_PROFILES.keys())}")
        bgr = cv2.imdecode(np.frombuffer(await file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image.")
        ref_id = save_reference_template(user_id, key, label, compute_layout_signature(bgr))
        return {"status": "saved", "reference_id": ref_id, "app_key": key, "total_references_for_app": len(get_reference_templates(user_id, key))}

    @api.get("/api/references")
    def api_references(user_id: str = "default"):
        refs = get_reference_templates(user_id)
        counts = {k: 0 for k in UPI_PROFILES}
        for r in refs:
            if r["app_key"] in counts:
                counts[r["app_key"]] += 1
        return {"counts": counts, "total": len(refs)}

    return api


# ============================================================================
# 6. BUILT-IN SELF-TEST (`python upi_guard_single_file.py --test`)
# ============================================================================

def _run_self_tests():
    def mock_phonepe(cover_patch: bool = False) -> bytes:
        img = np.full((1140, 540, 3), 255, dtype=np.uint8)
        img[0:320, :] = (67, 135, 28)
        cv2.circle(img, (270, 120), 42, (255, 255, 255), -1)
        cv2.putText(img, "Transaction Successful", (75, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, "Paid to Canteen", (40, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
        cv2.putText(img, "Rs. 120", (380, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
        if cover_patch:
            cv2.rectangle(img, (340, 370), (515, 440), (240, 240, 240), -1)
            cv2.putText(img, "Rs. 950", (355, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return buf.tobytes()

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        init_db(db)
        now = datetime(2026, 11, 21, 0, 45, 0)
        old_ts = (now - timedelta(hours=5)).strftime("%d %b %Y, %I:%M %p")
        fut_ts = (now + timedelta(minutes=45)).strftime("%d %b %Y, %I:%M %p")

        g_bytes = mock_phonepe(False)
        g_ocr = f"Transaction Successful\n{old_ts}\nPaid to Campus Canteen\n₹120\nTransaction ID T26112020081234567890\nUTR 632412345678\nPowered by PhonePe"

        # 1. Real older receipt (Strict Live Mode OFF) -> REAL for Shop A
        r1 = analyze_screenshot(g_bytes, user_id="shop_a", expected_amount="120", strict_live_mode=False, override_ocr_text=g_ocr, now=now, db_path=db)
        assert r1["verdict"] == "REAL" and r1["latency_ms"] < 3000

        # 2. Same screenshot on Shop B -> still REAL (proves per-user isolation!)
        r_iso = analyze_screenshot(g_bytes, user_id="shop_b", expected_amount="120", strict_live_mode=False, override_ocr_text=g_ocr, now=now, db_path=db)
        assert r_iso["verdict"] == "REAL"

        # 3. Reused screenshot on Shop A -> LIKELY FAKE
        r2 = analyze_screenshot(g_bytes, user_id="shop_a", expected_amount="120", strict_live_mode=False, override_ocr_text=g_ocr, now=now, db_path=db)
        assert r2["verdict"] == "LIKELY FAKE" and r2["pillars"]["utr"] == "FAIL"

        # 4. Cover-up patch -> flags Pixel Inconsistency
        r3 = analyze_screenshot(mock_phonepe(True), user_id="shop_c", override_ocr_text=f"Transaction Successful\n{old_ts}\nPaid to Campus Canteen\n₹950\nTransaction ID T26112020087777777777\nUTR 632477777777\nPowered by PhonePe", now=now, db_path=db)
        assert r3["pillars"]["pixel"] == "FAIL"

        # 5. Impossible future timestamp -> LIKELY FAKE
        r4 = analyze_screenshot(g_bytes, user_id="shop_d", override_ocr_text=f"Transaction Successful\n{fut_ts}\nPaid to Campus Canteen\n₹120\nTransaction ID T26112020089999999999\nUTR 632499999999\nPowered by PhonePe", now=now, db_path=db)
        assert r4["verdict"] == "LIKELY FAKE" and r4["pillars"]["timestamp"] == "FAIL"

        print(f"ALL 5 MULTI-USER & FORENSIC TESTS PASSED! (Genuine latency: {r1['latency_ms']} ms)")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_self_tests()
    else:
        import uvicorn
        uvicorn.run(create_app(), host="0.0.0.0", port=8000)