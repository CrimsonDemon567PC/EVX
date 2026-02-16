import struct
import cv2
import numpy as np
import os
import argparse
from concurrent.futures import ProcessPoolExecutor

# --- WRITER KLASSE ---

class EVXProWriter:
    """Binärer Writer für EVX Pro v3.0 (RGBA + Geometrie)"""
    def __init__(self, filename):
        self.filename = filename
        # Header: Magic 'EVX' + Version 0x03
        self.data = bytearray(b'EVX\x03') 

    def add_op(self, chunk):
        self.data.extend(chunk)

    def pack_rect(self, x, y, w, h, rgba):
        """OpCode 0x10: RGBA Rectangle"""
        return struct.pack('<BHHHHBBBB', 0x10, int(x), int(y), int(w), int(h), 
                           int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3]))

    def pack_circle(self, cx, cy, r, rgba):
        """OpCode 0x12: RGBA Circle"""
        return struct.pack('<BHHHBBBB', 0x12, int(cx), int(cy), int(r), 
                           int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3]))

    def pack_delta_header(self, timestamp):
        """OpCode 0x40: Video Frame Start"""
        return struct.pack('<BI', 0x40, int(timestamp))

    def save(self):
        self.data.append(0xFF) # EOF Marker
        with open(self.filename, 'wb') as f:
            f.write(self.data)
        print(f"✅ Datei gespeichert: {self.filename} ({len(self.data)/1024:.2f} KB)")

# --- WORKER FUNKTION FÜR PARALLELISIERUNG ---

def process_tile_task(args):
    """Analysiert einen Bildabschnitt auf Quadtree-Ebene."""
    img_tile, x_off, y_off, threshold, max_depth = args
    results = []

    def run_quadtree(roi, x, y, depth):
        h, w = roi.shape[:2]
        if w <= 2 or h <= 2 or depth >= max_depth:
            color = np.mean(roi, axis=(0, 1)).astype(int)
            results.append(('rect', x, y, w, h, color))
            return

        std_dev = np.mean(np.std(roi, axis=(0, 1)))

        # Strategie: Solid Color
        if std_dev < threshold:
            color = np.mean(roi, axis=(0, 1)).astype(int)
            results.append(('rect', x, y, w, h, color))
            return

        # Strategie: Kreis-Erkennung (nur bei quadratischen Blöcken sinnvoll)
        if 0.9 < (w/h) < 1.1 and std_dev > threshold * 1.5:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGBA2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, w, 
                                      param1=50, param2=30, minRadius=w//3, maxRadius=w//2)
            if circles is not None:
                c = circles[0][0]
                color = roi[int(c[1]), int(c[0])]
                results.append(('circle', x + c[0], y + c[1], c[2], color))
                return

        # Rekursiver Split
        hw, hh = w // 2, h // 2
        run_quadtree(roi[0:hh, 0:hw], x, y, depth + 1)
        run_quadtree(roi[0:hh, hw:w], x + hw, y, depth + 1)
        run_quadtree(roi[hh:h, 0:hw], x, y + hh, depth + 1)
        run_quadtree(roi[hh:h, hw:w], x + hw, y + hh, depth + 1)

    run_quadtree(img_tile, x_off, y_off, 0)
    return results

# --- HAUPT KONVERTER ---

class EVXProConverter:
    def __init__(self, threshold=15, workers=None):
        self.threshold = threshold
        self.workers = workers or os.cpu_count()
        self.last_frame = None

    def _prepare_rgba(self, frame):
        if frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)

    def process_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: return
        img_rgba = self._prepare_rgba(img)
        
        writer = EVXProWriter(path.rsplit('.', 1)[0] + ".evx")
        self._parallel_solve(img_rgba, writer)
        writer.save()

    def _parallel_solve(self, img, writer, x_off=0, y_off=0):
        h, w = img.shape[:2]
        # Bild in horizontale Streifen für die Worker unterteilen
        stripe_h = max(1, h // self.workers)
        tasks = []
        for i in range(self.workers):
            y1 = i * stripe_h
            y2 = h if i == self.workers - 1 else (i + 1) * stripe_h
            if y1 >= h: break
            tasks.append((img[y1:y2, :], x_off, y_off + y1, self.threshold, 10))

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            all_ops = executor.map(process_tile_task, tasks)
            for op_list in all_ops:
                for op in op_list:
                    if op[0] == 'rect':
                        writer.add_op(writer.pack_rect(op[1], op[2], op[3], op[4], op[5]))
                    elif op[0] == 'circle':
                        writer.add_op(writer.pack_circle(op[1], op[2], op[3], op[4]))

    def process_video(self, path):
        cap = cv2.VideoCapture(path)
        writer = EVXProWriter(path.rsplit('.', 1)[0] + ".evx")
        f_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            curr_rgba = self._prepare_rgba(frame)
            writer.add_op(writer.pack_delta_header(f_idx * 33))

            if self.last_frame is None:
                self._parallel_solve(curr_rgba, writer)
            else:
                # Delta-Erkennung
                diff = cv2.absdiff(self.last_frame, curr_rgba)
                mask = cv2.threshold(cv2.cvtColor(diff, cv2.COLOR_RGBA2GRAY), self.threshold, 255, cv2.THRESH_BINARY)[1]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w > 4 and h > 4:
                        roi = curr_rgba[y:y+h, x:x+w]
                        # Kleinerer Overhead: ROI direkt verarbeiten statt parallel
                        ops = process_tile_task((roi, x, y, self.threshold, 8))
                        for op in ops:
                            if op[0] == 'rect':
                                writer.add_op(writer.pack_rect(op[1], op[2], op[3], op[4], op[5]))
                            elif op[0] == 'circle':
                                writer.add_op(writer.pack_circle(op[1], op[2], op[3], op[4]))

            self.last_frame = curr_rgba.copy()
            f_idx += 1
            print(f"Frame {f_idx} verarbeitet...", end="\r")

        cap.release()
        writer.save()

# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-t", "--threshold", type=int, default=15)
    args = parser.parse_args()

    conv = EVXProConverter(threshold=args.threshold)
    if args.input.lower().endswith(('.mp4', '.mov', '.avi')):
        conv.process_video(args.input)
    else:
        conv.process_image(args.input)