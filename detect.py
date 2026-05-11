import imgui
import glfw
from imgui.integrations.glfw import GlfwRenderer
from OpenGL.GL import *
import cv2
import threading
import numpy as np
from collections import Counter
from ultralytics import YOLO
import time
from datetime import datetime
import winsound
from pathlib import Path

# ──────────────────────────────────────────────
#  10 known classes (matches best.pt)
# ──────────────────────────────────────────────
KNOWN_CLASSES = {
    0: "Beef",
    1: "Canned Goods",
    2: "Cheese",
    3: "Egg",
    4: "Fish",
    5: "Baboy",
    6: "Man",
    7: "Milk",
    8: "Sausage",
    9: "Chicken",
}

CLASS_BEEPS = {
    "Beef":         (400,  80),
    "Canned Goods": (500,  80),
    "Cheese":       (600,  80),
    "Egg":          (700,  80),
    "Fish":         (800,  80),
    "Mamoy":        (900,  80),
    "Man":          (1000, 80),
    "Milk":         (1100, 80),
    "Sausage":      (1200, 80),
    "Siken":        (1300, 80),
}

CHIP_COLORS = [
    (0.94, 0.27, 0.27),  # Beef (red)
    (0.98, 0.45, 0.09),  # Canned Goods (orange)
    (0.92, 0.70, 0.03),  # Cheese (yellow)
    (0.13, 0.77, 0.31),  # Egg (green)
    (0.02, 0.71, 0.83),  # Fish (cyan)
    (0.23, 0.51, 0.96),  # Mamoy (blue)
    (0.55, 0.36, 0.96),  # Man (purple)
    (0.93, 0.29, 0.60),  # Milk (pink)
    (0.08, 0.72, 0.65),  # Sausage (teal)
    (0.96, 0.62, 0.04),  # Siken (amber)
]

CV_COLORS = [
    (68,  68,  239),
    (22, 115, 249),
    (8,  179, 234),
    (94, 197,  34),
    (212,182,   6),
    (246,115,  59),
    (246, 92, 236),
    (185, 96, 139),
    (166,184,  20),
    (11, 184, 212),
]

# ── Theme ──────────────────────────────────────
THEME_BG      = (0.12, 0.12, 0.18, 1.0)
THEME_PANEL   = (0.16, 0.16, 0.24, 1.0)
THEME_ACCENT  = (0.09, 0.64, 0.29, 1.0)
THEME_ACCENT2 = (0.53, 0.94, 0.68, 1.0)
THEME_TEXT    = (0.88, 0.91, 0.94, 1.0)
THEME_SUBTEXT = (0.58, 0.64, 0.71, 1.0)
THEME_SUCCESS = (0.13, 0.77, 0.31, 1.0)
THEME_DANGER  = (0.94, 0.27, 0.27, 1.0)
THEME_WARNING = (0.96, 0.62, 0.04, 1.0)
THEME_TEAL    = (0.02, 0.71, 0.83, 1.0)


class DetectionApp:
    def __init__(self):
        self.window_width = 1380
        self.window_height = 820
        
        # OpenGL/GLFW setup
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(
            self.window_width, self.window_height,
            "🐄  Joweeee's Livestock Image Detector", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create window")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # ImGui setup
        imgui.create_context()
        # Use GlfwRenderer integration from pyimgui
        self.impl = GlfwRenderer(self.window)
        imgui.get_io().display_size = (self.window_width, self.window_height)
        
        self._setup_imgui_style()
        
        # App state
        self.model = None
        self.cap = None
        self.running = False
        self.paused = False
        self.thread = None
        
        self.confidence = 0.60
        self.model_path = "best.pt"
        self.camera_index = 0
        self.total_detected = 0
        self.fps_value = 0.0
        self.status = "● Idle"
        self.sound_enabled = True
        self.detection_log = []
        self.log_limit = 300
        
        self.class_filter_vars = {name: True for name in KNOWN_CLASSES.values()}
        self.class_counts = {}
        
        # Camera frame
        self.frame_texture = None
        self.frame_data = None
        
    def _setup_imgui_style(self):
        """Apply dark theme to ImGui"""
        style = imgui.get_style()
        style.window_rounding = 5.0
        style.frame_rounding = 3.0
        style.button_text_align = (0.5, 0.5)
        
        # Color scheme
        colors = style.colors
        colors[imgui.COLOR_WINDOW_BACKGROUND] = THEME_BG
        colors[imgui.COLOR_FRAME_BACKGROUND] = THEME_PANEL
        colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.20, 0.20, 0.28, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.25, 0.25, 0.32, 1.0)
        colors[imgui.COLOR_BUTTON] = THEME_ACCENT
        colors[imgui.COLOR_BUTTON_HOVERED] = (0.12, 0.75, 0.35, 1.0)
        colors[imgui.COLOR_BUTTON_ACTIVE] = (0.08, 0.60, 0.25, 1.0)
        colors[imgui.COLOR_TEXT] = THEME_TEXT
        colors[imgui.COLOR_TEXT_DISABLED] = THEME_SUBTEXT
        colors[imgui.COLOR_HEADER] = THEME_ACCENT
        colors[imgui.COLOR_HEADER_HOVERED] = (0.12, 0.75, 0.35, 1.0)
        colors[imgui.COLOR_HEADER_ACTIVE] = (0.08, 0.60, 0.25, 1.0)
        colors[imgui.COLOR_SEPARATOR] = (0.23, 0.23, 0.32, 1.0)
        colors[imgui.COLOR_SCROLLBAR_GRAB] = THEME_ACCENT
        colors[imgui.COLOR_SCROLLBAR_GRAB_HOVERED] = (0.12, 0.75, 0.35, 1.0)
        colors[imgui.COLOR_SCROLLBAR_GRAB_ACTIVE] = (0.08, 0.60, 0.25, 1.0)
    
    def _beep(self, name: str):
        """Play beep for detected class"""
        if not self.sound_enabled:
            return
        freq, dur = CLASS_BEEPS.get(name, (750, 80))
        threading.Thread(target=winsound.Beep, args=(freq, dur), daemon=True).start()
    
    def _enabled_names(self) -> set:
        """Get enabled class names"""
        return {name for name, enabled in self.class_filter_vars.items() if enabled}
    
    def _browse_model(self):
        """Browse for model file (placeholder - implement based on OS)"""
        # This is a simplified version. For full file dialog, use tkinter or native dialogs
        pass
    
    def _start(self):
        """Start detection"""
        if self.running:
            return
        
        try:
            self.status = "⏳ Loading model…"
            self.model = YOLO(self.model_path)
        except Exception as e:
            self.status = f"● Error: {str(e)}"
            return
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.status = "● Camera Error"
            return
        
        self.running = True
        self.paused = False
        self.total_detected = 0
        self.detection_log.clear()
        self.status = "● Running"
        
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
    
    def _toggle_pause(self):
        """Toggle pause"""
        if not self.running:
            return
        self.paused = not self.paused
        self.status = "⏸ Paused" if self.paused else "● Running"
    
    def _stop(self):
        """Stop detection"""
        self.running = False
        self.paused = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status = "● Idle"
        self.fps_value = 0.0
        self.total_detected = 0
        self.class_counts.clear()
    
    def _clear_log(self):
        """Clear detection log"""
        self.detection_log.clear()
    
    def _selection_all(self):
        """Select all classes"""
        for name in self.class_filter_vars:
            self.class_filter_vars[name] = True
    
    def _select_none(self):
        """Deselect all classes"""
        for name in self.class_filter_vars:
            self.class_filter_vars[name] = False
    
    def _detection_loop(self):
        """Background detection thread"""
        prev_time = time.time()
        prev_names = set()
        
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            conf_thresh = self.confidence
            enabled_names = self._enabled_names()
            class_names = self.model.names
            
            results = self.model(frame, verbose=False)
            detections = results[0].boxes
            
            filtered = [
                d for d in detections
                if float(d.conf[0].item()) >= conf_thresh
                and class_names.get(int(d.cls[0].item()), "") in enabled_names
            ]
            
            class_counts = Counter()
            new_log_entries = []
            current_names = set()
            ts = datetime.now().strftime("%H:%M:%S")
            
            for det in filtered:
                label = int(det.cls[0].item())
                conf = float(det.conf[0].item())
                name = class_names.get(label, str(label))
                class_counts[name] += 1
                current_names.add(name)
                new_log_entries.append((ts, name, f"{conf:.0%}"))
            
            newly_detected = current_names - prev_names
            for name in newly_detected:
                self._beep(name)
            prev_names = current_names
            
            # Draw boxes on frame
            for det in filtered:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                label = int(det.cls[0].item())
                conf = float(det.conf[0].item())
                name = class_names.get(label, str(label))
                color = CV_COLORS[label % len(CV_COLORS)]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                tag_text = f"{name}  {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
                cv2.putText(frame, tag_text, (x1+3, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Store frame and stats for UI update
            self.frame_data = frame_rgb
            self.class_counts = dict(class_counts)
            self.fps_value = fps
            self.total_detected = sum(class_counts.values())
            
            # Append log entries
            for ts, name, conf in new_log_entries:
                self.detection_log.append((ts, name, conf))
            if len(self.detection_log) > self.log_limit:
                self.detection_log = self.detection_log[-self.log_limit:]
        
        self.status = "● Idle"
    
    def _frame_to_texture(self, frame_rgb):
        """Convert numpy frame to OpenGL texture"""
        frame_rgb = np.flip(frame_rgb, axis=2)  # RGB to BGR
        frame_rgb = frame_rgb.astype(np.uint8)
        h, w = frame_rgb.shape[:2]
        
        # Create texture if needed
        if self.frame_texture is None:
            self.frame_texture = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.frame_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)
        
        return self.frame_texture, (w, h)
    
    def render(self):
        """Main render loop"""
        imgui.new_frame()
        
        # Main window
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.window_width, self.window_height)
        imgui.begin("##main", None, 
                   imgui.WINDOW_NO_TITLE_BAR | 
                   imgui.WINDOW_NO_RESIZE |
                   imgui.WINDOW_NO_MOVE)
        
        # Columns: Left (camera + log) | Right (controls)
        imgui.columns(2, "layout", True)
        
        # ──────────────────────────────────────────
        # LEFT COLUMN: Camera + Log
        # ──────────────────────────────────────────
        
        # Camera display
        imgui.text("📷  Camera Feed")
        if self.frame_data is not None:
            try:
                tex_id, (fw, fh) = self._frame_to_texture(self.frame_data)
                w = imgui.get_content_region_available()[0] - 8
                h = (w / fw) * fh if fw > 0 else 200
                imgui.image(tex_id, w, h)
            except Exception:
                imgui.text("(rendering frame...)")
        else:
            imgui.text("(waiting for frame...)")
        
        imgui.separator()
        
        # Status & stats
        imgui.text(f"Status: {self.status}")
        imgui.same_line()
        imgui.text(f"  |  Total: {self.total_detected}")
        imgui.same_line()
        imgui.text(f"  |  FPS: {self.fps_value:.1f}")
        
        imgui.separator()
        
        # Detection log
        imgui.text("📋  Detection Log")
        imgui.begin_child("log_area", 0, 0, True)
        for ts, name, conf in reversed(self.detection_log[-20:]):
            imgui.text(f"{ts} | {name:<15} | {conf}")
        imgui.end_child()
        
        if imgui.button("🗑 Clear Log", -1, 0):
            self._clear_log()
        
        imgui.next_column()
        
        # ──────────────────────────────────────────
        # RIGHT COLUMN: Controls
        # ──────────────────────────────────────────
        
        imgui.begin_child("controls", 0, 0, True)
        
        # Model file
        imgui.text("📂  Model File")
        _, self.model_path = imgui.input_text("##model", self.model_path, 256)
        if imgui.button("Browse##model", -1, 0):
            self._browse_model()
        
        imgui.separator()
        
        # Camera index
        imgui.text("📷  Camera Index")
        _, self.camera_index = imgui.radio_button_int("Cam 0##cam", self.camera_index, 0)
        imgui.same_line()
        _, self.camera_index = imgui.radio_button_int("Cam 1##cam", self.camera_index, 1)
        imgui.same_line()
        _, self.camera_index = imgui.radio_button_int("Cam 2##cam", self.camera_index, 2)
        imgui.same_line()
        _, self.camera_index = imgui.radio_button_int("Cam 3##cam", self.camera_index, 3)
        
        imgui.separator()
        
        # Sound toggle
        imgui.text("🔔  Sound Alerts")
        _, self.sound_enabled = imgui.checkbox("Play sound on detection", self.sound_enabled)
        
        imgui.separator()
        
        # Confidence threshold
        imgui.text("🎚️  Confidence Threshold")
        imgui.text(f"{self.confidence:.2f}", THEME_TEAL)
        _, self.confidence = imgui.slider_float("##conf", self.confidence, 0.10, 0.95)
        
        imgui.text("Preset confidence:")
        if imgui.button("Low (0.40)", 0.32, 0):
            self.confidence = 0.40
        imgui.same_line()
        if imgui.button("Mid (0.60)", 0.32, 0):
            self.confidence = 0.60
        imgui.same_line()
        if imgui.button("High (0.80)", 0.32, 0):
            self.confidence = 0.80
        
        imgui.separator()
        
        # Class filter
        imgui.text("☑️  Class Filter")
        imgui.text("Uncheck to ignore. Only checked classes are counted.")
        
        if imgui.button("✔ All", 0.48, 0):
            self._selection_all()
        imgui.same_line()
        if imgui.button("✖ None", 0.48, 0):
            self._select_none()
        
        for idx, name in KNOWN_CLASSES.items():
            color = CHIP_COLORS[idx]
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *color)
            _, self.class_filter_vars[name] = imgui.checkbox(f"  {name}##cls{idx}", 
                                                              self.class_filter_vars[name])
            imgui.pop_style_color()
        
        imgui.separator()
        
        # Live class counts
        imgui.text("📊  Live Class Counts")
        if self.class_counts:
            for name, count in sorted(self.class_counts.items()):
                idx = next((k for k, v in KNOWN_CLASSES.items() if v == name), 0)
                color = CHIP_COLORS[idx % len(CHIP_COLORS)]
                imgui.text_colored(f"● {name}: {count}", *color)
        else:
            imgui.text("(No detections)", THEME_SUBTEXT)
        
        imgui.separator()
        
        # Control buttons
        imgui.text("▶  Detection Controls")
        
        if not self.running:
            if imgui.button("▶  Start Detection", -1, 40):
                self._start()
        else:
            if imgui.button("⏸  Pause" if not self.paused else "▶  Resume", -1, 40):
                self._toggle_pause()
            
            if imgui.button("⏹  Stop", -1, 40):
                self._stop()
        
        imgui.end_child()
        
        imgui.end()
        
        imgui.end_frame()
    
    def run(self):
        """Main loop"""
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            
            # Clear screen
            glClearColor(*THEME_BG)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Render UI
            self.render()
            
            # Draw ImGui
            self.impl.render(imgui.get_draw_data())
            
            glfw.swap_buffers(self.window)
        
        self._stop()
        self.impl.shutdown()
        glfw.terminate()


if __name__ == "__main__":
    app = DetectionApp()
    app.run()
