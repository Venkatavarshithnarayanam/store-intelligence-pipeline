"""
Detection module for person detection from video frames.
Supports both YOLOv8 real detection and mock detection for testing.

Fallback detection for CPU-only environments:
If YOLOv8 is not available, MockDetector provides deterministic simulations
with realistic movement patterns for testing and development.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime
import random
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. YOLOv8 detection disabled.")
except Exception as e:
    YOLO_AVAILABLE = False
    print(f"Warning: ultralytics import failed: {e}. YOLOv8 detection disabled.")


@dataclass
class Detection:
    """Represents a single detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int  # 0 = person
    
    def to_tuple(self) -> Tuple[float, float, float, float, float, int]:
        """Convert to tuple format for tracker."""
        return (self.x1, self.y1, self.x2, self.y2, self.confidence, self.class_id)
    
    def get_centroid(self) -> Tuple[float, float]:
        """Get detection centroid."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def get_area(self) -> float:
        """Get detection area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def get_aspect_ratio(self) -> float:
        """Get detection aspect ratio (height/width)."""
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return height / width if width > 0 else 0
    
    def is_likely_staff(self) -> bool:
        """
        Heuristic to detect staff members.
        Staff typically have higher confidence and different aspect ratios.
        """
        # High confidence + tall aspect ratio suggests staff uniform
        return self.confidence > 0.9 and self.get_aspect_ratio() > 2.0


class MockDetector:
    """
    Mock detector for testing without YOLOv8.
    Simulates person detections in a frame.
    """
    
    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        """
        Initialize detector.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.person_class_id = 0
    
    def detect(self, frame_index: int, num_people: int = 3) -> List[Detection]:
        """
        Generate mock detections for a frame.
        
        Args:
            frame_index: Frame number (for deterministic simulation)
            num_people: Number of people to simulate
        
        Returns:
            List of Detection objects
        """
        detections = []
        
        # Simulate people moving through frame
        for i in range(num_people):
            # Deterministic but varying positions based on frame
            base_x = (frame_index * 10 + i * 200) % (self.frame_width - 100)
            base_y = 100 + (i * 300) % (self.frame_height - 300)
            
            # Add some randomness
            x1 = base_x + random.randint(-20, 20)
            y1 = base_y + random.randint(-20, 20)
            x2 = x1 + random.randint(50, 100)
            y2 = y1 + random.randint(100, 150)
            
            # Clamp to frame bounds
            x1 = max(0, min(x1, self.frame_width - 1))
            y1 = max(0, min(y1, self.frame_height - 1))
            x2 = max(x1 + 50, min(x2, self.frame_width))
            y2 = max(y1 + 100, min(y2, self.frame_height))
            
            confidence = random.uniform(0.7, 0.99)
            
            detections.append(Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=confidence,
                class_id=self.person_class_id
            ))
        
        return detections
    
    def get_timestamp(self, frame_index: int, fps: int = 15) -> str:
        """
        Generate ISO-8601 timestamp for frame.
        
        Args:
            frame_index: Frame number
            fps: Frames per second
        
        Returns:
            ISO-8601 UTC timestamp string
        """
        seconds = frame_index / fps
        dt = datetime.utcnow()
        # In real scenario, would use actual video timestamp
        return dt.isoformat() + "Z"


class YOLOv8Detector:
    """
    Real YOLOv8 detector for person detection.
    Downloads and uses YOLOv8 nano model for efficient inference.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.3):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to YOLOv8 model weights (downloads if not exists)
            confidence_threshold: Minimum confidence for detections
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path)  # Downloads model if not exists
        self.person_class_id = 0  # COCO person class
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 inference on frame.
        
        Args:
            frame: Input frame (numpy array, BGR format)
        
        Returns:
            List of Detection objects (person class only)
        """
        # Run inference
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Filter for person class only
                    if int(box.cls) == self.person_class_id:
                        confidence = float(box.conf)
                        
                        # Filter by confidence threshold
                        if confidence >= self.confidence_threshold:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            detections.append(Detection(
                                x1=float(x1),
                                y1=float(y1),
                                x2=float(x2),
                                y2=float(y2),
                                confidence=confidence,
                                class_id=self.person_class_id
                            ))
        
        return detections
    
    def detect_from_video(self, video_path: str, max_frames: Optional[int] = None) -> List[Tuple[int, List[Detection], str]]:
        """
        Process entire video file and return detections for each frame.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None for all)
        
        Returns:
            List of (frame_index, detections, timestamp) tuples
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_results = []
        frame_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_index >= max_frames:
                break
            
            # Get detections for this frame
            detections = self.detect(frame)
            
            # Generate timestamp
            timestamp = self.get_timestamp(frame_index, fps)
            
            frame_results.append((frame_index, detections, timestamp))
            frame_index += 1
        
        cap.release()
        return frame_results
    
    def get_timestamp(self, frame_index: int, fps: float) -> str:
        """
        Generate ISO-8601 timestamp for frame.
        
        Args:
            frame_index: Frame number
            fps: Frames per second
        
        Returns:
            ISO-8601 UTC timestamp string
        """
        seconds = frame_index / fps
        dt = datetime.utcnow()
        # In real scenario, would use actual video timestamp or add seconds offset
        return dt.isoformat() + "Z"


class VideoProcessor:
    """
    Processes video files with YOLOv8 detection.
    Handles multiple video formats and provides frame-by-frame processing.
    """
    
    def __init__(self, detector: YOLOv8Detector):
        """
        Initialize video processor.
        
        Args:
            detector: YOLOv8Detector instance
        """
        self.detector = detector
    
    def process_video_file(self, video_path: str, store_id: str, camera_id: str) -> List[Tuple[int, List[Detection], str]]:
        """
        Process single video file.
        
        Args:
            video_path: Path to video file
            store_id: Store identifier
            camera_id: Camera identifier
        
        Returns:
            List of (frame_index, detections, timestamp) tuples
        """
        print(f"Processing video: {video_path} for {store_id}/{camera_id}")
        
        try:
            frame_results = self.detector.detect_from_video(video_path)
            print(f"Processed {len(frame_results)} frames from {video_path}")
            return frame_results
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return []
    
    def process_video_directory(self, video_dir: str, store_id: str) -> dict:
        """
        Process all video files in directory.
        
        Args:
            video_dir: Directory containing video files
            store_id: Store identifier
        
        Returns:
            Dict mapping camera_id to frame results
        """
        import os
        from pathlib import Path
        
        video_dir_path = Path(video_dir)
        if not video_dir_path.exists():
            raise ValueError(f"Video directory not found: {video_dir}")
        
        # Map video files to camera IDs
        camera_mapping = {
            "CAM 1.mp4": "CAM_ENTRY_01",
            "CAM 2.mp4": "CAM_FLOOR_01", 
            "CAM 3.mp4": "CAM_BILLING_01",
            "CAM_1.mp4": "CAM_ENTRY_01",
            "CAM_2.mp4": "CAM_FLOOR_01",
            "CAM_3.mp4": "CAM_BILLING_01"
        }
        
        results = {}
        
        # Process each video file
        for video_file in video_dir_path.glob("*.mp4"):
            filename = video_file.name
            camera_id = camera_mapping.get(filename)
            
            if camera_id:
                frame_results = self.process_video_file(str(video_file), store_id, camera_id)
                results[camera_id] = frame_results
            else:
                print(f"Unknown video file format: {filename}")
        
        return results
