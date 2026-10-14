"""Original uploaded face-crop preprocessing, retained for this demo.
Training-time preprocessing equivalence still requires the training code.
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image
import torchvision.transforms as transforms


# MediaPipe FaceMesh indices approximating the dlib 68-point layout.
MP_TO_68 = [
    # jawline 0-16
    234, 93, 132, 58, 172, 136, 150, 149, 152, 148, 176, 377, 400, 378, 379, 365, 397,
    # right eyebrow 17-21
    70, 63, 105, 66, 107,
    # left eyebrow 22-26
    336, 296, 334, 293, 300,
    # nose bridge 27-30
    168, 6, 197, 195,
    # nose base 31-35
    5, 4, 1, 2, 94,
    # right eye 36-41
    33, 160, 158, 133, 153, 144,
    # left eye 42-47
    362, 385, 387, 263, 373, 380,
    # outer lip 48-59
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    # inner lip 60-67
    78, 95, 88, 178, 87, 14, 317, 402,
]


class NoFaceDetected(RuntimeError):
    pass


class VideoProcessor:
    """Extract face crops and 68 landmarks from a video."""

    def __init__(self, num_frames=20, target_size=(224, 224), crop_padding=0.20,
                 min_face_ratio=0.30):
        """
        num_frames    : frames sampled per video. Must match training (20).
        crop_padding  : fraction of face box added as margin on each side.
        min_face_ratio: minimum fraction of frames that must contain a face
                        before the video is rejected as unusable.
        """
        self.num_frames = num_frames
        self.target_size = target_size
        self.crop_padding = crop_padding
        self.min_face_ratio = min_face_ratio

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------
    def _face_box(self, landmarks, w, h):
        """Bounding box around all detected landmarks, with padding."""
        xs = np.array([lm.x for lm in landmarks]) * w
        ys = np.array([lm.y for lm in landmarks]) * h
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        pad_x = (x1 - x0) * self.crop_padding
        pad_y = (y1 - y0) * self.crop_padding

        x0 = int(max(0, x0 - pad_x))
        y0 = int(max(0, y0 - pad_y))
        x1 = int(min(w, x1 + pad_x))
        y1 = int(min(h, y1 + pad_y))

        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        return x0, y0, x1, y1

    # ------------------------------------------------------------------
    def process_video(self, video_path, progress_callback=None):
        """
        Returns
            video_tensor    : (1, T, 3, 224, 224) normalised face crops
            landmarks_tensor: (1, T, 68, 3) landmarks normalised to the CROP
            raw_crops       : list of RGB face crops (for display)
            info            : dict with diagnostics
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if total <= 0:
            cap.release()
            raise ValueError("Video reports zero frames.")

        indices = np.linspace(0, max(total - 1, 0), self.num_frames).astype(int)

        frames, landmarks, raw_crops = [], [], []
        detected = 0
        last_good_crop = None
        last_good_lm = None

        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                # reuse previous good frame to keep sequence length fixed
                if last_good_crop is None:
                    continue
                crop_rgb, lm68 = last_good_crop, last_good_lm
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                res = self.face_mesh.process(frame_rgb)

                crop_rgb, lm68 = None, None
                if res.multi_face_landmarks:
                    mp_lms = res.multi_face_landmarks[0].landmark
                    box = self._face_box(mp_lms, w, h)
                    if box is not None:
                        x0, y0, x1, y1 = box
                        crop_rgb = frame_rgb[y0:y1, x0:x1]
                        cw, ch = (x1 - x0), (y1 - y0)

                        pts = []
                        for mi in MP_TO_68:
                            lm = mp_lms[mi]
                            # express landmark relative to the CROP, not the frame
                            px = (lm.x * w - x0) / max(cw, 1)
                            py = (lm.y * h - y0) / max(ch, 1)
                            pts.append([px, py, lm.z])
                        lm68 = np.asarray(pts, dtype=np.float32)
                        detected += 1
                        last_good_crop, last_good_lm = crop_rgb, lm68

                if crop_rgb is None:
                    if last_good_crop is None:
                        continue
                    crop_rgb, lm68 = last_good_crop, last_good_lm

            raw_crops.append(crop_rgb)
            pil = Image.fromarray(crop_rgb).resize(self.target_size)
            frames.append(self.transform(pil))
            landmarks.append(lm68)

            if progress_callback:
                progress_callback(int((i + 1) / len(indices) * 100),
                                  f"Frame {i + 1}/{len(indices)}")

        cap.release()

        if len(frames) == 0:
            raise NoFaceDetected(
                "No face could be detected in any sampled frame. "
                "This model only works on videos containing a clearly visible face."
            )

        # pad to fixed length by repeating the last frame
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
            landmarks.append(landmarks[-1])
            raw_crops.append(raw_crops[-1])

        ratio = detected / float(self.num_frames)
        if ratio < self.min_face_ratio:
            raise NoFaceDetected(
                f"A face was found in only {detected}/{self.num_frames} sampled "
                "frames. The result would not be trustworthy, so it is not shown. "
                "Try a video where the face is larger and clearly visible."
            )

        video_tensor = torch.stack(frames).unsqueeze(0)
        landmarks_tensor = torch.from_numpy(np.asarray(landmarks)).float().unsqueeze(0)

        info = {
            "total_frames": total,
            "fps": fps,
            "sampled": self.num_frames,
            "face_detected_frames": detected,
            "face_detection_ratio": ratio,
        }
        return video_tensor, landmarks_tensor, raw_crops, info
