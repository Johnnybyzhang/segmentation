import os
import cv2
import subprocess
import argparse
from dataclasses import dataclass
from typing import List

@dataclass
class Segment:
    segment_type: str  # 'still' or 'animation'
    start_frame: int
    end_frame: int

def get_video_properties(video_path: str):
    """Retrieve video properties like framerate and total frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Failed to retrieve FPS from the video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames

def compare_frames(frame1, frame2):
    """Compare two frames for exact equality."""
    return (frame1 == frame2).all()

def segment_video(video_path: str, fps: float, total_frames: int) -> List[Segment]:
    """Process video frames and identify still and animation segments."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    segments = []
    current_segment_start = 0
    current_segment_type = None

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Failed to read the first frame from the video.")

    frame_idx = 0

    while True:
        ret, next_frame = cap.read()
        if not ret:
            # End of video; close the last segment
            if current_segment_type is not None:
                segments.append(Segment(
                    segment_type=current_segment_type,
                    start_frame=current_segment_start,
                    end_frame=frame_idx
                ))
            break

        frame_idx += 1

        if compare_frames(prev_frame, next_frame):
            # Current frame is identical to the previous frame
            if current_segment_type == 'animation':
                # Close the current animation segment
                segments.append(Segment(
                    segment_type='animation',
                    start_frame=current_segment_start,
                    end_frame=frame_idx - 1
                ))
                # Start a new still segment
                current_segment_start = frame_idx - 1
                current_segment_type = 'still'
            elif current_segment_type is None:
                # Starting with a still segment
                current_segment_type = 'still'
            # If already in 'still', continue
        else:
            # Current frame is different from the previous frame
            if current_segment_type == 'still':
                # Close the current still segment
                segments.append(Segment(
                    segment_type='still',
                    start_frame=current_segment_start,
                    end_frame=frame_idx - 1
                ))
                # Start a new animation segment
                current_segment_start = frame_idx - 1
                current_segment_type = 'animation'
            elif current_segment_type is None:
                # Starting with an animation segment
                current_segment_type = 'animation'
            # If already in 'animation', continue

        prev_frame = next_frame

    cap.release()
    return segments

def map_segments_to_timestamps(segments: List[Segment], fps: float):
    """Convert frame indices to timestamps based on framerate."""
    mapped_segments = []
    for seg in segments:
        start_time = seg.start_frame / fps
        # end_frame is inclusive, so we add 1 frame's duration
        duration = (seg.end_frame - seg.start_frame + 1) / fps
        mapped_segments.append({
            'type': seg.segment_type,
            'start_time': start_time,
            'duration': duration
        })
    return mapped_segments

def cut_segments_with_ffmpeg(video_path: str, segments: List[dict], output_dir: str):
    """Use ffmpeg to cut the video into segments without re-encoding."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, seg in enumerate(segments):
        segment_type = seg['type']
        start_time = seg['start_time']
        duration = seg['duration']
        out_file = os.path.join(output_dir, f"segment_{idx:03d}_{segment_type}.mp4")

        # ffmpeg command to cut without re-encoding
        # Using -c copy attempts to avoid re-encoding.
        # Note: frame-accurate cuts may require additional steps depending on your source format.
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-i", video_path,
            "-ss", f"{start_time:.3f}",
            "-t", f"{duration:.3f}",
            "-c", "copy",
            out_file
        ]

        print(f"Processing segment {idx+1}/{len(segments)}: {segment_type}")
        print("Running command:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Segment saved to: {out_file}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error processing segment {idx}: {e.stderr.decode()}")
            print("Skipping this segment.\n")

def main():
    parser = argparse.ArgumentParser(description="Segment video into still and animation parts.")
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("-o", "--output", default="cuts", help="Output directory for segments.")
    args = parser.parse_args()

    video_path = args.video
    output_dir = args.output

    if not os.path.isfile(video_path):
        print(f"Error: File '{video_path}' does not exist.")
        return

    print(f"Processing video: {video_path}")

    try:
        fps, total_frames = get_video_properties(video_path)
        print(f"Detected framerate: {fps} FPS")
        print(f"Total frames: {total_frames}")
    except Exception as e:
        print(f"Error retrieving video properties: {e}")
        return

    print("Segmenting video based on frame differences...")
    try:
        segments = segment_video(video_path, fps, total_frames)
        print(f"Identified {len(segments)} segments.\n")
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return

    print("Mapping segments to timestamps...")
    mapped_segments = map_segments_to_timestamps(segments, fps)

    print("Cutting segments using ffmpeg...")
    cut_segments_with_ffmpeg(video_path, mapped_segments, output_dir)

    print("All segments have been processed and saved.")

if __name__ == "__main__":
    main()
