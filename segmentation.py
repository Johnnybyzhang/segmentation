import os
import cv2
import subprocess
import argparse
from dataclasses import dataclass
from typing import List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for progress bars

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

def load_all_frames(video_path: str, total_frames: int):
    """Load all frames into memory as a list of numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frames = []
    # Initialize tqdm progress bar for frame loading
    with tqdm(total=total_frames, desc="Loading Frames", unit="frame") as pbar:
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            pbar.update(1)
    cap.release()

    if len(frames) != total_frames:
        raise ValueError("Could not read all frames from the video.")
    return frames

def compare_two_frames(args):
    """Compare two frames for exact equality. args is a tuple (frame1, frame2)."""
    frame1, frame2 = args
    return (frame1 == frame2).all()

def segment_video(frames) -> List[Segment]:
    """
    Given a list of frames, identify still and animation segments.
    We assume frames is a list of numpy arrays of the same shape.
    """
    # Create list of frame pairs for comparison
    frame_pairs = [(frames[i], frames[i+1]) for i in range(len(frames)-1)]

    # Use multiprocessing to speed up comparison
    results = []
    with ProcessPoolExecutor() as executor:
        # Initialize tqdm progress bar for frame comparison
        with tqdm(total=len(frame_pairs), desc="Comparing Frames", unit="comparison") as pbar:
            for result in executor.map(compare_two_frames, frame_pairs):
                results.append(result)
                pbar.update(1)

    # Now build segments from the comparison results
    segments = []
    current_segment_start = 0
    current_segment_type = None

    for i, identical in enumerate(results):
        if identical:
            # Frames i and i+1 are identical
            if current_segment_type == 'animation':
                # Close the current animation segment
                segments.append(Segment(
                    segment_type='animation',
                    start_frame=current_segment_start,
                    end_frame=i
                ))
                # Start a still segment
                current_segment_start = i
                current_segment_type = 'still'
            elif current_segment_type is None:
                # Start from the beginning as still
                current_segment_type = 'still'
            # If already 'still', just continue
        else:
            # Frames differ
            if current_segment_type == 'still':
                # Close the current still segment
                segments.append(Segment(
                    segment_type='still',
                    start_frame=current_segment_start,
                    end_frame=i
                ))
                # Start an animation segment
                current_segment_start = i
                current_segment_type = 'animation'
            elif current_segment_type is None:
                # Start as animation
                current_segment_type = 'animation'
            # If already 'animation', just continue

    # Close the last segment
    last_frame_index = len(frames) - 1
    if current_segment_type is not None:
        segments.append(Segment(
            segment_type=current_segment_type,
            start_frame=current_segment_start,
            end_frame=last_frame_index
        ))

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

    # Initialize tqdm progress bar for cutting segments
    with tqdm(total=len(segments), desc="Cutting Segments", unit="segment") as pbar:
        for idx, seg in enumerate(segments):
            segment_type = seg['type']
            start_time = seg['start_time']
            duration = seg['duration']
            out_file = os.path.join(output_dir, f"segment_{idx:03d}_{segment_type}.mp4")

            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output files without asking
                "-i", video_path,
                "-ss", f"{start_time:.3f}",
                "-t", f"{duration:.3f}",
                "-c", "copy",
                out_file
            ]

            # Optional: You can comment out the next line if you don't want to see the ffmpeg commands
            # print(f"Running command: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Optionally, you can print the progress
                # print(f"Segment saved to: {out_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing segment {idx}: {e.stderr.decode()}")
                print("Skipping this segment.")
            finally:
                pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Segment video into still and animation parts with progress bars.")
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

    print("Loading all frames into memory...")
    try:
        frames = load_all_frames(video_path, total_frames)
    except Exception as e:
        print(f"Error loading frames: {e}")
        return

    print("Segmenting video based on frame differences...")
    try:
        segments = segment_video(frames)
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
