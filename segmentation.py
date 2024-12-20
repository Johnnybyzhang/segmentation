import os
import math
import cv2
import subprocess
import argparse
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

@dataclass
class Segment:
    segment_type: str  # 'still' or 'animation'
    start_frame: int
    end_frame: int

def get_video_properties(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Failed to retrieve FPS from the video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames

def compare_two_frames(frame1, frame2):
    """Compare two frames for exact equality."""
    return (frame1 == frame2).all()

def process_chunk(frames, prev_frame, prev_frame_index, start_frame_index, executor, workers):
    """
    Process a single chunk of frames in parallel:
    - frames: List of frames read in this chunk
    - prev_frame: The last frame from the previous chunk (None if first chunk)
    - prev_frame_index: The frame index of prev_frame in the global sequence
    - start_frame_index: The index of the first frame in this chunk
    - executor: A ProcessPoolExecutor for parallel comparisons
    - workers: number of parallel workers (for informational prints)

    Returns:
    - results: List of booleans for each comparison in this chunk
    - last_frame_index: The index of the last frame in this chunk
    """
    if prev_frame is not None:
        combined_frames = [prev_frame] + frames
    else:
        combined_frames = frames

    # Prepare frame pairs (each will be processed in parallel)
    frame_pairs = [(combined_frames[i], combined_frames[i+1]) for i in range(len(combined_frames)-1)]
    n_pairs = len(frame_pairs)

    # Submit tasks to the executor
    futures = []
    with tqdm(total=n_pairs, desc="Comparing chunk frames", unit="comparison", leave=False) as pbar:
        for f1, f2 in frame_pairs:
            future = executor.submit(compare_two_frames, f1, f2)
            futures.append(future)

        results = []
        # As tasks complete, update the progress bar
        for f in as_completed(futures):
            res = f.result()
            results.append(res)
            pbar.update(1)

    last_frame_index = start_frame_index + len(frames) - 1
    return results, last_frame_index

def update_segments(segments: List[Segment], results: List[bool], prev_frame_index: Optional[int], start_frame_index: int,
                    current_segment_type: Optional[str], current_segment_start: Optional[int]):
    """
    Update segments based on the results of a chunk.
    """
    for i, identical in enumerate(results):
        if prev_frame_index is not None:
            # i=0: compares prev_frame_index and start_frame_index
            # i>0: compares start_frame_index+(i-1) and start_frame_index+i
            if i == 0:
                f1 = prev_frame_index
                f2 = start_frame_index
            else:
                f1 = start_frame_index + (i - 1)
                f2 = start_frame_index + i
        else:
            # If no prev_frame_index:
            # results[i] compares (start_frame_index+i) and (start_frame_index+i+1)
            f1 = start_frame_index + i
            f2 = start_frame_index + i + 1

        if identical:
            # frames f1 and f2 are identical
            if current_segment_type == 'animation':
                # Close the animation segment at f1
                segments.append(Segment(segment_type='animation', start_frame=current_segment_start, end_frame=f1))
                current_segment_start = f1
                current_segment_type = 'still'
            elif current_segment_type is None:
                current_segment_type = 'still'
        else:
            # frames differ
            if current_segment_type == 'still':
                # Close the still segment at f1
                segments.append(Segment(segment_type='still', start_frame=current_segment_start, end_frame=f1))
                current_segment_start = f1
                current_segment_type = 'animation'
            elif current_segment_type is None:
                current_segment_type = 'animation'

    return segments, current_segment_type, current_segment_start

def finalize_segments(segments: List[Segment], current_segment_type: Optional[str], current_segment_start: Optional[int], last_frame_index: int):
    """
    Close the last open segment after processing all chunks.
    """
    if current_segment_type is not None:
        segments.append(Segment(
            segment_type=current_segment_type,
            start_frame=current_segment_start,
            end_frame=last_frame_index
        ))
    return segments

def map_segments_to_timestamps(segments: List[Segment], fps: float):
    mapped_segments = []
    for seg in segments:
        start_time = seg.start_frame / fps
        duration = (seg.end_frame - seg.start_frame + 1) / fps
        mapped_segments.append({
            'type': seg.segment_type,
            'start_time': start_time,
            'duration': duration
        })
    return mapped_segments

def cut_segments_with_ffmpeg(video_path: str, segments: List[dict], output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with tqdm(total=len(segments), desc="Cutting Segments", unit="segment") as pbar:
        for idx, seg in enumerate(segments):
            segment_type = seg['type']
            start_time = seg['start_time']
            duration = seg['duration']
            out_file = os.path.join(output_dir, f"segment_{idx:03d}_{segment_type}.mp4")

            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-ss", f"{start_time:.3f}",
                "-t", f"{duration:.3f}",
                "-c", "copy",
                out_file
            ]

            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print(f"Error processing segment {idx}: {e.stderr.decode()}")
                print("Skipping this segment.")
            pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Segment video into still and animation parts with parallel processing.")
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("-o", "--output", default="cuts", help="Output directory for segments.")
    parser.add_argument("-c", "--chunk-size", type=int, default=512, help="Number of frames to process per chunk.")
    parser.add_argument("-w", "--workers", type=int, default=16, help="Number of parallel workers for frame comparison.")
    args = parser.parse_args()

    video_path = args.video
    output_dir = args.output
    chunk_size = args.chunk_size
    workers = args.workers

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

    if total_frames == 0:
        print("No frames found in the video.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return

    num_chunks = math.ceil(total_frames / chunk_size)
    segments = []
    current_segment_type = None
    current_segment_start = None

    prev_frame = None
    prev_frame_index = None
    frame_index = 0

    print(f"Using {workers} workers for parallel frame comparisons.")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for chunk_number in range(num_chunks):
            # Calculate chunk boundaries
            chunk_start_frame = frame_index
            chunk_end_frame = min(chunk_start_frame + chunk_size - 1, total_frames - 1)
            this_chunk_size = (chunk_end_frame - chunk_start_frame + 1)

            global_progress = (chunk_number + 1) / num_chunks * 100
            print(f"Processing chunk {chunk_number+1} out of {num_chunks}, total progress: {global_progress:.2f}%")

            # Load frames for this chunk
            frames = []
            with tqdm(total=this_chunk_size, desc="Loading chunk frames", unit="frame", leave=False) as pbar:
                for _ in range(this_chunk_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    pbar.update(1)

            if len(frames) == 0:
                # No more frames
                break

            # Process this chunk in parallel
            results, last_frame_index = process_chunk(frames, prev_frame, prev_frame_index, chunk_start_frame, executor, workers)

            # Update segments with these results
            segments, current_segment_type, current_segment_start = update_segments(
                segments,
                results,
                prev_frame_index,
                chunk_start_frame,
                current_segment_type,
                current_segment_start
            )

            # Prepare for next chunk
            prev_frame = frames[-1]
            prev_frame_index = last_frame_index
            frame_index = last_frame_index + 1

            # Release memory
            frames = None
            results = None

    cap.release()

    # Finalize segments
    segments = finalize_segments(segments, current_segment_type, current_segment_start, total_frames - 1)

    print(f"Identified {len(segments)} segments.\n")

    print("Mapping segments to timestamps...")
    mapped_segments = map_segments_to_timestamps(segments, fps)

    print("Cutting segments using ffmpeg...")
    cut_segments_with_ffmpeg(video_path, mapped_segments, output_dir)

    print("All segments have been processed and saved.")

if __name__ == "__main__":
    main()
