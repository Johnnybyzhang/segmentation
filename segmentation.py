import os
import math
import cv2
import subprocess
import argparse
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np


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


def frames_are_similar(frame1, frame2, pixel_threshold=0.0001, per_pixel_diff_threshold=30):
    """
    Compare two frames and determine if they are similar within a given threshold.

    Args:
        frame1: First frame as a NumPy array.
        frame2: Second frame as a NumPy array.
        pixel_threshold: Maximum allowable fraction of differing pixels (e.g., 0.0001 for 0.01%).
        per_pixel_diff_threshold: The per-pixel intensity difference to consider a pixel as "different".

    Returns:
        True if frames are similar within the threshold, False otherwise.
    """
    # Compute absolute difference per pixel
    diff = cv2.absdiff(frame1, frame2)
    # Convert to grayscale to simplify difference computation
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Threshold the difference to identify significant changes
    _, thresh = cv2.threshold(gray_diff, per_pixel_diff_threshold, 255, cv2.THRESH_BINARY)
    # Count the number of differing pixels
    num_diff_pixels = cv2.countNonZero(thresh)
    # Total number of pixels
    total_pixels = gray_diff.shape[0] * gray_diff.shape[1]
    # Calculate the fraction of differing pixels
    fraction_diff = num_diff_pixels / total_pixels
    return fraction_diff <= pixel_threshold


def compare_frame_pairs_subchunk(frame_pairs_sublist, pixel_threshold, per_pixel_diff_threshold):
    """Compare a sublist of frame pairs and return the list of boolean results."""
    return [
        frames_are_similar(f1, f2, pixel_threshold, per_pixel_diff_threshold)
        for (f1, f2) in frame_pairs_sublist
    ]


def split_into_subchunks(data, n_subchunks):
    """Split the data list into n_subchunks as evenly as possible."""
    length = len(data)
    if n_subchunks <= 0 or n_subchunks > length:
        # If n_subchunks is larger than data length, just return one chunk
        return [data]
    chunk_size = math.ceil(length / n_subchunks)
    return [data[i:i + chunk_size] for i in range(0, length, chunk_size)]


def process_chunk(
    frames,
    prev_frame,
    prev_frame_index,
    start_frame_index,
    executor,
    workers,
    pixel_threshold,
    per_pixel_diff_threshold,
):
    """
    Process a single chunk of frames in parallel.

    Args:
        frames: List of frames read in this chunk.
        prev_frame: The last frame from the previous chunk (None if first chunk).
        prev_frame_index: The frame index of prev_frame in the global sequence.
        start_frame_index: The index of the first frame in this chunk.
        executor: A ProcessPoolExecutor for parallel comparisons.
        workers: Number of parallel workers.
        pixel_threshold: Maximum allowable fraction of differing pixels.
        per_pixel_diff_threshold: Per-pixel intensity difference threshold.

    Returns:
        results: List of booleans indicating similarity for each frame comparison in this chunk.
        last_frame_index: The index of the last frame in this chunk.
    """
    if prev_frame is not None:
        combined_frames = [prev_frame] + frames
    else:
        combined_frames = frames

    # Prepare frame pairs
    frame_pairs = [
        (combined_frames[i], combined_frames[i + 1]) for i in range(len(combined_frames) - 1)
    ]
    n_pairs = len(frame_pairs)

    # Divide frame_pairs into sub-chunks for parallel processing
    subchunks = split_into_subchunks(frame_pairs, workers)
    n_subchunks = len(subchunks)

    # Submit subchunks to the executor with their indices to maintain order
    futures = {
        executor.submit(
            compare_frame_pairs_subchunk, sc, pixel_threshold, per_pixel_diff_threshold
        ): idx
        for idx, sc in enumerate(subchunks)
    }

    # Initialize results_ordered with placeholders
    results_ordered = [None] * n_subchunks

    # Progress bar for this chunk's comparisons
    with tqdm(
        total=n_pairs, desc="Comparing chunk frames", unit="comparison", leave=False
    ) as pbar:
        for future in as_completed(futures):
            idx = futures[future]
            try:
                sub_results = future.result()
            except Exception as e:
                print(f"Error in parallel comparison: {e}")
                sub_results = [False] * len(subchunks[idx])  # Treat errors as non-similar
            results_ordered[idx] = sub_results
            pbar.update(len(sub_results))

    # Check if any subchunk results are missing
    for idx, sublist in enumerate(results_ordered):
        if sublist is None:
            print(f"Warning: Subchunk {idx} did not return any results. Treating as all differing.")
            results_ordered[idx] = [False] * len(subchunks[idx])

    # Flatten results in the correct order
    results = [res for sublist in results_ordered for res in sublist]

    last_frame_index = start_frame_index + len(frames) - 1
    return results, last_frame_index


def update_segments(
    segments: List[Segment],
    results: List[bool],
    prev_frame_index: Optional[int],
    start_frame_index: int,
    current_segment_type: Optional[str],
    current_segment_start: Optional[int],
):
    """
    Update segments based on the results of a chunk.

    Args:
        segments: List of identified segments.
        results: List of booleans indicating similarity for each frame comparison.
        prev_frame_index: The frame index before the current chunk (None if first chunk).
        start_frame_index: The index of the first frame in this chunk.
        current_segment_type: Type of the current ongoing segment ('still' or 'animation').
        current_segment_start: Starting frame index of the current ongoing segment.

    Returns:
        Updated segments list, current_segment_type, current_segment_start.
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
            # i=0: compares frame 0 and frame 1
            # i=1: compares frame 1 and frame 2
            f1 = start_frame_index + i
            f2 = start_frame_index + i + 1

        if identical:
            # frames f1 and f2 are similar
            if current_segment_type == 'animation':
                # Close the animation segment at f1
                segments.append(Segment('animation', current_segment_start, f1))
                # Start a new still segment
                current_segment_start = f1
                current_segment_type = 'still'
            elif current_segment_type is None:
                # Starting a still segment
                current_segment_type = 'still'
                current_segment_start = f1
            # If already in 'still', continue
        else:
            # frames f1 and f2 are different
            if current_segment_type == 'still':
                # Close the still segment at f1
                segments.append(Segment('still', current_segment_start, f1))
                # Start a new animation segment
                current_segment_start = f1
                current_segment_type = 'animation'
            elif current_segment_type is None:
                # Starting an animation segment
                current_segment_type = 'animation'
                current_segment_start = f1
            # If already in 'animation', continue

    return segments, current_segment_type, current_segment_start


def finalize_segments(
    segments: List[Segment],
    current_segment_type: Optional[str],
    current_segment_start: Optional[int],
    last_frame_index: int,
):
    """
    Close the last open segment after processing all chunks.

    Args:
        segments: List of identified segments.
        current_segment_type: Type of the current ongoing segment ('still' or 'animation').
        current_segment_start: Starting frame index of the current ongoing segment.
        last_frame_index: The index of the last frame in the video.

    Returns:
        Finalized segments list.
    """
    if current_segment_type is not None and current_segment_start is not None:
        segments.append(Segment(
            segment_type=current_segment_type,
            start_frame=current_segment_start,
            end_frame=last_frame_index
        ))
    return segments


def purge_short_animation_segments(segments: List[Segment]) -> List[Segment]:
    """
    Remove animation segments that are exactly 2 frames long by reassigning their frames
    to the previous and next 'still' segments respectively, ensuring 'still' segments remain separate.

    Args:
        segments: List of initial segments.

    Returns:
        Updated list of segments with short animation segments purged.
    """
    new_segments = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg.segment_type == 'animation' and (seg.end_frame - seg.start_frame + 1) == 2:
            f1 = seg.start_frame
            f2 = seg.end_frame

            # Assign f1 to the previous 'still' segment if it exists
            if new_segments and new_segments[-1].segment_type == 'still':
                new_segments[-1].end_frame = f1
            else:
                # If no previous 'still' segment, create a new one for f1
                new_segments.append(Segment('still', f1, f1))

            # Assign f2 to the next 'still' segment
            if (i + 1) < len(segments) and segments[i + 1].segment_type == 'still':
                next_seg = segments[i + 1]
                # Modify the next 'still' segment's start_frame to f2
                modified_next_seg = Segment('still', f2, next_seg.end_frame)
                new_segments.append(modified_next_seg)
                i += 1  # Skip the next segment as it's handled
            else:
                # If no next 'still' segment, create a new one for f2
                new_segments.append(Segment('still', f2, f2))
        else:
            # Keep the segment as is
            new_segments.append(seg)
        i += 1  # Only increment once per loop iteration
    return new_segments


def verify_frame_coverage(segments: List[Segment], total_frames: int):
    """
    Verify that all frames in the video are covered by the segments without gaps or overlaps.

    Args:
        segments: List of finalized segments.
        total_frames: Total number of frames in the video.

    Raises:
        ValueError: If there are missing or overlapping frames.
    """
    covered_frames = set()
    for seg in segments:
        for frame in range(seg.start_frame, seg.end_frame + 1):
            if frame in covered_frames:
                raise ValueError(f"Overlapping frames detected at frame {frame}.")
            covered_frames.add(frame)

    missing_frames = set(range(total_frames)) - covered_frames
    if missing_frames:
        missing_frames_sorted = sorted(missing_frames)
        print(f"Missing frames: {missing_frames_sorted}")
    else:
        print("All frames are covered in the segments.")


def map_segments_to_timestamps(segments: List[Segment], fps: float):
    """
    Convert frame indices to timestamps based on framerate.

    Args:
        segments: List of identified segments.
        fps: Frames per second of the video.

    Returns:
        List of dictionaries with segment type, start time, and duration.
    """
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
    """
    Use ffmpeg to cut the video into segments without re-encoding.

    Args:
        video_path: Path to the input video.
        segments: List of segments with type, start_time, and duration.
        output_dir: Directory to save the output segments.
    """
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
                "-y",  # Overwrite output files without asking
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
    parser = argparse.ArgumentParser(
        description="Segment video into still and animation parts with parallel processing and similarity threshold."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("-o", "--output", default="cuts", help="Output directory for segments.")
    parser.add_argument(
        "-c", "--chunk-size", type=int, default=512, help="Number of frames to process per chunk."
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers (processes) for frame comparison.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.0001,
        help="Similarity threshold as a fraction (e.g., 0.0001 for 0.01%%).",
    )
    parser.add_argument(
        "-p",
        "--per-pixel-diff",
        type=int,
        default=30,
        help="Per-pixel intensity difference threshold.",
    )
    args = parser.parse_args()

    video_path = args.video
    output_dir = args.output
    chunk_size = args.chunk_size
    workers = args.workers
    pixel_threshold = args.threshold
    per_pixel_diff_threshold = args.per_pixel_diff

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
    print(f"Similarity threshold: {pixel_threshold*100:.4f}% differing pixels allowed.")
    print(f"Per-pixel difference threshold: {per_pixel_diff_threshold}")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for chunk_number in range(num_chunks):
            chunk_start_frame = frame_index
            chunk_end_frame = min(chunk_start_frame + chunk_size - 1, total_frames - 1)
            this_chunk_size = (chunk_end_frame - chunk_start_frame + 1)

            global_progress = (chunk_number + 1) / num_chunks * 100
            print(
                f"Processing chunk {chunk_number+1} out of {num_chunks}, total progress: {global_progress:.2f}%"
            )

            # Load frames for this chunk with a progress bar
            frames = []
            with tqdm(
                total=this_chunk_size, desc="Loading chunk frames", unit="frame", leave=False
            ) as pbar:
                for _ in range(this_chunk_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    pbar.update(1)

            if len(frames) == 0:
                break

            # Process this chunk
            results, last_frame_index = process_chunk(
                frames,
                prev_frame,
                prev_frame_index,
                chunk_start_frame,
                executor,
                workers,
                pixel_threshold,
                per_pixel_diff_threshold,
            )

            # Update segments with these results
            segments, current_segment_type, current_segment_start = update_segments(
                segments,
                results,
                prev_frame_index,
                chunk_start_frame,
                current_segment_type,
                current_segment_start,
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
    segments = finalize_segments(
        segments, current_segment_type, current_segment_start, total_frames - 1
    )

    # Purge short animation segments (exactly 2 frames)
    segments = purge_short_animation_segments(segments)

    print(f"Identified {len(segments)} segments.\n")

    # Verify frame coverage
    try:
        verify_frame_coverage(segments, total_frames)
    except ValueError as ve:
        print(f"Frame coverage verification failed: {ve}")
        return

    # Optional: Print segments for verification
    print("Final Segments:")
    for idx, seg in enumerate(segments):
        print(f"Segment {idx+1}: Type={seg.segment_type}, Start Frame={seg.start_frame}, End Frame={seg.end_frame}")

    print("\nMapping segments to timestamps...")
    mapped_segments = map_segments_to_timestamps(segments, fps)

    print("Cutting segments using ffmpeg...")
    cut_segments_with_ffmpeg(video_path, mapped_segments, output_dir)

    print("All segments have been processed and saved.")


if __name__ == "__main__":
    main()
