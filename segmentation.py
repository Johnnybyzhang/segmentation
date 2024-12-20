import os
import cv2
import subprocess
import argparse
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor
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

def compare_two_frames(args):
    """Compare two frames (frame1, frame2) for exact equality."""
    frame1, frame2 = args
    return (frame1 == frame2).all()

def process_chunk(frames, prev_frame, prev_frame_index, start_frame_index, executor):
    """
    Process a single chunk of frames:
    - frames: List of frames read in this chunk
    - prev_frame: The last frame from the previous chunk (None if this is the first chunk)
    - prev_frame_index: The frame index of prev_frame in the global sequence
    - start_frame_index: The index of the first frame in this chunk
    - executor: A ProcessPoolExecutor for parallel comparisons

    Returns:
    - results: List of booleans for each comparison in this chunk (including prev_frame if provided)
    - last_frame_index: The index of the last frame in this chunk
    """
    if prev_frame is not None:
        combined_frames = [prev_frame] + frames
    else:
        combined_frames = frames

    # Prepare frame pairs
    frame_pairs = [(combined_frames[i], combined_frames[i+1]) for i in range(len(combined_frames)-1)]

    # Parallel comparison
    results = list(executor.map(compare_two_frames, frame_pairs))

    last_frame_index = start_frame_index + len(frames) - 1
    return results, last_frame_index

def update_segments(segments: List[Segment], results: List[bool], prev_frame_index: Optional[int], start_frame_index: int,
                    current_segment_type: Optional[str], current_segment_start: Optional[int]):
    """
    Update segments based on the results of a chunk.
    results: boolean list of comparisons
    prev_frame_index: The index of the frame before the first frame in this chunk (None if none)
    start_frame_index: The index of the first frame in this chunk
    current_segment_type, current_segment_start: segment state carried over from previous chunks

    Returns updated segments, current_segment_type, current_segment_start.
    """
    # The indexing of frames for results:
    # If prev_frame_index is not None:
    #   results[0] compares prev_frame_index and start_frame_index
    #   results[1] compares start_frame_index and start_frame_index+1
    #   ...
    # If prev_frame_index is None (first chunk):
    #   results[0] compares start_frame_index and start_frame_index+1
    #   results[i] compares (start_frame_index+i) and (start_frame_index+i+1)

    for i, identical in enumerate(results):
        if prev_frame_index is not None:
            # The pair of frames for results[i]:
            # difference i=0: (prev_frame_index, start_frame_index)
            # difference i>0: (start_frame_index+(i-1), start_frame_index+i)
            if i == 0:
                f1 = prev_frame_index
                f2 = start_frame_index
            else:
                f1 = start_frame_index + (i - 1)
                f2 = start_frame_index + i
        else:
            # No prev frame:
            # results[i]: (start_frame_index+i, start_frame_index+i+1)
            f1 = start_frame_index + i
            f2 = start_frame_index + i + 1

        # Update segments logic
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
    parser = argparse.ArgumentParser(description="Segment video into still and animation parts in chunks.")
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("-o", "--output", default="cuts", help="Output directory for segments.")
    parser.add_argument("-c", "--chunk-size", type=int, default=512, help="Number of frames to process per chunk.")
    args = parser.parse_args()

    video_path = args.video
    output_dir = args.output
    chunk_size = args.chunk_size

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

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return

    total_comparisons = max(0, total_frames - 1)  # number of frame-to-frame comparisons
    segments = []
    current_segment_type = None
    current_segment_start = None

    prev_frame = None
    prev_frame_index = None

    frame_index = 0
    with ProcessPoolExecutor() as executor:
        with tqdm(total=total_comparisons, desc="Comparing Frames", unit="comparison") as pbar:
            while True:
                # Read up to chunk_size frames
                frames = []
                for _ in range(chunk_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)

                if len(frames) == 0:
                    # No more frames
                    break

                # Process this chunk
                chunk_start_frame = frame_index
                results, last_frame_index = process_chunk(frames, prev_frame, prev_frame_index, chunk_start_frame, executor)

                # Update segments with these results
                segments, current_segment_type, current_segment_start = update_segments(
                    segments, 
                    results, 
                    prev_frame_index, 
                    chunk_start_frame,
                    current_segment_type, 
                    current_segment_start
                )

                # Update progress bar
                pbar.update(len(results))

                # Prepare for next chunk
                prev_frame = frames[-1]
                prev_frame_index = last_frame_index
                frame_index = last_frame_index + 1

    cap.release()

    # Finalize segments
    if total_frames > 0:
        segments = finalize_segments(segments, current_segment_type, current_segment_start, total_frames - 1)

    print(f"Identified {len(segments)} segments.\n")

    print("Mapping segments to timestamps...")
    mapped_segments = map_segments_to_timestamps(segments, fps)

    print("Cutting segments using ffmpeg...")
    cut_segments_with_ffmpeg(video_path, mapped_segments, output_dir)

    print("All segments have been processed and saved.")

if __name__ == "__main__":
    main()
