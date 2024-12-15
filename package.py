from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import h5py
import OpenEXR

from tqdm import tqdm

def get_argument_parser():
    parser = ArgumentParser(description="Package rendering resources as a single HDF5")
    parser.add_argument(
        'dir', type=str, 
        help="path to the directory containing different render resolutions")
    parser.add_argument(
        '-o', '--output', type=str, required=True, 
        help="path to the output HDF5 file")
    parser.add_argument(
        '-t', '--test-sequences', type=int, nargs='+', required=True, 
        help="which sequences (zero-based indices) to use for testing")
    parser.add_argument(
        '-a', '--append', action='store_true',
        help="append all missing resolutions found in dir to the dataset found at output")
    parser.add_argument(
        '-w', '--overwrite', action='store_true',
        help="overwrite the output file if it exists")
    parser.add_argument(
        '-fps', '--frames-per-sequence', type=int, default=100,
        help="number of frames per sequence")
    parser.add_argument(
        '-fd', '--frame-index-digits', type=int, default=3,
        help="number of digits in the frame name")
    parser.add_argument(
        '-sd', '--sequence-index-digits', type=int, default=3,
        help="number of digits in the sequence name")
    return parser

def peak_frame_count(subdir: Path) -> int:
    return len([file for file in subdir.glob("*.exr")])

def load_exr_layer(file, name, width, height, with_alpha=False, use_xyzw=False):
    r = file.channel(f'ViewLayer.{name}.R' if not use_xyzw else f'ViewLayer.{name}.X')
    g = file.channel(f'ViewLayer.{name}.G' if not use_xyzw else f'ViewLayer.{name}.Y')
    b = file.channel(f'ViewLayer.{name}.B' if not use_xyzw else f'ViewLayer.{name}.Z')
    
    r_array = np.frombuffer(r, dtype=np.float32).reshape(height, width)
    g_array = np.frombuffer(g, dtype=np.float32).reshape(height, width)
    b_array = np.frombuffer(b, dtype=np.float32).reshape(height, width)
    channel_arrays = [r_array, g_array, b_array]

    if with_alpha:
        a = file.channel(f'ViewLayer.{name}.A' if not use_xyzw else f'ViewLayer.{name}.W')
        a_array = np.frombuffer(a, dtype=np.float32).reshape(height, width)
        channel_arrays.append(a_array)

    return np.stack(channel_arrays, axis=0)

def load_exr_layer_single(file, name, width, height):
    x = file.channel(f'ViewLayer.{name}')
    return np.frombuffer(x, dtype=np.float32).reshape(height, width)

def get_image_dims(header):
    width  = header['dataWindow'].max.x - header['dataWindow'].min.x + 1
    height = header['dataWindow'].max.y - header['dataWindow'].min.y + 1
    return width, height

if __name__ == "__main__":
    args = get_argument_parser().parse_args()

    input_dir = Path(args.dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"[!] Directory {input_dir} does not exist")
    
    output_file = Path(args.output)
    existing_resolutions = []
    if output_file.exists() and not args.append and not args.overwrite:
        print(f"[!] Output file {output_file} already exists. If you want to append more resolutions to it, run the script again with the -a flag. If you want to force overwriting the file, run the script again with the -w flag")
        exit(1)
    elif output_file.exists() and not args.overwrite:
        with h5py.File(output_file, 'r') as f:
            existing_resolutions = list(f.keys())
        print(f"[>] Found the following resolutions in {output_file}: {existing_resolutions}")
    
    subdirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name not in existing_resolutions]
    print(f"[>] Packaging the following resolutions: {[d.name for d in subdirs]}")

    n_frames = peak_frame_count(subdirs[0])
    print(f"[>] Found {n_frames} frames for resolution {subdirs[0].name}")
    for subdir in subdirs[1:]:
        if peak_frame_count(subdir) != n_frames:
            raise ValueError(f"[!] Detected inconsistent number of frames for resolution {subdir.name}")
        print(f"[>] Found {n_frames} frames for resolution {subdir.name}")

    n_sequences = n_frames // args.frames_per_sequence
    if n_sequences <= 0:
        raise ValueError(f"[!] Not enough frames to assemble sequences with {args.frames_per_sequence} frames")
    print(f"[>] Using {args.frames_per_sequence} frames per sequence, assembling {n_sequences} sequences for each resolution")
    print(f"[>] Using sequences {args.test_sequences} for testing")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, 'a' if args.append else 'w') as f:
        f.attrs['total-frames'] = n_frames
        f.attrs['frames-per-sequence'] = args.frames_per_sequence
        f.attrs['sequence-index-digits'] = args.sequence_index_digits
        f.attrs['frame-index-digits'] = args.frame_index_digits
        f.attrs['test-sequences'] = len(args.test_sequences)
        f.attrs['train-sequences'] = n_sequences - len(args.test_sequences)
        
        for subdir in subdirs:
            print(f"[>] Packaging resolution {subdir.name}")
            resolution_group = f.create_group(subdir.name)
            frame_dimensions = None

            train_group = resolution_group.create_group("train")
            test_group  = resolution_group.create_group("test")

            train_seq_index = 0
            test_seq_index  = 0
            
            for sequence in range(n_sequences):
                if sequence in args.test_sequences:
                    print(f"[>] Assembling sequence {sequence} | Test")
                    seq_group = test_group.create_group(f"seq-{test_seq_index:0{args.sequence_index_digits}d}")
                    test_seq_index += 1
                else:
                    print(f"[>] Assembling sequence {sequence} | Train")
                    seq_group = train_group.create_group(f"seq-{train_seq_index:0{args.sequence_index_digits}d}")
                    train_seq_index += 1

                progress_bar = tqdm(range(args.frames_per_sequence))
                for frame in progress_bar:
                    frame_group = seq_group.create_group(f"frame-{frame:0{args.frame_index_digits}d}")
                    frame_file = subdir / f"frame-{frame + args.frames_per_sequence * sequence + 1:04d}.exr"

                    progress_bar.set_description(f"{subdir.name} | {frame_file.name}")

                    exr = OpenEXR.InputFile(str(frame_file))
                    if frame_dimensions is None:
                        frame_dimensions = get_image_dims(exr.header())
                    elif frame_dimensions != get_image_dims(exr.header()):
                        progress_bar.write(f"[!] Detected inconsistent frame dimensions in {frame_file.name}")

                    width, height = frame_dimensions
                    frame_group.create_dataset('combined', data=load_exr_layer(exr, 'Combined', width, height, with_alpha=True), dtype=np.float32)
                    frame_group.create_dataset('normal', data=load_exr_layer(exr, 'Normal', width, height, use_xyzw=True), dtype=np.float32)
                    frame_group.create_dataset('vector', data=load_exr_layer(exr, 'Vector', width, height, use_xyzw=True, with_alpha=True), dtype=np.float32)
                    frame_group.create_dataset('depth', data=load_exr_layer_single(exr, 'Mist.Z', width, height), dtype=np.float32)
                    frame_group.create_dataset('diffuse-col', data=load_exr_layer(exr, 'DiffCol', width, height), dtype=np.float32)
                    frame_group.create_dataset('diffuse-dir', data=load_exr_layer(exr, 'DiffDir', width, height), dtype=np.float32)
                    frame_group.create_dataset('diffuse-ind', data=load_exr_layer(exr, 'DiffInd', width, height), dtype=np.float32)
                    frame_group.create_dataset('glossy-col', data=load_exr_layer(exr, 'GlossCol', width, height), dtype=np.float32)
                    frame_group.create_dataset('glossy-dir', data=load_exr_layer(exr, 'GlossDir', width, height), dtype=np.float32)
                    frame_group.create_dataset('glossy-ind', data=load_exr_layer(exr, 'GlossInd', width, height), dtype=np.float32)
                    frame_group.create_dataset('emission', data=load_exr_layer(exr, 'Emit', width, height), dtype=np.float32)
                    frame_group.create_dataset('environment', data=load_exr_layer(exr, 'Env', width, height), dtype=np.float32)
                    frame_group.create_dataset('roughness', data=load_exr_layer_single(exr, 'Roughness.X', width, height), dtype=np.float32)
        
            resolution_group.attrs['frame-width']  = frame_dimensions[0]
            resolution_group.attrs['frame-height'] = frame_dimensions[1]

