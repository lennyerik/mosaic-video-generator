#!/usr/bin/env python3

from os import listdir, path
from math import sqrt, ceil
from time import time
from fractions import Fraction
import PyNvCodec as nvc
from pycuda.gpuarray import GPUArray
import pycuda
import argparse
from tqdm import tqdm
import numpy as np
import av


class DictArgumentAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(DictArgumentAction, self).__init__(*args, **kwargs)
        self.value_dict = self.default.copy()

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            for value in values:
                key, val = map(lambda x: x.strip(), value.split('='))
                self.value_dict[key] = val
        except ValueError:
            raise argparse.ArgumentError(
                self, f'Invalid value: "{value}". Must be in the format key=value')
        setattr(namespace, self.dest, self.value_dict)


parser = argparse.ArgumentParser(
    description='An application to convert multiple videos into one mosaic using Nvidia hardware acceleration',
    epilog='Available encoder arguments:\n' +
    '\n'.join([f"  - {argname:15} {desc}" for argname,
              desc in nvc.GetNvencParams().items()]),
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-ow', '--output-width', type=int, metavar='WIDTH',
                    help='Width of the generated video. Defaults to the width of the first input file')
parser.add_argument('-oh', '--output-height', type=int, metavar='HEIGHT',
                    help='Height of the generated video. Defaults to the height of the first input file')
parser.add_argument('--enable-non-divisable', action='store_true',
                    help='Enable resolutions which do not divide by the mosaic sizes evenly. This will cut off some pixels at the edges.')
parser.add_argument('-o', '--output', default='mosaic.mp4', metavar='FILENAME',
                    help='output filename')
parser.add_argument('-f', '--force', action='store_true',
                    help='Force override the output file')
parser.add_argument('-q', '--quiet', action='store_true',
                    help='Only display error messages')
parser.add_argument('-g', '--gpu-id', default=0, type=int,
                    help='GPU id to use for transcoding')
parser.add_argument('-ea', '--encoder-arg', action=DictArgumentAction, metavar='ARGS', nargs='+',
                    default={'preset': 'P4', 'codec': 'h264', 'gop': '30'}, help='Pass arguments on to the nvidia encoder, e.g. -ea preset=P4 codec=h264')
parser.add_argument(
    'VIDEO_FILES', help='The folder containing the video files to mosaic')
args = parser.parse_args()

# Create the decoders
decoders = []
mappers = []
ifmt, ifps = None, None
for f in listdir(args.VIDEO_FILES):
    full_path = path.join(args.VIDEO_FILES, f)
    if path.isfile(full_path):
        if not args.quiet:
            print(f'loading {f}...')

        decoder = nvc.PyNvDecoder(full_path, args.gpu_id)
        decoders.append(decoder)

        # Set the output width height and fps to the input's if they're not specified
        if args.output_width is None:
            args.output_width = decoder.Width()
        if args.output_height is None:
            args.output_height = decoder.Height()

        if ifmt is None and ifps is None:
            ifmt = decoder.Format()
            ifps = decoder.Framerate()

        assert ifmt == decoder.Format() and ifps == decoder.Framerate(
        ), 'All mosaic files must have the same pixel format and frame rate'

if len(decoders) == 0:
    print(f'No files found in {args.VIDEO_FILES}. Not rendering mosaic')
    exit(0)

# Calculate the grid sizes for the mosaic
mosaics_per_dimension = ceil(sqrt(len(decoders)))
mosaic_width = args.output_width / mosaics_per_dimension
mosaic_height = args.output_height / mosaics_per_dimension
if mosaic_width != int(mosaic_width) or mosaic_height != int(mosaic_height):
    if args.enable_non_divisable:
        mosaic_width = ceil(mosaic_width)
        mosaic_height = ceil(mosaic_height)
    else:
        print(f'Output dimensions {args.output_width}x{args.output_height} are not evenly divisable '
              f'by {mosaics_per_dimension} rows and columns!')
        print('Rerun with --enable-non-divisable to generate the video anyway.')
        print('Be aware that this will cut off some amount of pixels at the edges of certain clips.')
        exit(1)
else:
    mosaic_width = int(mosaic_width)
    mosaic_height = int(mosaic_height)

# Create the rgb and yuv converter and the resizer
to_rgb = nvc.PySurfaceConverter(
    mosaic_width, mosaic_height, nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, args.gpu_id
)
colour_space = nvc.ColorspaceConversionContext(
    nvc.ColorSpace.BT_601, nvc.ColorRange.JPEG)
to_yuv = nvc.PySurfaceConverter(
    args.output_width, args.output_height, nvc.PixelFormat.RGB, nvc.PixelFormat.YUV444, args.gpu_id
)
mosaic_resizer = nvc.PySurfaceResizer(
    mosaic_width, mosaic_height, ifmt, args.gpu_id)

# Create the encoder and output file
encoder = nvc.PyNvEncoder(args.encoder_arg | {
    's': f'{args.output_width}x{args.output_height}'}, args.gpu_id, nvc.PixelFormat.YUV444)

if not args.force and path.isfile(args.output):
    if not input(f'Outfile {args.output} already exists. Overwrite? [y/N] ').lower().startswith('y'):
        exit(0)
output_file = av.open(args.output, 'w')
output_stream = output_file.add_stream(args.encoder_arg['codec'], rate=ifps)
output_stream.width = args.output_width
output_stream.height = args.output_height
output_stream.pix_fmt = 'yuv444p'
output_stream.time_base = Fraction(1, int(ifps * 1000))
output_stream.gop_size = int(args.encoder_arg['gop'])


def write_packet(idx, data):
    packet = av.packet.Packet(bytes(data))
    packet.dts = packet.pts = idx * 1000
    packet.time_base = Fraction(1, int(ifps * 1000))
    packet.stream = output_stream
    output_file.mux(packet)


def encode_single_frame(frame_idx):
    frame_surf = nvc.Surface.Make(
        nvc.PixelFormat.RGB, args.output_width, args.output_height, args.gpu_id)
    frame_plane = frame_surf.PlanePtr()
    frame_buf = GPUArray((args.output_height, args.output_width, 3),
                         dtype=np.uint8, gpudata=frame_plane.GpuMem())
    frame_buf.strides = (frame_plane.Pitch(), 3, 1)

    encoded_packet = np.empty([], dtype=np.uint8)

    for i, decoder in enumerate(decoders):
        surf = decoder.DecodeSingleSurface()

        surf = mosaic_resizer.Execute(surf)
        surf = to_rgb.Execute(surf, colour_space)

        surf_plane = surf.PlanePtr()
        surf_buf = GPUArray((mosaic_height, mosaic_width, 3),
                            dtype=np.uint8, gpudata=surf_plane.GpuMem())
        surf_buf.strides = (surf_plane.Pitch(), 3, 1)

        row = i // mosaics_per_dimension
        col = i % mosaics_per_dimension

        # Cut off overlapping edge pixels
        if args.enable_non_divisable:
            if row == (mosaics_per_dimension - 1):
                surf_buf = surf_buf[mosaics_per_dimension *
                                    mosaic_height-args.output_height:]
            if col == (mosaics_per_dimension - 1):
                surf_buf = surf_buf[:, mosaics_per_dimension *
                                    mosaic_width-args.output_width:]

        # This should happen entirely on the GPU and be lightning fast
        frame_buf[row*mosaic_height:row*mosaic_height+mosaic_height,
                  col*mosaic_width:col*mosaic_width+mosaic_width] = surf_buf

    # Go back to YUV for compression
    frame_surf = to_yuv.Execute(frame_surf, colour_space)

    if encoder.EncodeSingleSurface(frame_surf, encoded_packet, sync=False):
        write_packet(frame_idx, encoded_packet)


# Main loop
frames = []
start = time()

frame_iter = range(min(map(lambda x: x.Numframes(), decoders)))
if not args.quiet:
    frame_iter = tqdm(frame_iter)

frame_idx = 0
for frame_idx in frame_iter:
    encode_single_frame(frame_idx)

# Flush the encoder queue
encoded_packet = np.empty([], dtype=np.uint8)
while encoder.FlushSinglePacket(encoded_packet):
    frame_idx += 1
    write_packet(frame_idx, encoded_packet)

if not args.quiet:
    print(f'Finished generating in {time() - start:.3f} seconds')
