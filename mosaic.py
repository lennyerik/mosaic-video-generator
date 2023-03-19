#!/usr/bin/env python3

from fractions import Fraction
from os import listdir, path
from math import sqrt, ceil
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
                    help='Width of the generated video. Defaults to the width of the input files')
parser.add_argument('-oh', '--output-height', type=int, metavar='HEIGHT',
                    help='Height of the generated video. Defaults to the height of the input files')
parser.add_argument('-o', '--output', default='mosaic.mp4', metavar='filename',
                    help='output filename')
parser.add_argument('-f', '--force', action='store_true',
                    help='Force override the output file')
parser.add_argument('-q', '--quiet', action='store_true',
                    help='Only display error messages')
parser.add_argument('-g', '--gpu-id', default=0, type=int,
                    help='GPU id to use for transcoding')
parser.add_argument('-ea', '--encoder-arg', action=DictArgumentAction, metavar='args', nargs='+',
                    default={'preset': 'P4', 'codec': 'h264'}, help='Pass arguments on to the nvidia encoder, e.g. -ea preset=P4 codec=h264')
parser.add_argument(
    'VIDEO_FILES', help='The folder containing the video files to mosaic')
args = parser.parse_args()

# Create the decoders
decoders = []
mappers = []
iw, ih, ifmt, irate = None, None, None, None
for f in listdir(args.VIDEO_FILES):
    full_path = path.join(args.VIDEO_FILES, f)
    if path.isfile(full_path):
        if not args.quiet:
            print(f'loading {f}...')

        decoder = nvc.PyNvDecoder(full_path, args.gpu_id)
        decoders.append(decoder)

        if iw is None and ih is None and ifmt is None and irate is None:
            iw, ih = decoder.Width(), decoder.Height()
            ifmt = decoder.Format()
            irate = decoder.Framerate()

        assert iw == decoder.Width() and ih == decoder.Height() and ifmt == decoder.Format(
        ) and irate == decoder.Framerate(), 'All mosaic files must have the same width, height, pixel format and frame rate'

if len(decoders) == 0:
    print(f'No files found in {args.VIDEO_FILES}. Not rendering mosaic')
    exit(0)

# Set the output width to the input width if it's not specified
if not args.output_width:
    args.output_width = iw
    args.output_height = ih

assert args.output_width / args.output_height == iw / \
    ih, 'The aspect ratio of input and output must be the same'

# Calculate the grid sizes for the mosaic
mosaics_per_dimension = ceil(sqrt(len(decoders)))
mosaic_width = args.output_width // mosaics_per_dimension
mosaic_height = args.output_height // mosaics_per_dimension

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
output_stream = output_file.add_stream('h264', rate=irate)
output_stream.width = args.output_width
output_stream.height = args.output_height

# TODO: This is a cheeky hack
if args.quiet:
    def tqdm(x): return x

global frame_counter
frame_counter = 0


def write_packet(data):
    global frame_counter

    packet = av.packet.Packet(bytes(data))
    packet.dts = packet.pts = frame_counter * 1000
    packet.time_base = Fraction(1, int(irate * 1000))
    packet.stream = output_stream
    output_file.mux(packet)

    frame_counter += 1


# Main loop
frames = []
for frame in tqdm(range(min(map(lambda x: x.Numframes(), decoders)))):
    frame_surf = nvc.Surface.Make(
        nvc.PixelFormat.RGB, args.output_width, args.output_height, args.gpu_id)
    frame_plane = frame_surf.PlanePtr()
    frame_buf = GPUArray((args.output_height, args.output_width, 3),
                         dtype=np.uint8, gpudata=frame_plane.GpuMem())
    frame_buf.strides = (frame_plane.Pitch(), 3, 1)

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
        # This should happen entirely on the GPU and be lightning fast
        frame_buf[row*mosaic_height:row*mosaic_height+mosaic_height,
                  col*mosaic_width:col*mosaic_width+mosaic_width] = surf_buf

    # Go back to YUV for compression
    frame_surf = to_yuv.Execute(frame_surf, colour_space)

    encoded = np.empty([], dtype=np.uint8)
    if encoder.EncodeSingleSurface(frame_surf, encoded, sync=False):
        write_packet(encoded)

# Flush the encoder queue
encoded = np.empty([], dtype=np.uint8)
while encoder.FlushSinglePacket(encoded):
    write_packet(encoded)
