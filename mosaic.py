#!/usr/bin/env python3

from os import listdir, path
from math import sqrt, ceil
import PyNvCodec as nvc
from pycuda.gpuarray import GPUArray
import pycuda
import argparse
from tqdm import tqdm
import numpy as np


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
                    default={'preset': 'hq', 'codec': 'h264'}, help='Pass arguments on to the nvidia encoder, e.g. -ea peset=hq codec=h264')
parser.add_argument(
    'VIDEO_FILES', help='The folder containing the video files to mosaic')
args = parser.parse_args()

decoders = []
mappers = []
iw, ih, ifmt = None, None, None
for f in listdir(args.VIDEO_FILES):
    full_path = path.join(args.VIDEO_FILES, f)
    if path.isfile(full_path):
        if not args.quiet:
            print(f'loading {f}...')

        decoder = nvc.PyNvDecoder(full_path, args.gpu_id)
        decoders.append(decoder)

        if iw is None and ih is None and ifmt is None:
            iw, ih = decoder.Width(), decoder.Height()
            ifmt = decoder.Format()

        # TODO: Do they really need to have the same pixel format?
        assert iw == decoder.Width() and ih == decoder.Height() and ifmt == decoder.Format(
        ), 'All mosaic files must have the same width, height and pixel format'

        from os import getenv
        if getenv('DEBUG', False):
            break

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

# Create the rgb converter and Remappers
to_rgb = nvc.PySurfaceConverter(
    mosaic_width, mosaic_height, nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, args.gpu_id
)
cc1 = nvc.ColorspaceConversionContext(
    nvc.ColorSpace.BT_709, nvc.ColorRange.JPEG)
mosaic_resizer = nvc.PySurfaceResizer(mosaic_width, mosaic_height, ifmt, args.gpu_id)


def show(surf):
    dl = nvc.PySurfaceDownloader(
        surf.Width(), surf.Height(), surf.Format(), args.gpu_id)
    out = np.zeros([], dtype=np.uint8)
    dl.DownloadSingleSurface(surf, out)
    from matplotlib import pyplot as plt
    plt.imshow(out.reshape((surf.Height(), surf.Width(), 3)),
               interpolation='nearest')
    plt.show()

frames = []
for frame in tqdm(range(min(map(lambda x: x.Numframes(), decoders)))):
    frame_surf = nvc.Surface.Make(nvc.PixelFormat.RGB, args.output_width, args.output_height, args.gpu_id)
    frame_plane = frame_surf.PlanePtr()
    frame_buf = GPUArray((args.output_height, args.output_width, 3), dtype=np.uint8, gpudata=frame_plane.GpuMem())
    frame_buf.strides = (frame_plane.Pitch(), 3, 1)
    print(frame_buf.strides)

    for i, decoder in enumerate(decoders):
        surf = decoder.DecodeSingleSurface()
        surf = mosaic_resizer.Execute(surf)
        surf = to_rgb.Execute(surf, cc1)

        surf_plane = surf.PlanePtr()
        surf_buf = GPUArray((mosaic_height, mosaic_width, 3), dtype=np.uint8, gpudata=surf_plane.GpuMem())
        surf_buf.strides = (surf_plane.Pitch(), 3, 1)

        row = i // mosaics_per_dimension
        col = i % mosaics_per_dimension
        frame_buf[row*mosaic_height:row*mosaic_height+mosaic_height,col*mosaic_width:col*mosaic_width+mosaic_width] = surf_buf

    show(frame_surf)
