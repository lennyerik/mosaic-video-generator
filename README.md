# mosaic-video-generator
A tiling mosaic-esque video generator for the impatient with powerful GPUs.

## Installing prerequisites
You must already have CUDA, FFmpeg, cmake and standard gnu build-tools installed.

    pip install pycuda numpy tqdm av
    git clone https://github.com/NVIDIA/VideoProcessingFramework.git
    cd VideoProcessingFramework
    pip install .

## Running
Look at the help:

    ./mosaic.py -h

Example

    ./mosaic.py -o myvideo.mp4 somefolder/with/videos/

## On the topic of performance
The code is already pretty fast.
I tested with 16 20s 1440p 60fps clips, which took 40s to tile, so about 0.5x realtime.
That amounts to a normalised speed of 8x realtime per clip, or a decoding performance of at least 480 frames per second (due to the reencoding overhead).

Although I would like even more speed, I doubt this is possible without rewriting the whole thing in C++.
It's definitely not worth the effort for maybe like 1% more performance.

The reason I'm fairly sure about this is because I profiled the python code: 30 of the 40 seconds of runtime
on a batch of 15 videos were spent in `decoder.DecodeSingleSurface()` which is pretty much only a high-level wrapper
around a few CUDA / CUVID API calls.
Most of the work seems to already be happening on the GPU, mostly bottlenecked by memcpy speed as far as I can tell.
