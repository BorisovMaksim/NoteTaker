from decord import VideoReader, cpu    # pip install decord
import numpy as np
from PIL import Image
import sys


def combine_images(images):
    
    widths, heights = zip(*(i.size for i in images))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    new_im = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
        
    return new_im



def encode_video(video_path, start_time=-1, end_time=float('inf'), max_frames=100):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    
    times = vr.get_frame_timestamp(range(len(vr))).mean(-1)
    # print(times)
    indexes = np.where((times < end_time) & (times > start_time))[0]
    # print(indexes)

    print(len(indexes) / len(times) * 100)
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(indexes[0], indexes[-1], sample_fps)]
    if len(frame_idx) > max_frames:
        frame_idx = uniform_sample(frame_idx, max_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

