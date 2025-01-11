import os
import ssl
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import cv2
import torch
from pathlib import Path
logger = logging.getLogger(__name__)

os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

import easyocr
from ultralytics import YOLOv10

from src.tools import encode_video
from src.models import MODELS
from src.tools import combine_images
from src.boxes import Box, BoxCounter


class VideoFilter:
    def __init__(
        self, char_filter=5, sim_filter=0.95, area_filter=15, quality_filter=100
    ):
        self.ocr_reader = easyocr.Reader(["en"])
        self.clip = MODELS["CLIP"]()
        self.yolo = YOLOv10("yolov10x.pt")
        self.char_filter = char_filter
        self.sim_filter = sim_filter
        self.area_filter = area_filter
        self.quality_filter = quality_filter

    def filter_similar_frames(self, samples):
        logger.info("Filtering similar frames...")
        sims = self.clip.compute_sim(images=[s["frame"] for s in samples])

        samples_filtered = []
        for i, sim in enumerate(sims):
            if sim > self.sim_filter:
                continue

            sample = samples[i]
            sample["sim"] = sim
            samples_filtered.append(sample)

        return samples_filtered

    def filter_by_ocr(self, samples, plot=False):
        logger.info("Filtering by the number of characters on each frame...")

        samples_filtered = []

        for i, sample in enumerate(tqdm(samples)):

            image = sample["frame"]
            image_np = np.array(image)

            # fig = plt.figure()
            # plt.imshow(image)
            # Path("outputs/images/").mkdir(exist_ok=True, parents=True)
            # fig.savefig(f"outputs/images/image_original_{i}.png")
            fig = plt.figure()

            yolo_pred = self.yolo.predict(image)
            if len(yolo_pred[0].boxes) == 0 or int(yolo_pred[0].boxes[0].cls) != 0:
                continue

            res = self.ocr_reader.readtext(image_np)

            x1, y1, w, h = yolo_pred[0].boxes[0].xywh[0].cpu()
            human_box = Box((x1 - w / 2, y1 - h / 2), w, h)

            counter = BoxCounter()

            for r in res:
                x0, x1, x2, x3 = r[0]
                width = x3[1] - x0[1]
                height = x1[0] - x0[0]

                box = Box(x0, height, width)
                counter.add(box)

            num_boxes = len(counter.mid_points)
            sample["num_boxes"] = num_boxes

            if num_boxes < self.char_filter:
                continue

            average_box_area = (counter.total_area / len(counter.mid_points)) * 100
            average_box_area_norm = (
                average_box_area / (image_np.shape[0] * image_np.shape[1]) * 100
            )
            sample["average_box_area_norm"] = average_box_area_norm

            if average_box_area_norm < self.area_filter:
                continue
            counter.cluster()
            counter.calculate_cluster_box(plot=plot)

            closest_label = -1
            closest_dist = float("inf")
            print(f"{len(counter.label2cluster_box)=}")
            for label, cluster_point in counter.label2cluster_box.items():
                dist = np.sqrt(
                    (human_box.mid_point[0] - cluster_point.mid_point[0]) ** 2
                    + (human_box.mid_point[1] - cluster_point.mid_point[1]) ** 2
                )
                if dist < closest_dist:
                    closest_dist = dist
                    closest_label = label


            if closest_label == -1:
                continue
            closest_box = counter.label2cluster_box[closest_label]
            sample["frame_cropped"] = image.crop(
                (
                    closest_box.x0[0],
                    closest_box.x1[1],
                    closest_box.x2[0],
                    closest_box.x3[1],
                )
            )
            # sample_embedding = self.clip.forward([sample["frame_cropped"]])[0]

            samples_filtered.append(sample)

            # if len(samples_filtered) == 0:
            #     samples_filtered.append(sample)
            #     samples_embeddings = sample_embedding
            # else:
            #     sims = self.clip.cos(samples_embeddings, sample_embedding)
            #     mask = sims > 0.9
            #     logger.debug(f"{sims=}")
            #     logger.debug(f"{samples_embeddings.shape=}")
            #     logger.debug(f"{mask=}")
            #     if mask.any():
            #         samples_filtered = [sample if is_similar else existing_sample for existing_sample, is_similar in zip(samples_filtered, mask)]
            #         samples_embeddings[sims > 0.9] = sample_embedding
            #     else:
            #         samples_filtered.append(sample)
            #         samples_embeddings = torch.concat([samples_embeddings, sample_embedding], axis=0)

            if plot:
                plt.imshow(image)
                human_box.plot(color="purple")
                for box in counter.boxes:
                    box.plot()
                counter.plot_mid_points(axis=plt.gca())
                plt.plot(
                    [
                        human_box.mid_point[0],
                        counter.label2cluster_box[closest_label].mid_point[0],
                    ],
                    [
                        human_box.mid_point[1],
                        counter.label2cluster_box[closest_label].mid_point[1],
                    ],
                    c="orange",
                )
                Path("outputs/images/").mkdir(exist_ok=True, parents=True)
                fig.savefig(f"outputs/images/image_cluster_{i}.png")
                



                # fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))

                # ax1.imshow(image)
                # human_box.plot(color="purple", axis=ax1)
                # for box in counter.boxes:
                #     box.plot(axis=ax1)
                # counter.plot_mid_points(axis=ax1)
                # ax1.plot(
                #     [
                #         human_box.mid_point[0],
                #         counter.label2cluster_box[closest_label].mid_point[0],
                #     ],
                #     [
                #         human_box.mid_point[1],
                #         counter.label2cluster_box[closest_label].mid_point[1],
                #     ],
                #     c="orange",
                # )
                
                # ax2.imshow(sample['frame_cropped'])
                # plt.show()
            
        return samples_filtered

    def filter_by_quality(self, samples):
        logger.info("Filtering by the image quality...")

        samples_filtered = []

        for sample in tqdm(samples):
            var = cv2.Laplacian(np.array(sample["frame"]), cv2.CV_64F).var()

            if var < self.quality_filter:
                continue

            sample["quality"] = var
            samples_filtered.append(sample)

        return samples_filtered

    def forward(self, video_path, start_time, end_time, max_num_frames):
        logger.info("Reading input video...")
        frames = encode_video(
            video_path,
            start_time=start_time,
            end_time=end_time,
            max_frames=max_num_frames,
        )

        samples = [{"frame": frame} for frame in frames]

        samples = self.filter_by_quality(samples)
        samples = self.filter_similar_frames(samples)
        samples = self.filter_by_ocr(samples, plot=True)

        self.save_results(samples)

        logger.info(f"Total num samples = {len(samples)}")

        return samples

    def plot_results(self, samples):
        for i, sample in enumerate(samples):
            for k, v in sample.items():
                if k != "frame":
                    print(f"{k} = {v}")

            plt.imshow(sample["frame"])
            plt.show()

    def save_results(self, samples):
        Path("outputs/images/").mkdir(exist_ok=True, parents=True)
        for i, sample in enumerate(samples):
            text_data = "image {i}:\n"
            for k, v in sample.items():
                if k != "frame" and k != 'frame_cropped':
                    text_data += f"{k} = {v}\n"
            Path(f"outputs/images/image_{i}.txt").write_text(text_data)

            image = sample['frame_cropped']
            cv2.imwrite(f"outputs/images/image_{i}.png", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    video_path = "/home/maksim/Downloads/youtube_LY7YmuDbuW0_1920x1080_h264.mp4"
    start_time, end_time = 0, 5 * 60
    max_num_frames = 150
