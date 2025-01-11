import os

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_TOKEN"] = ""
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "1193"


import sys
from pathlib import Path

sys.path.append(Path("./llama-models/").absolute().as_posix())

from llama_models.llama3.api.datatypes import (
    CompletionMessage,
    StopReason,
    SystemMessage,
    UserMessage,
)

from llama_models.llama3.reference_impl.generation import Llama

import requests
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import AutoModel, AutoTokenizer
import clip
from PIL import Image
from tqdm import tqdm
from torch.nn import CosineSimilarity

import numpy as np


class InferTextLLama:
    def __init__(
        self,
        model_id="/home/maksim/.llama/checkpoints/Llama3.1-8B-Instruct/",
        device="cuda",
        max_seq_len=8000,
        max_batch_size=4,
        model_parallel_size=None,
    ):

        self.generator = Llama.build(
            ckpt_dir=model_id,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size,
        )

        self.device = device

    def forward(self, text):
        message = [UserMessage(content=text)]
        result = self.generator.chat_completion(
            message,
            max_gen_len=None,
            temperature=0.6,
            top_p=0.9,
        )
        out_message = result.generation
        return out_message.content


class InferLLama:
    def __init__(
        self, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct", device="cuda"
    ):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device

    def forward(self, text, image):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = self.processor(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=30000)
        assert len(output) == 1
        return self.processor.decode(output[0])


class InferMiniCMP:
    def __init__(self, model_id="openbmb/MiniCPM-V-2_6", device="cuda"):
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )  # sdpa or flash_attention_2, no eager
        self.model = self.model.eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        self.device = device

    def forward(self, text, image):
        msgs = [{"role": "user", "content": [image, text]}]

        res = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        generated_text = ""
        for new_text in res:
            generated_text += new_text
            # print(new_text, flush=True, end='')
        return generated_text

    def forward_multiple(self, text, images):
        images = [i.convert("RGB") for i in images]

        msgs = [{"role": "user", "content": images + [text]}]

        answer = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        generated_text = ""
        for new_text in answer:
            generated_text += new_text
        return generated_text


class InferVideoMiniCMP:
    def __init__(self, model_id="openbmb/MiniCPM-V-2_6", device="cuda"):
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )  # sdpa or flash_attention_2, no eager
        self.model = self.model.eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        self.device = device

    def forward(self, text, frames):
        msgs = [
            {"role": "user", "content": frames + [question]},
        ]
        params = {}
        params["use_image_id"] = True
        params["max_slice_nums"] = (
            1  # use 1 if cuda OOM and video resolution >  448*448
        )

        answer = self.model.chat(
            image=None, msgs=msgs, tokenizer=self.tokenizer, **params
        )

        return answer


class InferCLIP:
    def __init__(self, model_id="openbmb/MiniCPM-V-2_6", device="cuda"):

        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

        self.cos = CosineSimilarity()
        self.device = device

    def forward(self, images):
        features = []
        with torch.no_grad():
            for img in tqdm(images):
                image_processed = self.preprocess(img).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image_processed)
                features.append(image_features)
        return features

    def compute_sim(self, images):
        features = self.forward(images)
        features_lag = features.copy()
        features_lag.insert(0, torch.ones(features[0].shape).to(self.device))
        features_lag.pop()

        features_concat = torch.concat(features)
        features_concat_lag = torch.concat(features_lag)

        sims = self.cos(features_concat, features_concat_lag)
        sims = sims.cpu().numpy()

        return sims
            

    def compute_quality(self, images):
        text = clip.tokenize(["The image is clear", "The image is not clear"]).to(
            self.device
        )
        # text_features = self.model.encode_text(text)

        res = []
        with torch.no_grad():
            for img in images:
                image_processed = self.preprocess(img).unsqueeze(0).to(self.device)
                # image_features = self.model.encode_image(image_processed)

                logits_per_image, logits_per_text = self.model(image_processed, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                res.append(probs)

        prediction = np.argmax(res, axis=2)
        return prediction


MODELS = {
    "text_llama": InferTextLLama,
    "llama": InferLLama,
    "MiniCPM": InferMiniCMP,
    "VideoMiniCMP": InferVideoMiniCMP,
    "CLIP": InferCLIP,
}
