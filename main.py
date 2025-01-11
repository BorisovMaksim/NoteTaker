import logging
import warnings
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.DEBUG)


logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import shutil
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch
import gc
from pathlib import Path
import time

from src.filtration import VideoFilter
from src.models import MODELS


def main(
    video_path,
    start_time,
    end_time,
    max_num_frames,
    model_name,
    step,
    prompt,
    text_model_name,
):
    start = time.time()
    if Path("./outputs").exists():
        shutil.rmtree("./outputs/")

    filtrator = VideoFilter()
    samples = filtrator.forward(video_path, start_time, end_time, max_num_frames)

    torch.cuda.empty_cache()
    gc.collect()

    model = MODELS[model_name]()
    images = [s["frame"] for s in samples]

    logging.info("Predicting text from frames...")
    Path("outputs/latex/").mkdir(exist_ok=True)
    Path("outputs/pdf/").mkdir(exist_ok=True)

    latex_files = []
    for i in tqdm(range(0, len(samples), step)):
        cur_text = model.forward_multiple(text=prompt, images=images[i : i + step])
        file_name = Path(f"outputs/latex/example_{i}.tex")
        file_name.write_text(cur_text)
        latex_files.append(file_name)

    del filtrator
    del samples
    del model

    torch.cuda.empty_cache()
    gc.collect()

    text_model = MODELS[text_model_name]()
    correct_latexs = ""
    for file in latex_files:
        compile_code = os.system(
            f"cd outputs/latex/; pdflatex -interaction=nonstopmode {file.name}  2>&1 > /dev/null"
        )
        pdf_path = Path(f"./outputs/latex/{Path(file.name).with_suffix('.pdf')}")

        code = Path(file).read_text()
        corrected_code = text_model.forward(
            "Summarize and rewrite the fololowing latex code. Use \documentclass{article}. Return latex code:\n"
            + code
        )
        file = file.parent / f"{file.stem}_corrected.tex"
        code_start = corrected_code.find("```latex")
        code_end = corrected_code.rfind("```")
        file.write_text(corrected_code[code_start + len("```latex") : code_end])
        print("Resulting code after LLM:\n", corrected_code)
        compile_code = os.system(
            f"cd outputs/latex/; pdflatex -interaction=nonstopmode {file.name}  2>&1 > /dev/null"
        )
        pdf_path = Path(f"./outputs/latex/{Path(file.name).with_suffix('.pdf')}")
        logging.info(f"After correction the code is {compile_code}")

        if pdf_path.exists():
            correct_latexs = (
                correct_latexs + "\n" + corrected_code[code_start + len("```latex") : code_end]
            )
            shutil.move(pdf_path, f"outputs/pdf/")
        else:
            logging.info(f"Could not compile {file}")

    resulting_code = text_model.forward(
        "Summarize the following latex documents. Return latex code:"
        + correct_latexs
    )
    print("Resulting code after LLM:\n", resulting_code)
    file = Path("./outputs/latex/result.tex")
    code_start = resulting_code.find("```latex")
    code_end = resulting_code.rfind("```")
    file.write_text(resulting_code[code_start + len("```latex") : code_end])
    compile_code = os.system(
        f"cd outputs/latex/; pdflatex -interaction=nonstopmode {file.name}  2>&1 > /dev/null"
    )
    pdf_path = Path(f"./outputs/latex/{Path(file.name).with_suffix('.pdf')}")
    if pdf_path.exists():
        shutil.move(pdf_path, f"outputs/pdf/")

    print("Seconds passed: ", start - time.time())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="/home/maksim/Downloads/youtube_LY7YmuDbuW0_1920x1080_h264.mp4",
    )
    parser.add_argument("--start_time", type=int, default=0)
    parser.add_argument("--end_time", type=int, default=10 * 60)
    parser.add_argument("--max_num_frames", type=int, default=500)
    parser.add_argument("--model_name", type=str, default="MiniCPM")
    parser.add_argument("--text_model_name", type=str, default="text_llama")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument(
        "--prompt", type=str, default="Take leacture notes. Output latex code"
    )

    args = parser.parse_args()
    main(**vars(args))
