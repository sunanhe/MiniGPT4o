import os
import sys
import math
import shutil
import numpy as np
from datetime import datetime


import torch
import gradio as gr

from transformers import AutoConfig

from llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llavavid.conversation import conv_templates, SeparatorStyle
from llavavid.model.builder import load_pretrained_model
from llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from parrots import SpeechRecognition
import parrots
from parrots import TextToSpeech

parrots_path = parrots.__path__[0]
sys.path.append(parrots_path) 

from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip


def copy_file_with_timestamp(src_name, cache_dir):

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    base_name = src_name.split(".")[-1]
    dest_name = f"{timestamp}.{base_name}"
    dest_path = os.path.join(cache_dir, dest_name)

    shutil.copy(src_name, dest_path)

    return dest_path

cache_dir = "cache"



# Initialize the ASR and TTS
tts_model = TextToSpeech(
    speaker_model_path="shibing624/parrots-gpt-sovits-speaker-maimai",
    speaker_name="MaiMai",
    device="cuda",
    half=True
)
asr_model = SpeechRecognition(model_name_or_path="BELLE-2/Belle-distilwhisper-large-v2-zh")

# Initialize the LlaVa-Next Video Model
model_path = "lmms-lab/LLaVA-NeXT-Video-7B-DPO"
model_name = get_model_name_from_path(model_path)

overwrite_config = {}
overwrite_config["mm_resampler_type"] = "spatial_pool"
overwrite_config["mm_spatial_pool_stride"] = 2
overwrite_config["mm_spatial_pool_out_channels"] = 1024
overwrite_config["mm_spatial_pool_mode"] = "average"
overwrite_config["patchify_video_feature"] = False

cfg_pretrained = AutoConfig.from_pretrained(model_path)


if "224" in cfg_pretrained.mm_vision_tower:
    least_token_number = 32*(16//2)**2 + 1000
else:
    least_token_number = 32*(24//2)**2 + 1000

scaling_factor = math.ceil(least_token_number/4096)

if scaling_factor >= 2:
    if "mistral" not in cfg_pretrained._name_or_path.lower() and "7b" in cfg_pretrained._name_or_path.lower():
        print(float(scaling_factor))
        overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, overwrite_config=overwrite_config)

def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, 32, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def stop_recording_fn(video):

    cur_video = copy_file_with_timestamp(video, cache_dir)
    return cur_video

def process_video(video_path):

    video_path = video_path  
    audio_path = 'temp.mp3'  
    
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)
    
    audio_clip.close()
    video_clip.close()

    asr_model = SpeechRecognition(model_name_or_path="BELLE-2/Belle-distilwhisper-large-v2-zh")
    asr_result = asr_model.recognize_speech_from_file('temp.mp3')

    sample_set = {}
    question = asr_result["text"]

    sample_set["Q"] = question
    sample_set["video_name"] = video_path

    video = load_video(video_path)
    video = [image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().to(model.device)]

    qs = question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    cur_prompt = question

    with torch.inference_mode():
        model.update_prompt([[cur_prompt]])
        output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria], temperature=0.2)


    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    tts_model.predict(
        text=outputs,
        text_language="auto",
        output_path="temp_output.mp3",
    )
    audio = "temp_output.mp3"
    return asr_result, outputs, audio



title = """<h1 align="center">TinyGPT4o Demo</h1>"""

with gr.Blocks() as demo:

    cur_video = gr.Textbox(label="Current Video", type="text", value=False, interactive=False, visible=False)
    asr_result = gr.Textbox(label="Asr Result", type="text", value=False, interactive=False, visible=False)
    
    gr.Markdown(title)
    video = gr.Video(sources=["webcam"], format="mp4", height=600, label="Video", scale=0.8, interactive=True, include_audio=True)
    audio = gr.Audio(type="filepath", label="Audio", scale=1, autoplay=True, interactive=False, visible=True)
    model_output = gr.Textbox(label="Output Text", type="text", interactive=False, visible=False)

    video.stop_recording(
        stop_recording_fn,
        inputs=[video],
        outputs=[cur_video],
    ).success(
        process_video,
        inputs=[cur_video],
        outputs=[asr_result, model_output, audio],
    )

    video.upload(
        stop_recording_fn,
        inputs=[video],
        outputs=[cur_video],
    ).success(
        process_video,
        inputs=[cur_video],
        outputs=[asr_result, model_output, audio],
    )            

demo.launch(share=False)