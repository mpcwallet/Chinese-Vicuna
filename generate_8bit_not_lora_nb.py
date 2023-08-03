import sys
import torch
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
import transformers
import gradio as gr
import argparse
import warnings
import os
from utils import StreamPeftGenerationMixin,StreamLlamaForCausalLM
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

BASE_MODEL = "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"
LOAD_8BIT = False
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=LOAD_8BIT,
    torch_dtype=torch.float16,
    device_map="auto", #device_map={"": 0},
)



def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
    input,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    min_new_tokens=1,
    repetition_penalty=2.0,
    **kwargs,
):
    prompt = generate_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=max_new_tokens, # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
        **kwargs,
    )
    with torch.no_grad():
        # if args.use_typewriter:
        #     for generation_output in model.stream_generate(
        #         input_ids=input_ids,
        #         generation_config=generation_config,
        #         return_dict_in_generate=True,
        #         output_scores=False,
        #         repetition_penalty=float(repetition_penalty),
        #     ):
        #         outputs = tokenizer.batch_decode(generation_output)
        #         show_text = "\n--------------------------------------------\n".join(
        #             [output.split("### Response:")[1].strip().replace('�','')+" ▌" for output in outputs]
        #         )
        #         # if show_text== '':
        #         #     yield last_show_text
        #         # else:
        #         yield show_text
        #     yield outputs[0].split("### Response:")[1].strip().replace('�','')
        # else:
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            repetition_penalty=1.3,
        )
        output = generation_output.sequences[0]
        output = tokenizer.decode(output).split("### Response:")[1].strip()
        print(output)
        yield output


gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Input", placeholder="Tell me about alpacas."
        ),
        gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
        gr.components.Slider(minimum=1, maximum=10, step=1, value=4, label="Beams Number"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=256, label="Max New Tokens"
        ),
        gr.components.Slider(
            minimum=1, maximum=300, step=1, value=1, label="Min New Tokens"
        ),
        gr.components.Slider(
            minimum=0.1, maximum=10.0, step=0.1, value=2.0, label="Repetition Penalty"
        ),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=25,
            label="Output",
        )
    ],
    title="LLM",
    description="LLL",
).queue().launch(share=True)


'''
!git clone https://github.com/mpcwallet/Chinese-Vicuna
%cd Chinese-Vicuna
!git checkout ray
!pip install -r requirements.txt

# generate
!CUDA_VISIBLE_DEVICES=0 python generate_8bit_not_lora.py \
    --model_path TheBloke/Wizard-Vicuna-13B-Uncensored-HF \
    --use_local 0 \
    --use_typewriter 1
'''