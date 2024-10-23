#!/usr/bin/python3
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

def parse_arguments():
    """
    Parse command line arguments for prompts and filenames.
    Allows either a single prompt for all files or multiple prompts (one per file).
    Returns parsed arguments containing prompt(s) and filenames.
    """
    parser = argparse.ArgumentParser(
        description='''Process files with prompts. You can either:
        1. Provide a single prompt to apply to all files
        2. Provide multiple prompts, one for each file''',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Create a mutually exclusive group for prompt arguments
    prompt_group = parser.add_mutually_exclusive_group(required=True)

    prompt_group.add_argument(
        '-p', '--prompts',
        nargs='+',
        help='List of prompts (one per file)'
    )

    prompt_group.add_argument(
        '-sp', '--single-prompt',
        help='Single prompt to apply to all files'
    )

    parser.add_argument(
        '-f', '--files',
        nargs='+',
        help='List of filenames to process',
        required=True
    )

    return parser.parse_args()


def validate_args(args):
    """
    Validate command line arguments.
    Ensures prompt count matches file count when using multiple prompts.
    """
    if args.prompts and len(args.prompts) != len(args.files):
        raise ValueError(
            f"Number of prompts ({len(args.prompts)}) must match number of files ({len(args.files)}) "
            "when using multiple prompts. Use --single-prompt to apply one prompt to all files."
        )

# specify the path to the model
model_path = "deepseek-ai/Janus-1.3B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, attn_implementation="eager"
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

args = parse_arguments()

try:
    validate_args(args)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

# If using single prompt, create a list of the same prompt for each file
prompts = args.prompts if args.prompts else [args.single_prompt] * len(args.files)

print("Files to process:", args.files)
print("Using single prompt:" if args.single_prompt else "Using multiple prompts:")

# Process files with their prompts
for prompt, filename in zip(prompts, args.files):
    print(f"\nProcessing:")
    print(f"  Prompt: {prompt}")
    print(f"  File: {filename}")

    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{prompt}.",
            "images": [f"/app/images/{filename}"],
        },
        {"role": "Assistant", "content": ""},
    ]

    # load images and prepare inputs
    pil_images = load_pil_images(conversation)
    with torch.no_grad():  # Prevent gradient computation
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"{prepare_inputs['sft_format'][0]}", answer)

    # Clean up the largest tensors
    del outputs
    del inputs_embeds

    # Optional light cache clear if you notice memory growth
    if torch.cuda.is_available():
        torch.cuda.empty_cache()