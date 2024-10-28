import logging
import math
import os
import random
import glob
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, Dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import urllib.request

check_min_version("0.16.0.dev0")

logger = get_logger(__name__, log_level="INFO")

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def main():
    output_dir = "lora_output"

    # Set seed before initializing model.
    seed = 1337

    print("Configuring accelerator")
    logging_dir = os.path.join(output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(total_limit=None)

    accelerator = Accelerator(
            gradient_accumulation_steps=4,
            mixed_precision="fp16",
            log_with="tensorboard",
            logging_dir=logging_dir,
            project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)


    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=None
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=None
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=None)
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=None
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print(accelerator.device)

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        # print(name)
        if name.startswith("mid_block"):
            # print(unet.config.block_out_channels)
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            # print(hidden_size)
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            # print(hidden_size)

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)


    optimizer_cls = torch.optim.AdamW
    learning_rate = 1e-4
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08


    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=learning_rate,
        betas=(adam_beta1,adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    print('Loading data')

    if not os.path.exists("mini_classes.txt"):
        os.system("wget 'https://raw.githubusercontent.com/zaidalyafeai/zaidalyafeai.github.io/master/sketcher/mini_classes.txt'")


    f = open("mini_classes.txt","r")
    # And for reading use
    classes = f.readlines()
    f.close()

    classes = [c.replace('\n','').replace(' ','_') for c in classes]

    def download():
        base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
        for c in tqdm(classes):        
            cls_url = c.replace('_', '%20')
            path = base+cls_url+'.npy'
            # print(path)
            urllib.request.urlretrieve(path, 'scribble_data/'+c+'.npy')

    if not os.path.exists('scribble_data'):
        os.makedirs('scribble_data')
        download()
    else:
        print('scribble_data already exists')

    max_items_per_class = 20
    def load_data_for_diffusion(root, max_items_per_class= max_items_per_class ):
        all_files = glob.glob(os.path.join(root, '*.npy'))

        #initialize variables
        imgs = np.empty([0, 784])
        labels = []

        for idx, file in enumerate(all_files):
            data = np.load(file)
            data = data[0: max_items_per_class, :]

            class_name, ext = os.path.splitext(os.path.basename(file))
            labels.extend(["a scribble of " + class_name for i in range(data.shape[0])])

            imgs = np.concatenate((imgs, data), axis=0)


        return imgs, labels

    imgs, labels = load_data_for_diffusion('scribble_data')

    from PIL import Image
    def gen_for_hf(imgs, labels):
        for idx in range(len(imgs)):
            img = Image.fromarray(imgs[idx].reshape(28,28))
            label = labels[idx]
            yield {"image": img, "text": label}

    scribble_dataset = Dataset.from_generator(gen_for_hf, gen_kwargs={"imgs": imgs, "labels": labels})

    def tokenize_captions(examples):
        captions = []
        for caption in examples["text"]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
                
        inputs = tokenizer(
                captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
        return inputs.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomInvert(p=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    train_dataset = scribble_dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example['pixel_values'] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example['input_ids'] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_batch_size=8

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=2,
    )

    gradient_accumulation_steps = 4
    num_train_epochs = 70
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    lr_warmup_steps = 0

    lr_scheduler = "cosine"
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            lora_layers, optimizer, train_dataloader, lr_scheduler
        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(labels)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # load from checkpoint if there is one
    resume_from_checkpoint = True
    if resume_from_checkpoint:
        dirs = os.listdir(output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(f"No checkpoint found in {output_dir}. Starting from scratch")
        resume_from_checkpoint = False
    else:
        accelerator.print(f"Checkpoint found in {output_dir}. Resuming from checkpoint")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * gradient_accumulation_steps)


    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    checkpointing_steps = 500
    validation_epochs = 1
    num_validation_images = 1
    max_grad_norm = 1.0

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    
                
                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Gather the losses across all processes
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            # update procress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss":  train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        if accelerator.is_main_process:
            validation_prompt = "a scribble of cat"
            if epoch % validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {num_validation_images} images with prompt:"
                    f" {validation_prompt}."
                )

                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path,
                    safety_checker = None,
                    unet=accelerator.unwrap_model(unet),
                    revision=None,
                    torch_dtype=weight_dtype,
                )

                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device).manual_seed(seed)
                images = []
                for i in range(num_validation_images):
                    image = pipeline(validation_prompt, num_inference_steps=30, generator=generator).images[0]
                    image.save(os.path.join(output_dir, f"validation_{epoch}_{i}.png"))

                del pipeline
                torch.cuda.empty_cache()

    # save lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(output_dir)

    # final inference
    pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, safety_checker = None, revision=None, torch_dtype=weight_dtype
    )

    pipeline = pipeline.to(accelerator.device)

    # load attention processors
    pipeline.unet.load_attn_procs(output_dir)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)
    images = []

    for i in range(num_validation_images):
        image = pipeline(validation_prompt, num_inference_steps=30, generator=generator).images[0]
        image.save(os.path.join(output_dir, f"final_{i}.png"))
        
    accelerator.end_training()


if __name__ == "__main__":
    main()
