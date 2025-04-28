# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import importlib


import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
import numpy as np
import re
import json


from .utils import DPODataCollatorWithPadding, pad_to_length

def parse_titles(text: str):
    """Extract all substrings between double quotes."""
    raw_titles = re.findall(r'"([^"]+)"', text)
    cleaned_titles = []
    for t in raw_titles:
        # 如果最前面有單引號，就去掉它
        t = re.sub(r'^[^0-9A-Za-z#(]+', '', t)
        # 如果最後面有單引號，就去掉它
        if t.endswith("'"):
            t = t[:-1]
        cleaned_titles.append(t)
    return cleaned_titles

def avg_emb(titles, book2idx, item_emb):
    """Compute the average embedding of a list of titles."""
    idxs = [book2idx.get(t, None) for t in titles]
    idxs = [i for i in idxs if i is not None]
    if not idxs:
        for t in titles:
            if(book2idx.get(t, None) == None):
                print(f"{t} not found in book2idx!! \n")
        return None
    return item_emb[idxs].mean(axis=0)

def l2(a, b):
    """Compute Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)

def is_peft_available():
    return importlib.util.find_spec("peft") is not None

if is_peft_available():
    # from peft import get_peft_model, prepare_model_for_int8_training
    from peft import get_peft_model


class DSDPOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        beta: float = 0.1,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        # optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
        #     None,
        #     None,
        # ),
        optimizers: Optional[Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        # DS-DPO: 
        beta_range: Tuple[float, float] = (0.01, 2.0), # DS-DPO: β₀ 的范围
        max_neg: int = 10, # DS-DPO: max number of rejected samples
        distance_type: str = "dp",
        book2idx: Dict[str, int] = None,
        item_emb: np.ndarray = None,
        beta_log_path: str = None,
    ):
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
            #     model = prepare_model_for_int8_training(model)
            model = get_peft_model(model, peft_config)

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128

            data_collator = DPODataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        self.beta = beta
        self.ref_model = ref_model
        self.beta_range = beta_range
        beta_low, beta_high = self.beta_range
        print(f'beta_low: {beta_low}, beta_high: {beta_high}')
        self.max_neg = max_neg
        self.distance_type = distance_type
        self.book2idx = book2idx
        self.item_emb = item_emb
        self.beta_log_path =  beta_log_path
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            None,
            callbacks,
            # optimizers,
            preprocess_logits_for_metrics,
        )

        # Since we inherit from trainer we always have access to an accelerator
        if hasattr(self, "accelerator"):
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        # 把 chosen 和 rejected response 拼接起来
        rejected_max_len = max([batch[key].shape[1] for key in batch if key.startswith("rejected") and key.endswith("_input_ids")])
        max_length = max(batch["chosen_input_ids"].shape[1], rejected_max_len)
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                # concatenated_key = k.replace("rejected", "concatenated")
                prefix = k.split("_")[0]
                concatenated_key = "concatenated" + k[len(prefix):] 
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0, # 0 to 1
                ).to(self.accelerator.device)
        return concatenated_batch

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: Dict[str, torch.FloatTensor],
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: Dict[str, torch.FloatTensor],
        rejected_dc: Dict[str, np.float64], # DS-DPO: rejected_distances
        rejected_dp: Dict[str, np.float64], # DS-DPO: rejected_distances
        rejected_dpc: Dict[str, np.float64], # DS-DPO: rejected_distances
        reference_free: bool = False,
        beta0: float = None,                   # DS-DPO: origin β₀
        delta: float = None,                   # DS-DPO: origin δ  
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
         # pi_logratios = policy_chosen_logps - policy_rejected_logps
        # for key in policy_rejected_logps:
        # ref_logratios = reference_chosen_logps - reference_rejected_logps

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        # print(f"chosen:{chosen_logratios}")
        rejected_logratios = {}
        for key in policy_rejected_logps:
            rejected_logratios[key] = policy_rejected_logps[key] - reference_rejected_logps[key]
            # print(f"{key}_logratios:{rejected_logratios[key].shape}")
        # if reference_free:
        #     ref_logratios = 0

        # beta
        # 假设 rejected_dc: Dict[str, float] 存了各个 rejected{i} 的距离  
        # 首先收集所有距离，做一次批次内的 min-max 归一化  
        # distances = [rejected_dc[key] for key in rejected_dc]  
        if self.distance_type == "dc":
            distances = [rejected_dc[key] for key in rejected_dc]
        elif self.distance_type == "dp":
            distances = [rejected_dp[key] for key in rejected_dp]  
        elif self.distance_type == "dpc":  
            distances = [rejected_dpc[key] for key in rejected_dpc]

        d_min, d_max = min(distances), max(distances)  
        # 为了避免除零，做一个小保护  
        denom = (d_max - d_min) if d_max > d_min else 1.0  
        beta_low, beta_high = self.beta_range

        # 归一化后映射到 [0.01, 2.0]  
        adapted_betas: Dict[str, float] = {}  
        if self.distance_type == "dc":
            for key, d in rejected_dc.items():
                norm = (d - d_min) / denom
                adapted_betas[key] = beta_low + norm * (beta_high - beta_low)
        elif self.distance_type == "dp":
            for key, d in rejected_dp.items():
                norm = (d - d_min) / denom
                adapted_betas[key] = beta_low + norm * (beta_high - beta_low)
        elif self.distance_type == "dpc":
            for key, d in rejected_dpc.items():
                norm = (d - d_min) / denom
                adapted_betas[key] = beta_low + norm * (beta_high - beta_low)

        serializable_betas = {k: float(v) for k, v in adapted_betas.items()}
        with open(self.beta_log_path, "a") as f:
            json.dump(serializable_betas, f)
            f.write("\n")

        # 然后在计算 temp 的时候，用对应的 adapted_beta  
        temp = 0  
        for key in adapted_betas:  
            beta_i = adapted_betas[key]  
            temp += torch.exp(  
                beta_i *  
                (rejected_logratios[key] - chosen_logratios)  
            )  

        # logits = pi_logratios - ref_logratios
        # temp = sum(torch.exp(self.beta * (rejected_logratios[key] - chosen_logratios)) for key in rejected_logratios)
        temp1 = -torch.log(temp)
        losses = -F.logsigmoid(temp1)
        # losses = -F.logsigmoid(self.beta * logits)
        rejected_rewards = {}
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        for key in policy_rejected_logps:
            rejected_rewards[key] = self.beta * (policy_rejected_logps[key] - reference_rejected_logps[key]).detach()

        return losses, chosen_rewards, rejected_rewards

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor], torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        # print(concatenated_batch["concatenated_input_ids"].shape)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        step = batch["chosen_input_ids"].shape[0]
        rejected_logps = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logps[f"rejected{cnt}"] = all_logps[step*cnt : step*(cnt+1)]

        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        rejected_logits = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logits[f"rejected{cnt}"] = all_logits[step*cnt : step*(cnt+1)]
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, batch)
        
        # print("\n\n")
        # for k, v in batch.items():
        #     print(f"{k!r} -> {type(v).__name__}")
        #     if type(v).__name__ == "Tensor":
        #         print(f"  {v.shape} {v.dtype} {v.device}")
        #     else:
        #         print(f"  {len(v)} {type(v[0]).__name__} {type(v[0])} {v[0]}")
        #     print("-" * 20)
        #     print("\n\n")
        
        rejected_dc = {}
        rejected_dp = {}
        rejected_dpc = {}

        past_title = parse_titles(batch["prompt"][0])
        chosen_title =parse_titles(batch["chosen_response_only"][0])[0]
        # print(f"past_title: {past_title}")
        # print(f"chosen_title: {chosen_title}")

        emb_past   = avg_emb(past_title, self.book2idx, self.item_emb)
        emb_chosen = avg_emb([chosen_title], self.book2idx, self.item_emb)
        emb_pc     = avg_emb(past_title + [chosen_title], self.book2idx, self.item_emb)

        for i in range(1, self.max_neg + 1):
            # emb_rejected = self.item_emb[self.book2idx.get(batch[f"rejected{i}_response_only"][0], 0)]
            if len(batch[f"rejected{i}_response_only"][0]) == 0:
                rejected_dc[f"rejected{i}"] = 0
                rejected_dp[f"rejected{i}"] = 0
                rejected_dpc[f"rejected{i}"] = 0
            else:
                rejected_title = parse_titles(batch[f"rejected{i}_response_only"][0])[0]
                emb_rejected = avg_emb([rejected_title], self.book2idx, self.item_emb)
                rejected_dc[f"rejected{i}"] = l2(emb_rejected, emb_chosen)
                rejected_dp[f"rejected{i}"] = l2(emb_rejected, emb_past)
                rejected_dpc[f"rejected{i}"] = l2(emb_rejected, emb_pc)
        
        # for i in range(1, self.max_neg + 1):
        #     rejected_dc[f"rejected{i}"] = l2(batch[f"emb_rejected{i}"], batch["emb_chosen"])
        #     rejected_dp[f"rejected{i}"] = l2(batch[f"emb_rejected{i}"], batch["emb_past"])
        #     rejected_dpc[f"rejected{i}"] = l2(batch[f"emb_rejected{i}"], batch["emb_pc"])

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            rejected_dc,
            rejected_dp,
            rejected_dpc
        )
        
        # reward_accuracies 记录 chosen 比所有 rejected 的收益都大的比例是多少
        reward_accuracies = None
        for key in rejected_rewards:
            if reward_accuracies is None:
                reward_accuracies = (chosen_rewards > rejected_rewards[key]).float()
            else:
                reward_accuracies *= (chosen_rewards > rejected_rewards[key]).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/{key}"] = rejected_rewards[key].cpu().numpy().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/margins-{key}"] = (chosen_rewards - rejected_rewards[key]).cpu().numpy().mean()
        for key in policy_rejected_logps:    
            metrics[f"{prefix}logps/rejected-{key}"] = policy_rejected_logps[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        for key in policy_rejected_logits:    
            metrics[f"{prefix}logits/rejected-{key}"] = policy_rejected_logits[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # print(inputs.keys())
        # print(inputs)
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    # def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
    #     """Generate samples from the model and reference model for the given batch of inputs."""

    #     policy_output = model.generate(
    #         batch["prompt_input_ids"],
    #         attention_mask=batch["prompt_attention_mask"],
    #         max_length=self.config.max_length,
    #         do_sample=True,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #     )

    #     reference_output = self.ref_model.generate(
    #         batch["prompt_input_ids"],
    #         attention_mask=batch["prompt_attention_mask"],
    #         max_length=self.config.max_length,
    #         do_sample=True,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #     )

    #     policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
    #     policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

    #     reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
    #     reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

    #     return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "logits_test/chosen": metrics["logits_test/chosen"],
            # "logits_test/rejected": metrics["logits_test/rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)
        labels = torch.zeros(logits.shape[0])

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    # def log(self, logs: Dict[str, float]) -> None:
    #     """
    #     Log `logs` on the various objects watching training, including stored metrics.

    #     Args:
    #         logs (`Dict[str, float]`):
    #             The values to log.
    #     """
    #     # logs either has 'loss' or 'eval_loss'
    #     train_eval = "train" if "loss" in logs else "eval"
    #     # Add averaged stored metrics to logs
    #     for key, metrics in self._stored_metrics[train_eval].items():
    #         logs[key] = torch.tensor(metrics).mean().item()
    #     del self._stored_metrics[train_eval]
    #     return super().log(logs)
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`float`, *optional*):
                The time training started, used for logging throughput.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"

        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()

        del self._stored_metrics[train_eval]

        return super().log(logs, start_time)
