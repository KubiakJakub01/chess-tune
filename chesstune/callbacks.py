from typing import Any

import torch
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.integrations import TensorBoardCallback

from .config import TrainArgs
from .inference_ops import run_inference
from .sft_ops import check_token_embeddings_health
from .utils import log_info


class LogTextSamplesCallback(TensorBoardCallback):
    """A callback that logs generated text samples to console and TensorBoard during evaluation."""

    def __init__(
        self,
        args: TrainArgs,
        eval_dataset: Dataset | None = None,
    ):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.args = args

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Log text samples to logger and TensorBoard at evaluation time."""
        # pylint: disable=unused-argument
        if state.is_world_process_zero:
            model = kwargs['model']
            tokenizer = kwargs['processing_class']

            self._log_text_samples(
                model=model,
                tokenizer=tokenizer,
                state=state,
            )

            # Additional diagnostic: print embedding statistics for new tokens
            check_token_embeddings_health(model, tokenizer)

    @torch.inference_mode()
    def _log_text_samples(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        state: TrainerState,
    ) -> None:
        """Generate and log text samples."""
        if self.eval_dataset is None:
            log_info('Evaluation dataset not provided, skipping text sample logging.')
            return

        samples = self.eval_dataset.take(self.args.num_samples_to_log)
        prompts = []
        generated_texts = []

        log_info('--- Generated Text Samples ---')
        for i, sample in enumerate(samples):
            prompt = sample['messages'][0]['content']

            generated_text = run_inference(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                config=self.args.generation_config,
            )

            prompts.append(prompt)
            generated_texts.append(generated_text)

            log_info('--- Sample %d ---', i + 1)
            log_info('Prompt:\n%s', prompt)
            log_info('Generated Text:\n%s', generated_text)
            log_info('--- End of Sample %d ---', i + 1)

        log_info('--- End of Generated Text Samples ---')

        self._log_to_tensorboard(prompts, generated_texts, state)

    def _log_to_tensorboard(
        self, prompts: list[str], generated_texts: list[str], state: TrainerState
    ) -> None:
        if self.tb_writer is not None:
            text_to_log = ''
            for i in range(self.args.num_samples_to_log):
                text_to_log += f'### Sample {i + 1}\\n'
                text_to_log += f'**Prompt:**\\n```\\n{prompts[i]}\\n```\\n\\n'
                text_to_log += f'**Generated Text:**\\n```\\n{generated_texts[i]}\\n```\\n\\n---\\n'

            self.tb_writer.add_text(
                'eval/generated_samples',
                text_to_log,
                state.global_step,
            )
            log_info('Logged %d text samples to TensorBoard.', self.args.num_samples_to_log)
        else:
            log_info('TensorBoard writer not available, skipping TensorBoard logging.')
