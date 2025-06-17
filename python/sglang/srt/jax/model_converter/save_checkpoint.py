import time
import jax

import orbax.checkpoint
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager

import grain.python as grain

from sglang.srt.jax.model_converter import check_pointing
from sglang.srt.jax.model_converter import converter_logging
# Placeholder: internal

# pylint: disable=too-many-positional-arguments

EPS = 1e-8
_DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3


def save_checkpoint(
    checkpoint_manager,
    step,
    state,
    dataset_type="tfds",
    data_iterator=None,
    config=None,
    force=False,
) -> bool:
  """Wrapper for saving checkpoint."""
  if config and config.enable_checkpointing:
    if (
        force
        or (step % config.checkpoint_period == 0)
        or (config.enable_emergency_checkpoint and step % config.local_checkpoint_period == 0)
    ):
      blocking_until_ready_start = time.time()
      converter_logging.log(f"Waiting for step {step} to finish before checkpoint...")
      # We block here on the step finishing so that our checkpointing metrics
      # measure only checkpointing time, not training time.
      jax.block_until_ready(state)
      converter_logging.log(
          f"Waited {time.time() - blocking_until_ready_start} seconds for step "
          f"{step} to finish before starting checkpointing."
      )

  # specify chunk_byte_size to force orbax to control maximum file size in checkpoint
  chunk_byte_size = _DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
  if config:
    chunk_byte_size = config.checkpoint_storage_target_data_file_size_bytes
  save_args = jax.tree.map(lambda _: orbax.checkpoint.SaveArgs(chunk_byte_size=chunk_byte_size), state)

  if isinstance(
      checkpoint_manager,
      (
          emergency_checkpoint_manager.CheckpointManager,
          emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager,
      ),
  ):
    check_pointing.replicator_error_handler(config)
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            state=orbax.checkpoint.args.PyTreeSave(
                item=state,
                save_args=save_args,
                ocdbt_target_data_file_size=chunk_byte_size,
            )
        ),
        force=force,
    )

  if dataset_type == "grain":
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
            ),
            iter=grain.PyGrainCheckpointSave(data_iterator.local_iterator),
        ),
        force=force,
    )
  else:
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
            )
        ),
        force=force,
    )