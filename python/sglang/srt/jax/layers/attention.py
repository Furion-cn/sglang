import jax
import jax.numpy as jnp
from flax import nnx


def unpack_tokens_to_batch_vectorized(inputs: jax.Array, seq_lengths: jax.Array, pad_value: float = 0.0) -> jax.Array:
    """
    unpack tokens to batch vectorized
    
    Args:
        inputs: input tokens, the shape is (total_tokens, hidden_dim)
        seq_lengths: the shape is (batch_size,), the length of each sequence
        pad_value: padding value, default is 0.0

    Returns:
        the shape is (batch_size, max_seq_len, hidden_dim)
    """
    batch_size = seq_lengths.shape[0]
    max_seq_len = jnp.max(seq_lengths)
    hidden_dim = inputs.shape[-1]
    total_tokens = inputs.shape[0]

    # create output array, initialized with pad_value
    output = jnp.full((batch_size, max_seq_len, hidden_dim),
                      pad_value, dtype=inputs.dtype)

    # create cumulative indices to determine the start position of each sequence in the original tokens
    start_loc = jnp.concatenate(
        [jnp.array([0]), jnp.cumsum(seq_lengths[:-1])])

    # create index matrix
    pos_indices = jnp.arange(max_seq_len)[None, :]   # (1, max_seq_len)

    # calculate the global index in the original tokens
    global_indices = start_loc[:, None] + \
        pos_indices  # (batch_size, max_seq_len)

    # create mask to determine which positions are valid (not exceeding sequence length)
    # (batch_size, max_seq_len)
    valid_mask = pos_indices < seq_lengths[:, None]

    # ensure the index does not exceed the bounds
    global_indices = jnp.clip(global_indices, 0, total_tokens - 1)

    # extract tokens
    # (batch_size, max_seq_len, hidden_dim)
    extracted_tokens = inputs[global_indices]

    # apply mask, invalid positions are set to pad_value
    output = jnp.where(valid_mask[..., None], extracted_tokens, pad_value)

    return output


def pack_batch_to_tokens_vectorized(batch_inputs: jax.Array, seq_lengths: jax.Array, pad_value: float = 0.0) -> jax.Array:
    """
    向量化版本：将带padding的batch格式重新打包为连续的tokens
    这是unpack_tokens_to_batch_vectorized的逆操作
    
    Args:
        batch_inputs: 形状为 (batch_size, max_seq_len, hidden_dim) 的数组
        seq_lengths: 形状为 (batch_size,) 的数组，表示每个序列的长度
        pad_value: padding的值，用于识别和过滤padding位置
    
    Returns:
        tuple containing:
        - packed_tokens: 形状为 (total_tokens, hidden_dim) 的数组
        - start_loc: 形状为 (batch_size,) 的数组，表示每个序列在packed_tokens中的起始位置
    """
    batch_size, max_seq_len, hidden_dim = batch_inputs.shape

    # 计算每个序列在packed输出中的起始位置
    start_loc = jnp.concatenate([jnp.array([0]), jnp.cumsum(seq_lengths[:-1])])
    total_tokens = jnp.sum(seq_lengths)

    # 创建位置索引矩阵
    pos_indices = jnp.arange(max_seq_len)[None, :]   # (1, max_seq_len)

    # 创建mask来确定哪些位置是有效的（不是padding）
    # (batch_size, max_seq_len)
    valid_mask = pos_indices < seq_lengths[:, None]

    # 方法1: 使用where和reshape来提取有效tokens
    # 为每个有效位置分配在输出数组中的索引
    batch_indices = jnp.arange(batch_size)[:, None]  # (batch_size, 1)
    seq_indices = jnp.arange(max_seq_len)[None, :]    # (1, max_seq_len)

    # 计算每个位置在packed数组中的目标索引
    target_indices = start_loc[:, None] + \
        seq_indices  # (batch_size, max_seq_len)

    # 初始化输出数组
    packed_tokens = jnp.zeros(
        (total_tokens, hidden_dim), dtype=batch_inputs.dtype)

    # 使用scatter来填充有效的tokens
    # 需要将三维索引展平
    valid_positions = jnp.where(valid_mask)  # 返回有效位置的索引
    valid_batch_idx, valid_seq_idx = valid_positions

    # 计算对应的目标索引
    target_idx = start_loc[valid_batch_idx] + valid_seq_idx

    # 提取有效的tokens
    # (num_valid_tokens, hidden_dim)
    valid_tokens = batch_inputs[valid_batch_idx, valid_seq_idx]

    # 使用at[].set()来填充packed_tokens
    packed_tokens = packed_tokens.at[target_idx].set(valid_tokens)

    return packed_tokens

class Attention(nnx.Module):
    """attention layer."""

    def __init__(self,
                 num_heads: int,    
                 scale: float = None,
                 rngs: nnx.Rngs = None):
        self.scale = scale
        self.num_heads = num_heads

    def __call__(self,
                 q: jax.Array,
                 k: jax.Array,
                 v: jax.Array,
                 attention_mask: jax.Array = None,
                 is_causal: bool = True,
                 q_seq_lengths: jax.Array = None,
                 kv_seq_lengths: jax.Array = None):

        # Reshape to [seq_len, num_heads, head_dim] for attention
        seq_len = q.shape[0]
        total_dim = q.shape[-1]
        q_ = unpack_tokens_to_batch_vectorized(q, q_seq_lengths)
        k_ = unpack_tokens_to_batch_vectorized(k, kv_seq_lengths)
        v_ = unpack_tokens_to_batch_vectorized(v, kv_seq_lengths)

        q_attn = q_.reshape(*q_.shape[:2], self.num_heads, -1)
        k_attn = k_.reshape(*k_.shape[:2], self.num_heads, -1)
        v_attn = v_.reshape(*v_.shape[:2], self.num_heads, -1)
        
        attn_output = jax.nn.dot_product_attention(
            q_attn,
            k_attn,
            v_attn,
            mask= attention_mask,
            is_causal=is_causal,
            scale=self.scale,
            query_seq_lengths=q_seq_lengths,
            key_value_seq_lengths=kv_seq_lengths,
        )
        attn_output_reshaped = attn_output.reshape(*attn_output.shape[:2], -1)
        attn_output = pack_batch_to_tokens_vectorized(
            attn_output_reshaped, q_seq_lengths)
        return attn_output

    def _attn(self, q, k, v, attention_mask, is_causal):
        if self.scale is None:
            scale = 1.0 / jnp.sqrt(q.shape[-1])
        else:
            scale = self.scale

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # query-key product
        attn_weights = jnp.einsum("bnqh,bnkh->bnqk", q, k)

        # scale
        attn_weights = attn_weights * scale

        # apply causal mask
        if is_causal:
            causal_mask = jnp.tril(
                jnp.ones((q.shape[2], q.shape[2]), dtype=bool))
            causal_mask = causal_mask[None, None, :, :]
            mask_value = jnp.finfo(attn_weights.dtype).min
            attn_weights = jnp.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Compute attention output: attn_weights @ V
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = jnp.swapaxes(attn_output, 1, 2)

        return attn_output
