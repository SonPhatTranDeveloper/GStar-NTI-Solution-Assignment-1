import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504

    # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
    # Combine the GQA, SWA, and Sink logic.
    # Combine all code from previous problems, and add the sink logic.
    # You should have 3 phases:
    # 1. Phase 0: Sink blocks that are before the sliding window
    # 2. Phase 1: Off-Diagonal Blocks (within the window)
    # 3. Phase 2: Diagonal Blocks
    # Calculate q_mask
    q_mask = q_offsets < SEQ_LEN

    # Define NEG_INF
    NEG_INF = -1e9

    # Convert q_block to f32
    q_block = tl.cast(q_block, tl.float32)

    # Pointers to K and V blocks
    k_ptrs_base = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h
    v_ptrs_base = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h
    
    # One unified loop for all key blocks. The masking logic inside the loop
    # will handle causality, sliding window, and the attention sink.
    num_k_blocks = tl.cdiv(SEQ_LEN, BLOCK_N)
    for k_block_idx in range(0, num_k_blocks):
        k_offsets = k_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        k_mask = k_offsets < SEQ_LEN
        
        # Load and cast K and V to fp32
        k_ptrs = k_ptrs_base + (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_ptrs = v_ptrs_base + (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        
        k_block = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        v_block = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        k_block = tl.cast(k_block, tl.float32)
        v_block = tl.cast(v_block, tl.float32)

        # Compute raw scores: Q @ K^T  -> (BLOCK_M, BLOCK_N)
        s_ij = tl.dot(q_block, tl.trans(k_block)) * qk_scale

        # Combined mask for causality, sliding window, and attention sink
        q_pos = q_offsets[:, None]
        k_pos = k_offsets[None, :]
        
        causal_mask = q_pos >= k_pos
        swa_mask = (q_pos - k_pos) < WINDOW_SIZE
        sink_mask = k_pos < SINK_SIZE
        
        # Final valid mask: Causal AND (Sliding Window OR Sink)
        valid_mask = causal_mask & (swa_mask | sink_mask) & (q_mask[:, None]) & (k_mask[None, :])
        
        # Set invalid scores to a large finite negative to avoid NaNs
        s_ij = tl.where(valid_mask, s_ij, NEG_INF)

        # Online softmax update
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        exp_term = tl.exp2(s_ij - m_new[:, None])
        p_ij_rowsum = tl.sum(exp_term, axis=1)

        scale = tl.exp2(m_i - m_new)
        l_i = scale * l_i + p_ij_rowsum
        acc = scale[:, None] * acc + tl.dot(exp_term, v_block)
        m_i = m_new
    # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128, sink_size=4):
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel with attention sink support.
    """
    # Shape checks
    batch, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape
    
    # Assertions
    assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3]
    assert k.shape == v.shape
    assert head_dim <= 128
    assert n_q_heads % n_kv_heads == 0
    assert is_causal, "This kernel only supports causal attention"
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_swa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        WINDOW_SIZE=window_size,
        SINK_SIZE=sink_size,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o