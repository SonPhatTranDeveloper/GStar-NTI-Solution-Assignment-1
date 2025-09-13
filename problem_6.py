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
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel template for Sliding Window Attention (SWA) with GQA.
    """
    # 1. Boilerplate setup
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- STUDENT IMPLEMENTATION REQUIRED (Part 1: GQA Logic) ---
    # This problem combines GQA and SWA. First, implement the GQA logic.
    # 1. Calculate the number of query heads per group.
    # 2. Determine the correct kv_head_idx for the current q_head_idx.
    q_heads_per_kv = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // q_heads_per_kv
    # --- END OF GQA IMPLEMENTATION ---


    # 2. Initialize accumulators
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load query block
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504

    # --- STUDENT IMPLEMENTATION REQUIRED (Part 2: SWA Logic) ---
    # Now, implement the "sliding window" by changing the loop bounds.
    # The kernel should only attend to the `WINDOW_SIZE` most recent key/value tokens.
    # 1. Calculate the starting position of the attention window (window_start).
    # 2. Modify the range of the Phase 1 loop to start from your window_start.

    diag_start = q_block_idx * BLOCK_M
    window_seq_start = diag_start - (WINDOW_SIZE - 1)
    if window_seq_start < 0:
        window_seq_start = 0
    window_start = (window_seq_start // BLOCK_N) * BLOCK_N

    # Case q_block
    q_block = tl.cast(q_block, tl.float32)

    # Define NEG_INF
    NEG_INF = -1e9  # use a large finite negative instead of -inf

    # Calculate q_mask
    q_mask = q_offsets < SEQ_LEN

    # --- Phase 1: Off-Diagonal Blocks (within the window) ---
    for start_n in range(window_start, q_block_idx * BLOCK_M, BLOCK_N):
        # STUDENT IMPLEMENTATION REQUIRED (Part 3: SWA Logic)
        # Hint: You might need to apply the per-element sliding window mask to s_ij.
        #    - A score is invalid if `(query_offset - key_offset) >= WINDOW_SIZE`.
        k_offsets = (start_n + tl.arange(0, BLOCK_N))
        k_mask = k_offsets < SEQ_LEN

        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])

        # Load and cast K and V to fp32
        k_block = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        v_block = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        k_block = tl.cast(k_block, tl.float32)
        v_block = tl.cast(v_block, tl.float32)

        # Compute raw scores: Q @ K^T  -> (BLOCK_M, BLOCK_N)
        s_ij = tl.dot(q_block, tl.trans(k_block)) * qk_scale

        # Sliding-window & causality mask
        q_pos = q_offsets[:, None]
        k_pos = k_offsets[None, :]
        valid_mask = (q_pos >= k_pos) & ((q_pos - k_pos) < WINDOW_SIZE) & (q_mask[:, None]) & (k_mask[None, :])

        # IMPORTANT: set invalid scores to large finite NEG_INF (not -inf) to avoid nan in reductions
        s_ij = tl.where(valid_mask, s_ij, NEG_INF)

        # Online softmax update (stable)
        # m_ij is finite (>= NEG_INF). If entire row is invalid, m_ij == NEG_INF.
        m_ij = tl.max(s_ij, axis=1)              # (BLOCK_M,)
        m_new = tl.maximum(m_i, m_ij)            # (BLOCK_M,)

        # compute exp safely: subtract finite m_new
        exp_term = tl.exp2(s_ij - m_new[:, None])  # (BLOCK_M, BLOCK_N)
        # But exp_term rows will be exp(NEG_INF - m_new) => ~0 for fully-masked rows
        p_ij_rowsum = tl.sum(exp_term, axis=1)    # (BLOCK_M,)

        # rescale previous accumulators
        scale = tl.exp2(m_i - m_new)               # (BLOCK_M,)
        l_i = scale * l_i + p_ij_rowsum           # (BLOCK_M,)
        acc = scale[:, None] * acc + tl.dot(exp_term, v_block)
        m_i = m_new
     # --- END OF SWA IMPLEMENTATION ---

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        k_offsets = (start_n + tl.arange(0, BLOCK_N))
        k_mask = k_offsets < SEQ_LEN

        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])

        k_block = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        v_block = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        k_block = tl.cast(k_block, tl.float32)
        v_block = tl.cast(v_block, tl.float32)

        s_ij = tl.dot(q_block, tl.trans(k_block)) * qk_scale

        q_pos = q_offsets[:, None]
        k_pos = k_offsets[None, :]
        valid_mask = (q_pos >= k_pos) & ((q_pos - k_pos) < WINDOW_SIZE) & (q_mask[:, None]) & (k_mask[None, :])
        s_ij = tl.where(valid_mask, s_ij, NEG_INF)

        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        exp_term = tl.exp2(s_ij - m_new[:, None])
        p_ij_rowsum = tl.sum(exp_term, axis=1)

        scale = tl.exp2(m_i - m_new)
        l_i = scale * l_i + p_ij_rowsum
        acc = scale[:, None] * acc + tl.dot(exp_term, v_block)
        m_i = m_new
    # --- END OF SWA IMPLEMENTATION ---


    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128):
    
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel.
    """
    batch, n_q_heads, seq_len, head_dim = q.shape
    n_kv_heads = k.shape[1]
    
    assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"
    assert is_causal, "This kernel is only supported for causal attention"

    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    # if window_size != 4096:
    #     raise ValueError("This kernel is compiled for a fixed window size of 4096")

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
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o