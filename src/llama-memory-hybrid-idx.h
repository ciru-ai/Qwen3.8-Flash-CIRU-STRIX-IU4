#pragma once

#include "llama-memory-hybrid.h"

#include <memory>
#include <unordered_map>
#include <vector>

//
// llama_memory_hybrid_idx
//

// Derived QSA cache: one normalized/rotated F16 key for each completed
// four-token compression block.  The raw indexer KV remains the state-save
// authority; this cache is an append-only p1 acceleration and is invalidated
// on any non-append sequence operation.
class llama_qsa_block_cache {
public:
    llama_qsa_block_cache(
            const llama_model & model,
                     uint32_t   key_dim,
                     uint32_t   kv_size,
                         bool   offload,
      const llama_memory_i::layer_filter_cb & filter);

    uint32_t capacity() const;
    uint32_t valid_blocks = 0;
    int32_t valid_base = -1;
    llama_seq_id valid_seq = -1;

    ggml_tensor * get(ggml_context * ctx, int32_t il, uint32_t n_blocks) const;
    ggml_tensor * get_range(ggml_context * ctx, int32_t il, uint32_t first, uint32_t count) const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct layer {
        uint32_t il;
        ggml_tensor * k;
    };

    const uint32_t key_dim;
    const uint32_t n_blocks;

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;
    std::vector<layer> layers;
    std::unordered_map<int32_t, int32_t> map_layer_ids;
};

struct llama_qsa_block_plan {
    bool enabled = false;
    bool sparse_attention = false;
    uint32_t ratio = 0;
    uint32_t new_blocks = 0;
    uint32_t total_blocks = 0;
    int32_t cell_base = -1;
    uint32_t first_block = 0;
    llama_seq_id seq_id = -1;
};

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias) const;
    llama_qsa_block_cache * get_mem_idx_blocks() const;

    llama_qsa_block_plan plan_qsa_blocks(
            const llama_ubatch & ubatch,
            const llama_kv_cache::slot_info & sinfo,
            uint32_t ratio);
    llama_qsa_block_plan plan_qsa_cached_pool(
            const llama_ubatch & ubatch,
            const llama_kv_cache::slot_info & sinfo, uint32_t ratio);

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;
    const std::unique_ptr<llama_qsa_block_cache> mem_idx_blocks;

    bool qsa_blocks_fast = true;
    int32_t qsa_blocks_cell_base = -1;

    void reset_qsa_blocks();
    void invalidate_qsa_blocks();
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;
    llama_qsa_block_cache * get_idx_blocks() const;

    llama_qsa_block_plan plan_qsa_blocks(const llama_ubatch & ubatch, uint32_t ratio) const;
    llama_qsa_block_plan plan_qsa_cached_pool(const llama_ubatch & ubatch, uint32_t ratio) const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias) const;

    // H102 append-only p1 path.  All work is O(new blocks + selected-map
    // metadata), with no scan over historical cache cells and no dense bias.
    void set_input_qsa_blocks(
            ggml_tensor * new_blk_cells,
            ggml_tensor * new_blk_pos,
            ggml_tensor * new_blk_ids,
            const llama_ubatch * ubatch,
            const llama_qsa_block_plan & plan) const;

private:
    llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
