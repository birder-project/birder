# NaFlex Multi-Scale Training

`--naflex-sizes` configures one or more square-equivalent resolution budgets while keeping the image batch size fixed. With multiple budgets, one is selected uniformly for each batch and applied to every image in that batch.

Without this option, NaFlex uses one budget derived from the resolved `--size`, whether that size was supplied explicitly or came from the model signature. A single `--naflex-sizes` value also uses the normal static pipeline; the coordinated multi-scale pipeline is activated only when more than one budget is configured.

The option affects training only. Validation always uses the budget derived from `--size`.

## Pixel Sizes and Token Budgets

A configured `--naflex-sizes` value is a square-equivalent area budget, not an output height or width. It is converted once the model patch size is known:

```text
max_seq_len = (size / patch_size)²
```

For example, size 192 with 16-pixel patches allows at most 144 patch tokens. Each image still gets a patch grid matching its source aspect ratio as closely as possible, with `grid_h * grid_w <= max_seq_len`.

When `--naflex-sizes` is omitted, a resolved image size `(height, width)` produces one budget:

```text
max_seq_len = (height / patch_size) * (width / patch_size)
```

Internally, training works with the derived sequence lengths.

## Multi-Scale Data Flow

The following processing applies only when multiple sequence lengths are configured:

```text
       training seed + epoch + worker batch
                        |
                        v
          NaFlexSequenceLengthSchedule
                 one max_seq_len
                        |
             +----------+----------+
             |                     |
      map-style dataset        WebDataset stream
       batch of indices        decoded samples
             |                     |
       __getitems__             iter_batches
      load + transform        group + transform
       one at a time           one at a time
             |                     |
             +----------+----------+
                        |
                  base_collator
    patchify / pad / mask / targets / optional MixUp
                        |
                      model
```

`NaFlexSequenceLengthSchedule` selects the budget. `NaFlexBatchProcessor` tracks batch position, applies the corresponding transform and delegates final collation to `base_collator`. With one sequence length, the dataset applies the single transform normally and the standard DataLoader batching path remains unchanged.

Map-style datasets are wrapped by `NaFlexMultiScaleDataset`. PyTorch passes the wrapper a complete batch of indices through `__getitems__`; the wrapper lazily calls the underlying dataset and transforms each image before loading the next one.

WebDataset has no indices, so `NaFlexBatchProcessor.iter_batches` owns the batch boundary. Normal WebLoader batching is disabled for this path.

Both paths transform immediately to avoid retaining a complete batch of decoded, full-resolution images.

## Distributed Multi-Scale Schedule

No collective communication is used to synchronize sizes. Corresponding batches independently make the same deterministic selection:

```text
global_batch_idx = worker_batch_idx * num_workers + worker_id
batch_seed = training_seed + epoch * EPOCH_SEED_STRIDE + global_batch_idx
```

`EPOCH_SEED_STRIDE` is 1,000,003, so its epoch seed ranges remain distinct while an epoch has fewer than that many global batches.

This requires every rank to use the same training seed, active worker count and ordered batch delivery. For WebDataset, the loader limits workers to the available shards per rank and rejects configurations with fewer shards than ranks.

Persistent workers observe the epoch through shared state. A WebDataset iterator snapshots its epoch so stale prefetched work cannot switch schedules. Virtual epochs keep one continuous iterator and do not reset the schedule.

Synchronization aligns only the selected maximum budget. Different source aspect ratios can still produce different local sequence lengths and padding.
