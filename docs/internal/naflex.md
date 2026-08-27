# NaFlex Training Decisions

This document records the NaFlex training decisions that are not obvious from the implementation.

## Batch Specifications and Sampling

A batch specification is `(patch_size, max_seq_len)`. Let `P0` be the model's canonical patch size and `A` one of the
configured resolution-area budgets (`R^2` for a square-equivalent resolution, or `height * width` for a rectangular
budget). The natural-patch sequence ceiling is:

```text
L_cap = max(floor(A / P0^2))
```

For every candidate patch size `P` and resolution budget `A`, we form:

```text
L(A, P) = floor(A / P^2)
```

The pair is admissible exactly when:

```text
0 < L(A, P) <= L_cap
```

This excludes the expensive small-patch/large-resolution corner without allowing the choice of patch size to increase
the maximum natural-patch workload. Every requested patch size and every requested resolution must participate in at
least one admissible pair, otherwise configuration fails.

Sampling is hierarchical rather than uniform over pairs:

1. Select a patch size uniformly.
2. Select uniformly from its admissible resolution entries.
3. Apply the resulting specification to the entire batch.

Resolution entries are retained when two resolutions quantize to the same specification. The transform is shared, but
the duplicate schedule entries preserve the resolutions' sampling weight. Coordinated scheduling is needed only when
more than one distinct specification remains.

`max_seq_len` is a ceiling used by the image transform, not the padded sequence length. Each transformed image has an
aspect-ratio-preserving patch grid with `grid_h * grid_w <= max_seq_len`, and the collator pads only to the longest
actual sequence in the local batch. Consequently, padded sequence lengths can be smaller than the selected ceiling and
can differ across distributed ranks.

## Fixed Image Batch Size

The image batch size does not grow when a specification produces a shorter sequence. Consequently, every uniformly
sampled patch size has the same expected number of images contributing to training. Adapting batch size to sequence
length would give short-sequence choices more training contribution. The resulting underutilization on short sequences
is intentional.

## Regional MixUp

NaFlex MixUp operates on patch grids before padding. It is eligible only for batches with more than one sample and is
applied with probability `p`, which defaults to `0.5`. When it is applied:

- Draw one `lambda ~ Beta(alpha, alpha)` for the batch.
- Pair sample `i` with the original sample `(i - 1) mod batch_size`.
- Take the largest axis-aligned rectangle shared by their grid shapes, and independently sample its source and
  destination offsets.
- Replace the destination region with `lambda * destination + (1 - lambda) * source`.

The recipient grid is not resized. Its effective target weight accounts for the fraction of recipient patches touched:

```text
overlap_fraction = overlap_area / recipient_area
effective_lambda = 1 - (1 - lambda) * overlap_fraction
target_i = effective_lambda * target_i + (1 - effective_lambda) * target_(i-1)
```

The mixed sequences are padded only after this operation.

## Deterministic Distributed Schedule

Each batch selection uses a fresh generator seeded with:

```text
global_batch_idx = worker_batch_idx * num_workers + worker_id
batch_seed = training_seed + epoch * 1_000_003 + global_batch_idx
```

Despite its name, `global_batch_idx` is global only across the DataLoader workers within one rank, it is not a
world-global batch index. Corresponding ranks independently reconstruct the same logical index.

`1_000_003` serves as the epoch seed stride. For ordinary epochs, seed ranges do not overlap provided every used
`global_batch_idx` is less than `1_000_003`. An index at or beyond that stride can produce the same seed as a batch in a
later epoch.

This avoids collective communication: ranks select the same specification when they use the same seed, epoch, worker
count, and ordered batch delivery. It synchronizes the specification only, different image aspect ratios can still
produce different local sequence lengths and padding. Virtual epochs keep one continuous iterator and therefore do not
reset this schedule.
