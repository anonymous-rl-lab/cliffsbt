# Round 13 | Paired boundary transport in a second neural-network domain

## Decision

**Official decision: `MECHANISM_DOMAIN_CONFIRMED` (12/12 frozen gates).**

Round 13 confirms the Cliff formation law in an official CIFAR-10-C image
corruption benchmark using three independently trained convolutional networks,
15 standard corruption families, five ordered severities, and all 10,000 paired
test identities. The result is a second sample-paired neural-network mechanism
domain after TorchSig. It is not yet the intended real-world CURE-TSR result.

## Staged evidence

| Stage | Design | Result | Licensed decision |
|---|---|---:|---|
| 13A | CURE-TSR file and identity eligibility | Data link unavailable; pairing undocumented | Hold CURE claims |
| 13B | One seed, 3 controlled paths, 2,000 paired identities | 8/8 gates | Advance |
| 13C | 3 fresh seeds, 6 controlled paths, 10,000 identities | 9/9 gates | Advance to official benchmark |
| 13D | Official CIFAR-10-C, 3 seeds, 15 families, 5 severities | **12/12 gates** | **Mechanism domain confirmed** |
| 13E | 3 fresh seeds × 4 repair arms, equal 5% budget, 8,000 sealed identities | **6/9 gates** | **Coverage-only dominance rejected** |

The official archive passed its published MD5 checksum
`56bf5dcef84df0e2308c6dcbcbbd8499`. A pretarget schema correction records that
the archive repeats its 10,000 labels in five severity blocks; no model output
had been generated when this correction was made.

## Primary official results

| Quantity | Estimate | Training-seed cluster 95% CI |
|---|---:|---:|
| Endpoint risk increase | 0.26858 | [0.25089, 0.27885] |
| Endpoint persistence among crossing cells | 0.88056 | [0.87349, 0.88543] |
| Normalized first-crossing entropy | 0.91901 | [0.90417, 0.93170] |
| Prior true-margin separation | 1.23488 | [1.16775, 1.27827] |
| Placebo-minus-true accounting RMSE | 0.07487 | [0.06846, 0.07882] |

- maximum paired accounting error: `9.72e-17`;
- 15/15 families increased endpoint risk for all three training seeds;
- 11/15 families exhausted headroom for all three seeds;
- pixelation crossed for two of three seeds;
- brightness, elastic transform, and JPEG compression crossed for zero seeds;
- median cliff levels occupied four distinct severity levels;
- all declared first crossings occurred exactly when cumulative net flux reached
  the frozen risk headroom.

## Mechanistic interpretation

### 1. Risk growth is not a Cliff

All 15 corruption families raised endpoint error, yet only 11 crossed the
operational boundary reproducibly. Brightness increased risk by roughly
0.10--0.11 without exhausting the 0.15 headroom; elastic transform increased
risk by roughly 0.12 but had substantial recovery flux; JPEG compression raised
risk by only 0.04--0.07. These are within-domain negative controls showing that
distributional degradation and even positive risk drift do not imply a Cliff.

### 2. Net flux, not incident failure count, is the conserved object

Several paths showed large recovery. Total recovery across five steps was about
0.42 for glass blur, 0.41--0.44 for snow, and 0.41--0.42 for elastic transform.
Glass blur and snow could still cross because incident flow was larger; elastic
transform did not because recovery prevented net transport from exhausting
headroom. This directly distinguishes net boundary transport from raw failure
hits.

### 3. Synchronization is not required

The normalized entropy of first-crossing severity was 0.919, while cliff levels
varied across corruption families and training seeds. Risk accumulation was
persistent and distributed rather than a single synchronized pulse, matching
the mechanism retained after the failed TorchSig synchronization hypothesis.

### 4. The relevant boundary is the trained task boundary

The exact correct-to-error minus error-to-correct accounting reconstructed every
adjacent risk increment to floating-point precision. A fixed incorrect
pseudo-label boundary failed by an additional RMSE of 0.07487. Thus the result
is tied to transport through the model's task boundary, not arbitrary logit
motion.

## What Round 13 establishes

Round 13 supports the following chain in a convolutional image classifier:

```text
fixed training support -> trained decision regions
ordered deployment corruption -> heterogeneous signed boundary transport
cumulative incident flow - recovery flow -> risk-headroom depletion
headroom exhausted -> operational Cliff
headroom retained -> degradation without a Cliff
```

The result strengthens the transport law beyond a single signal-processing
system and beyond non-neural tree models. Round 13E additionally establishes
that training intervention can reduce the same held-out boundary flux and
operational crossings in this image domain. It does not establish natural
deployment drift, CURE-TSR cross-level identity, or warning performance.

## Round 13E repair boundary

Round 13E compared clean-only training, random official-domain augmentation,
deep first-crossing concentration, and broad first-crossing coverage with the
same 1,000-image budget. Hazard and coverage both had 100% confirmed dangerous
hits. Coverage occupied 731 corruption × class × first-severity fragments versus
262 for hazard, but had lower mean boundary-pressure score (1.40 versus 2.55).

The held-out endpoint-error ordering was hazard (0.54172), coverage (0.55494),
random (0.56257), and baseline (0.60235). Coverage minus hazard was +0.01322
with a training-seed cluster 95% CI of [+0.01041, +0.01611]; risk-area and
crossing-fraction gates failed in the same direction. Coverage nevertheless beat
baseline by −0.04741 [−0.05912, −0.03596] and beat random by −0.00763
[−0.00882, −0.00697]. The maximum paired accounting error remained below
9.72e−17.

The licensed repair law is therefore not “coverage alone.” Coverage supplies a
support floor; boundary pressure, deployment mass, redundancy, and transfer
determine allocation leverage once that floor is present. In this experiment the
hazard arm already spanned all 15 families, 10 classes, and five severities, so
spending the fixed budget on many additional low-pressure fragments reduced
efficiency.

See `ROUND13E_REPAIR_REPORT.md` for the complete frozen design, gates, and
interpretation.

## Next licensed experiment

The next repair study should use a frozen 2×2 design that independently matches
fragment breadth and hazard-score distribution, then add deployment-mass and net-
flux weighting. This can distinguish crossing depth from flow mass and transfer
without rewriting the negative Round 13E result.

CURE-TSR remains the preferred real-world extension once its licensed files are
available and the cross-severity identity audit passes.
