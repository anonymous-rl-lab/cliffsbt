# Source-package provenance

The unified package was assembled from the following frozen inputs and aligned
first to manuscript v5, extended for manuscript v6, and then extended to
manuscript v8 by integrating Round 13A--E, and finally aligned to manuscript
v8.2 and then refreshed with the authoritative registered CURE-OR v2 closed-loop
experiment. No retained-domain frozen config, raw result, pretarget evidence, or
terminal decision was changed:

| Artifact | SHA-256 |
|---|---|
| `Cliff_Covertype_mechanism_v1.zip` | `c39dc8e53c311b17b6a28b312af15c64b67010e4724e9992050ece99d99f8800` |
| `cliff_repo_v1_round11c_official_source_flux.zip` | `b394d898c18de6882a0cd3a9993fcceeb651e3250e47690ac4feef9cd59c958c` |
| `Cliff_main_v5.zip` | `14a23d3b01f8b6abb5f8703bd690280586cc8bc348b2c051c2f3a0936327a908` |
| `Cliff_round12c_coverage_vs_hazard_pilot_v1.zip` | `938cbb7e15389c55e8376abcf4563d5b75bfac24146b83bcf36fc140bc28a843` |
| `Cliff_boundary_transport_code_v3.zip` | `188dde2ce0a69ce4d46c65340e4d2a41c74f56c61019fb923d8e9bb8f8b88bd1` |
| `Cliff_Round13_CIFAR10C_mechanism_and_repair_v1.zip` | `1c748e0c864eb3fb5e93cbbb8521861d0255a335dbf7071685a3d49c1614cf8b` |
| `Cliff_boundary_transport_code_v4.zip` | `d2879bd9a4e56e8df2bd525813c0bd3c1e5831344fbcb94dc50525c318cfba2b` |
| `Cliff_boundary_transport_code_v5.zip` integration base | `453cb7aabaa4d87e07bc80515fae465714608a3f835637158eb16d515dadc088` |
| `CURE_OR_V2_COMPLETE_REPRODUCIBILITY_c6ygf.zip` | `e3c3508eb800859baace03f6d1259cde6c4d49fdc453a902fc91d61bd196d143` |

The v6 extension adds the Round 12C protocol, runner, tests, raw CSV/JSON
results, retained pre-evaluation technical aborts, report, and publication
figure, plus cross-experiment documentation and audits. The Covertype frozen
manifest remains byte-valid. TorchSig and package-root manifests are regenerated
because documentation and a post-result figure were added; the independent
Round 11C and Round 12C pretarget release hashes and raw result manifests remain
the evidentiary controls.

The v4 extension adds `round13_second_domain/` as a first-class directory. The
source Round 13 archive expands to one unique path tree and collides with no v3
path. Its 103-entry internal SHA-256 manifest remains valid after relocation.
Root documentation, the unified audit, and the package-level manifest are
updated; Round 13 protocols, snapshots, code, tests, models, raw paired arrays,
selection records, figures, independent audits, and terminal reports remain
byte-identical to the source archive.

The v6 extension replaces the `cure_or/` tree with the complete
`CURE_OR_V2_COMPLETE_REPRODUCIBILITY_c6ygf.zip` contents: frozen runtime code,
configuration, official metadata and input tables, derived feature tensors,
outcome-blind predictions/telemetry, Phase 2 result tables, execution audits,
and scientific documentation. The source ZIP is not nested. The only
integration-local module change makes the strict audit emit parseable JSON;
scientific code, frozen parameters, arrays, tabular results, gates, and the
Phase 1 blind commitment are unchanged. Module and repository manifests are
rebuilt after integration.
