<!-- Cliff NMI Supplementary Information, version v7 -->

# Supplementary Information

## Generalization cliffs in classifiers emerge from persistent signed boundary transport

### Scope and relation to the main text

This Supplementary Information accompanies **Generalization cliffs in classifiers emerge from persistent signed boundary transport**. It preserves the complete theory, qualification studies, negative results, formal protocols and sensitivity analyses supporting the main manuscript. Round identifiers are retained only as provenance labels; scientific headings state the role of each experiment.

The scalar identity

\[
R_{t+1}-R_t=J_t^+-J_t^-\equiv\operatorname{SBT}_t
\]

is exact for a fixed deterministic classifier on paired identities, but it contains no more scalar information than \(\Delta R_t\). The substantive object is the resolved transport ledger: incident and recovery mass, gross turnover, crossing identities and fragments, first-crossing times, path-conditioned persistence and response to intervention. Formation accounting is exact; local geometry requires smoothness; warning is conditional on telemetry, temporal information, calibration, horizon and effective population; and repair allocation remains empirically bounded rather than governed by a universal rule.

Version v7 retains all previously committed-output diagnostics and adds focused clarification and sensitivity analyses without changing any registered decision: explicit separation of first crossing from a persistence-confirmed cliff, formal distinction between task-error SBT and an outcome-blind prediction-state transport proxy, an augmented-state definition of dynamic sufficiency, classifier-head-cluster summaries for the nested proxy analysis, false-alarm-budget sensitivity and domain-specific trained-peer effect sizes. These analyses were specified after confirmation reveal and are therefore post hoc, even where fitting and threshold selection use only calibration data. Stopped experiments remain stopped, and pre-target or preregistered conclusions are not relabelled.

## Supplementary Note 1 | Formal theory, resolved transport and claim boundaries


### S1.1 Objects, conditioning, and scope

The theory conditions on a realized trained model

\[
W=\mathcal A_{\mathrm{train}}(D_{\mathrm{train}},\lambda,\xi).
\]

This conditioning is essential. A generalization cliff is not a property of a deployment distribution \(P_t\) alone. It is a property of the interaction among:

1. the decision regions induced by \(D_{\mathrm{train}}\), the training procedure, and the realized training randomness;
2. the deployment law \(P_t\) and its motion over time;
3. the task loss and frozen operational boundary;
4. the observation channel available before outcomes are revealed.

For classification, the exact transport result is stated for balanced zero–one risk. The true-class margin and class-\(y\) error region are

\[
m_W(x,y)=f_{W,y}(x)-\max_{k\ne y}f_{W,k}(x),
\qquad
\mathcal E_{W,y}=\{x:m_W(x,y)\le0\}.
\]

With class-conditional deployment density \(p_{y,t}\),

\[
R_W(t)
=
\frac1K\sum_y\int_{\mathcal E_{W,y}}p_{y,t}(x)\,dx.
\]

The exact theory does not require a density. For any identity-preserved
panel, define

\[
e_{i,t}=\mathbf1\{W(x_{i,t})\ne y_i\},
\qquad
R_t=N^{-1}\sum_i e_{i,t},
\]

\[
J_t^+=N^{-1}\sum_i\mathbf1\{e_{i,t}=0,e_{i,t+1}=1\},
\qquad
J_t^-=N^{-1}\sum_i\mathbf1\{e_{i,t}=1,e_{i,t+1}=0\}.
\]

Then, identically,

\[
\boxed{
\Delta R_t
=J_t^+-J_t^-
\equiv\operatorname{SBT}_t
}.
\tag{SBT1}
\]

The four possibilities for each binary error indicator prove the result pointwise. The scalar quantity \(\operatorname{SBT}_t\) is therefore exactly \(\Delta R_t\); no additional scalar predictive information is claimed from renaming the increment. One convenient schematic representation of the richer resolved ledger used throughout the experiments is

\[
\mathcal L_t
=
\left(J_t^+,J_t^-,G_t,\mathcal I_t^+,\mathcal I_t^-,\mathcal F_t,\mathcal T_t,\mathcal P_t\right),
\]

where \(G_t=J_t^++J_t^-\) is gross turnover, \(\mathcal I_t^+\) and \(\mathcal I_t^-\) are the incident and recovery identities, \(\mathcal F_t\) records declared boundary fragments, \(\mathcal T_t\) records first-crossing times, and \(\mathcal P_t\) records persistence. The exact components satisfy

\[
J_t^+=\frac{G_t+\operatorname{SBT}_t}{2},
\qquad
J_t^-=\frac{G_t-\operatorname{SBT}_t}{2}.
\tag{SBT1a}
\]

Thus equal net risk increments need not describe equal mechanisms. For example, \((J_t^+,J_t^-)=(0.10,0)\) and \((0.40,0.30)\) both give \(\operatorname{SBT}_t=0.10\), but the second transition has seven times greater gross turnover, substantial recovery and a different susceptible population. The scalar risk curve cannot recover this distinction.

Consequently,

\[
R_T=R_0+\sum_{t=0}^{T-1}\operatorname{SBT}_t,
\qquad
\tau_\beta^{\times}
=\inf\left\{T:
\sum_{t=0}^{T-1}\operatorname{SBT}_t
\ge\beta-R_0\right\}.
\tag{SBT2}
\]

Equations SBT1--SBT2 are valid for every fixed deterministic classifier,
including trees, convolutional networks, or any other architecture with a
well-defined zero–one decision. They do not require smooth scores,
differentiability, a physical degradation coordinate, monotonic risk, or a
neural-network parameterization. For fixed non-negative weights \(w_i\) with \(\sum_iw_i=1\), the same proof yields

\[
R_{t+1}^{w}-R_t^{w}=J_{t,w}^+-J_{t,w}^-.
\tag{SBT2a}
\]

This covers uniform empirical risk, class-balanced risk and fixed cost-sensitive operational risk. Extensions to randomized prediction use the corresponding conditional error probabilities.

Equation SBT2 defines ordinary first crossing \(\tau_\beta^{\times}\). Operational studies additionally use a persistence functional \(\mathcal P_\beta\), fixed before target reveal:

\[
\tau_{\beta,\mathcal P}
=
\inf\{T\ge\tau_\beta^{\times}:\mathcal P_\beta(R_{T:T+q})=1\}.
\tag{SBT2b}
\]

Cumulative SBT determines first crossing exactly; persistence is an additional event-confirmation rule. “Persistent harmful SBT” refers to sustained positive net imbalance over the declared analysis scale, not to every step being positive and not to path-independent persistence of individual errors.

The transport ledger is defined before any operational boundary is chosen. Changing \(\beta\) changes the headroom and first-passage time but does not change \(J_t^+\), \(J_t^-\), gross turnover or the identity of the observed transitions. The operational boundary may be externally meaningful or protocol relative. The mathematical crossing time exists in either case, but only the former carries an engineering or safety interpretation. Every TorchSig boundary in this paper is protocol relative; the frozen-output threshold diagnostics test robustness of the intervention ordering, not the safety meaning of alternative boundaries.

The framework is not restricted to neural networks. Any data-trained predictor with measurable decision regions can instantiate it, including the ExtraTrees models used in the principal TorchSig studies. Conversely, not every trained model or deployment path produces a cliff. If incident mass is absent, directed toward recovery, too weak to exhaust headroom, or cancelled by recovery, no crossing occurs. This is the precise meaning of the paper's central statement: a model does not fail because deployment data are merely far from training data; it fails when sustained probability mass crosses a training-shaped decision boundary in the harmful net direction.

### S1.2 Continuous boundary current is a conditional representation

Assume \(p_{y,t}\) is differentiable in time and satisfies

\[
\partial_t p_{y,t}
+
\nabla\cdot j_{y,t}=0,
\]

where \(j_{y,t}\) is a probability current. In the deterministic-advection case,
\(j_{y,t}=p_{y,t}v_{y,t}\). Differentiating risk for a fixed model gives

\[
\begin{aligned}
\dot R_W(t)
&=
\frac1K\sum_y
\int_{\mathcal E_{W,y}}\partial_t p_{y,t}(x)\,dx\\
&=
-\frac1K\sum_y
\int_{\mathcal E_{W,y}}\nabla\cdot j_{y,t}(x)\,dx\\
&=
-\frac1K\sum_y
\int_{\partial\mathcal E_{W,y}}
j_{y,t}(x)^\top n_{W,y}(x)\,dS\\
&\equiv J_W(t).
\end{aligned}
\]

This representation extends directly from deterministic velocity to any
deployment law with a well-defined probability current. It is not required
for the exact paired law in Eq. SBT1. The experimental TorchSig, CIFAR-10-C and CURE-OR ladders consist of discrete challenge states and are not assumed to satisfy this continuity equation; the current is a conditional interpolation and theoretical extension. For a diffusion with drift \(v\) and
diffusion tensor \(D\),

\[
j=pv-\frac12\nabla\cdot(Dp),
\]

and both drift and diffusion can contribute to boundary crossing. The experiments estimate realized transitions or margin-distribution transport; they do not separately identify these physical terms.

If class priors vary, ordinary error contains an additional prior-shift term. The balanced-risk construction used here removes that term by weighting class-conditional errors equally. With time-dependent weights \(\pi_y(t)\),

\[
\dot R_W(t)
=
\sum_y\dot\pi_y(t)R_{W,y}(t)
+
\sum_y\pi_y(t)J_{W,y}(t).
\]

This decomposition explains why class balancing is necessary for a clean boundary-transport interpretation in the present experiments.

Integrating the current yields

\[
R_W(t)-R_W(t_0)
=
\int_{t_0}^{t}J_W(s)\,ds.
\]

When \(R_W(t_0)<\beta\) and \(R_W\) is continuous, the cliff time is the first exhaustion of risk headroom:

\[
\tau_\beta
=
\inf\left\{
t:
\int_{t_0}^{t}J_W(s)\,ds
\ge
\beta-R_W(t_0)
\right\}.
\]

This equivalence is exact under the stated model. It does not imply that the current is observable, predictable, or causally attributable to a unique deployment mechanism.

### S1.3 What promotes exact accounting to mechanism evidence

Equation SBT1 follows directly for paired samples. Writing

\[
e_{i,t}=\mathbf1\{m_W(x_{i,t},y_i)\le0\}.
\]

Then

\[
e_{i,t+1}-e_{i,t}
=
\mathbf1\{0\to1\}
-
\mathbf1\{1\to0\}.
\]

and summing over \(i\) gives again

\[
R_{t+1}-R_t=\frac{F_t-B_t}{N}.
\]


This identity is true for every finite paired panel and is not, by itself, a scientific discovery. Its non-trivial content must come from quantities not recoverable from the scalar risk curve and from interventions or placebos that can fail. The principal falsifiable consequences are:

| Consequence | Competing explanation | Empirical discriminator |
|---|---|---|
| Unsigned displacement does not determine risk direction | Distance from training support is sufficient | Covertype signed-margin transport versus unsigned hidden-space distance |
| Equal net increments can hide different turnover and recovery | \(\Delta R_t\) alone is a sufficient state | Separate incident, recovery, persistence and fragment ledgers |
| A cliff need not be a synchronized pulse | Boundary-near identities must fail together | First-crossing entropy, short-window share and velocity-shuffle placebo |
| The resolved organization is focal-boundary specific | Random partitions or another trained boundary with similar anchor risk reconstructs the same increments | Complementary fixed-incorrect and closest-risk trained-peer placebos, with normalized peer error |
| Training changes future transport, not only anchor accuracy | Retraining merely improves the starting point | Common-stream anchor-versus-post-anchor decomposition |
| Warning requires dynamically sufficient telemetry, but explicit history is only one way to supply it | Time, current risk or generic shift is sufficient; alternatively, long history is always necessary | TorchSig temporal-order intervention plus matched-calibration and strictly nested CURE-OR state comparisons |

The mechanism claim requires the following empirical restrictions:


| Restriction | Scientific purpose |
|---|---|
| Positive cumulative net flux | Distinguishes a crossing-producing path from balanced turnover. |
| Endpoint persistence along the declared path | Shows that incident errors are not immediately repaired under that path geometry; it is not a path-independent classifier property. |
| Broad first-crossing distribution | Distinguishes persistent distributed accumulation from one synchronized pulse. |
| Focal-boundary placebos | Compare random or fixed-incorrect partitions with a closest-risk trained peer; focal self-closure is used only as the exact zero reference. |
| Common-stream training intervention | Shows that changing training support changes subsequent flux on the same latent challenge. |

The combined paired evidence satisfies this joint pattern. Round 11C establishes distributed path-conditioned accumulation, random-boundary locality and intervention response in TorchSig. Official CIFAR-10-C provides complementary fixed-incorrect and closest-risk trained-peer placebos on the same cells, while CURE-OR supplies a closest-risk trained-peer comparison on shared representations. The result is stronger than endpoint comparison but narrower than a universal causal law.

### S1.4 Heterogeneous boundary fragments and the non-necessity of synchrony

Partition the true boundary into fragments \(\mathcal B_g\). The total current is additive:

\[
J_W(t)=\sum_gJ_{W,g}(t),
\qquad
J_{W,g}(t)
=
-\int_{\mathcal B_g}j_t^\top n_W\,dS.
\]

A synchronized cliff would require a large share of total incident crossing to occur in a short interval. A distributed cliff instead has multiple fragments or sample groups contributing at different times. Both can exhaust the same headroom:

\[
\sum_g\int_{t_0}^{t}J_{W,g}(s)\,ds
\ge
\beta-R_W(t_0).
\]

The Round 11 ladder measures temporal concentration using normalized first-crossing entropy and the largest three-window incident share. These are descriptive organization statistics, not components of the accounting identity. High entropy and modest short-window share reject a dominant-pulse interpretation; they do not imply that every fragment contributes equally. The principal paired studies use ordered impairment or severity ladders. Endpoint persistence is therefore a property of each model–path pair. Non-negligible recovery shows that ordered severity does not mechanically force one-way transitions, but a dwell–reversal or round-trip arm is required before persistence can be treated as path independent.

This decomposition also distinguishes two depletion concepts:

- **Risk-headroom depletion:** cumulative net flux raises aggregate risk to \(\beta\).
- **Susceptible-mass depletion:** probability mass capable of crossing a currently active fragment becomes scarce, reducing later forward current.

The second can produce a plateau after the first has already produced a cliff. Recovery can make net current negative and reverse risk. Consequently, a post-crossing plateau or reversal does not refute a previous boundary-transport event.

### S1.5 Margin-space reduction and unpaired distributions

Let \(q_t(m)\) be the true-margin density. Risk is

\[
R_W(t)=\int_{-\infty}^{0}q_t(m)\,dm.
\]

If the margin law obeys a one-dimensional continuity equation

\[
\partial_tq_t+\partial_m(q_tv_t)=0,
\]

then

\[
\dot R_W(t)=-q_t(0)v_t(0).
\]

This expression explains why unsigned feature distance is insufficient: only probability current normal to the zero-margin boundary changes zero–one risk. Large motion parallel to the boundary can have no first-order effect, while modest motion of a dense margin shoulder can change risk substantially.

In unpaired windows, individual \(v_i(t)\) and first-crossing times are not identifiable. Equal-quantile transport instead compares the distributions through

\[
\widehat v_t(a)
=
\frac{Q_{t+\Delta}(a)-Q_t(a)}{\Delta},
\qquad a\in(0,1),
\]

where \(Q_t\) is the margin quantile function. A robust summary such as the median of \(\widehat v_t(a)\), multiplied by boundary density, can estimate bulk signed transport while reducing sensitivity to extreme quantile motion. It is not identical to paired sample current without additional monotone-coupling assumptions. This is why Covertype is described as distribution-level support rather than longitudinal confirmation.

### S1.6 Risk geometry as a pullback of transport

Let \(P_u\) be a smooth deployment family. If coordinate \(u_j\) induces a current \(j_{j,u}\), then

\[
b_j(u)
=
\frac{\partial r_W(u)}{\partial u_j}
=
-\frac1K\sum_y
\int_{\mathcal B_{W,y}}j_{j,y,u}^\top n_{W,y}\,dS.
\]

Thus \(b\) depends jointly on the trained boundary and deployment family. It is not a model-only vulnerability vector. The Hessian

\[
H_{jk}
=
\frac{\partial b_j}{\partial u_k}
\]

records how aggregate boundary susceptibility changes with state. Contributions include changes in boundary-near density, the local flow field, recovery, and the mixture weights of heterogeneous fragments.

For a path \(u(t)\),

\[
\dot r_W(t)
=
\nabla r_W(u(t))^\top\dot u(t)
\approx
\left[b+Hu(t)\right]^\top\dot u(t).
\]

This approximation is useful only on a domain and scale where both risk levels and risk differences are reliable. Covertype shows the failure mode: risk levels can be reproducible while 15-metre numerical derivatives are not. A stable global \(b\) or \(H\) should not be inferred from unreliable differences or across states where different boundary fragments dominate.

### S1.7 Observation geometry and auditability

The response model

\[
\bar O_n\mid u\sim N(\mu_0+Au,\Sigma/n)
\]

induces

\[
\mathcal Q^{\mathrm{obs}}=A^\top\Sigma^{-1}A.
\]

The main text abbreviates this matrix as \(Q\) inside experimental
formulas. The superscript emphasizes that it belongs to observation
geometry, not to the exact formation law.

The roles of \((b,H)\) and \(Q\) are distinct:

| Quantity | Depends on | Role |
|---|---|---|
| \(b,H\) | Model boundary and deployment family | Local risk susceptibility |
| \(A\) | Observation response to deployment state | Visible tangent directions |
| \(\Sigma\) | Observation noise and correlation | Precision penalty |
| \(Q\) | \(A,\Sigma\) | Observation geometry |
| \(b^\top Q^\dagger b\) | Risk and observation geometries | Risk-direction noise burden |

For a linear risk surface, the least observable risk-separated pair has

\[
D_*^2=\frac{4\gamma^2}{b^\top Q^\dagger b},
\]

and exact minimax balanced accuracy

\[
\mathcal A_n^*
=
\Phi\left(
\frac{\gamma\sqrt n}{\sqrt{b^\top Q^\dagger b}}
\right).
\]

This result requires the least-observable pair to remain inside the local domain and \(b\) to lie in the retained range of \(Q\). If a feasible risk-changing vector lies in \(\ker(Q)\), no batch size repairs the structural blindness.

For quadratic risk, exact noiseless risk sufficiency of \(Au\) requires

\[
b^\top n=0,\qquad Hn=0,
\qquad\forall n\in\ker(A).
\]

Finite-sample auditability and structural observability should therefore not be conflated. A channel may be structurally sufficient but noisy, or precise on visible directions while discarding a harmful one.

### S1.8 Warning as headroom prediction under finite coherence

Let

\[
\Delta_t=\beta-R_t,
\qquad
J_t=\dot R_t.
\]

At the exact discrete level, the future crossing margin is

\[
\Pi_t^{\mathrm{SBT}}(L)
=
\sum_{s=t}^{t+L-1}\operatorname{SBT}_s-\Delta_t.
\tag{SBT18}
\]

A positive value implies ordinary first crossing within the horizon. A persistence-confirmed event additionally requires the frozen functional \(\mathcal P_\beta\) in Eq. SBT2b. Online outcomes are unavailable, so a warning system must estimate a persistent-event target from outcome-blind telemetry. This is a partial-observation forecasting problem, not a restatement of the accounting identity.

Under locally persistent current, the raw forecast margin is \(hJ_t-\Delta_t\). With current-risk error \(e_R\), current error \(e_J\), and a lower acceleration bound \(\ddot R\ge-a\), the conservative margin is

\[
C_t(h)
=
h\widehat J_t-\Delta(\widehat R_t)
-
\left(e_R+he_J+\frac12ah^2\right).
\]

The main-text certificate follows when \(C_t(h)\ge0\). The implemented Q90 rule covers only a current-risk residual component, so the experiment evaluates operational warning rather than a complete certificate.

For an \(H\)-window least-squares slope with batch size \(B\),

\[
\operatorname{Var}(\widehat J_{t,H})
=
\frac{12\sigma^2}{BH(H^2-1)}
\]

under independent homoscedastic window errors. Approximate trend signal-to-noise is

\[
\frac{\nu\sqrt{BH^3}}{\sigma}.
\]

The useful history range is constrained by

\[
H_{\min}
\lesssim H
\le
\frac{T_{\mathrm{coh}}}{\Delta t}.
\]

This produces four qualitative regimes:

| Flux state | Observation state | Consequence |
|---|---|---|
| Cumulative flux below headroom | Any | Drift without a cliff |
| Crossing flux, observation-null | Blind cliff | No channel-based warning guarantee |
| Observable flux, coherence too short | Abrupt or weakly warnable cliff | Detection may be possible; advance warning is not |
| Observable flux, sufficient coherence | Warnable cliff | Position-plus-flux forecast can lead the crossing |

Round 8 identifies the temporal-order component used by the frozen warning functional. It does not establish that every physical degradation process possesses the required coherence.

Static observability is not identical to warning sufficiency. Let \(O_t^{(n)}\) be current outcome-blind telemetry aggregated over effective population \(n\), and let

\[
S_t=\phi(O_t^{(n)},O_{t-1}^{(n)})
\]

be an augmented state that may include a one-step prediction-state transport proxy. Let \(H_{t-1}\) denote earlier history and \(C_{t,L}^{\mathcal P}\) indicate a persistence-confirmed cliff within \(L\) windows. We call \(S_t\) dynamically sufficient under calibration context \(\mathcal C\) when

\[
I\!\left(C_{t,L}^{\mathcal P};H_{t-1}\mid S_t,\mathcal C\right)=0.
\tag{SBT18a}
\]

If the conditional mutual information is positive, explicit history retains incremental predictive content. Here \(\mathcal C\) includes the baseline identity anchor or coupling, target calibration domain, boundary, persistence rule, horizon and frozen decision rule. This separates four requirements:

1. **state sufficiency:** current or augmented telemetry retains the risk-relevant state;
2. **dynamic sufficiency:** earlier history adds no information after the augmented state is controlled;
3. **calibration consistency:** the risk baseline and persistent-event target are calibrated in the deployment domain;
4. **population-scale coherence:** the decision unit is large enough for coherent flow to dominate finite-sample boundary-fragment noise.

The CURE-OR nested analysis does not prove the information equality. It asks whether an augmented one-step state empirically approaches the fitted multi-window detector on one frozen benchmark field. TorchSig provides the complementary intervention showing that ordered history adds information when the chosen current state is not dynamically sufficient.

A local empirical decomposition is

\[
\widehat J_{n,t}=J_t^{\mathrm{coh}}+\varepsilon_{n,t},
\qquad
\operatorname{Var}(\varepsilon_{n,t})
\approx \sigma_{\mathrm{frag}}^2/n_{\mathrm{eff}}.
\tag{SBT18b}
\]

The approximation need not hold under arbitrary dependence; its role is to state why nominal batch size and effective population can differ. Given frozen gates \(\mathcal G=(a,b,\ell)\), define

\[
n_\star=\min\{n:\mathrm{TPR}_n\ge a,\ \mathrm{FPR}_n\le b,\ \operatorname{MedLead}_n\ge\ell\}.
\tag{SBT18c}
\]

CURE-OR v2 fixes \(n=50\) and therefore confirms warning at one declared operating population without estimating \(n_\star\). Equation SBT18c remains a design definition for future population-scale experiments, not an empirical claim of the present confirmation.


The CURE-OR v2 diagnostics sharpen the distinction between state, time and dynamic information. With the registered detector held fixed, time alone warned 0/72 persistent cliffs, current state alone warned 1/72, and current state plus time warned 3/72. Removing normalized time from the full readout retained 70/72 timely warnings compared with 71/72 for the complete detector. These are mean-substitution diagnostics of one frozen readout, not re-fitted best-subset comparisons. They show that normalized time was neither sufficient nor necessary under the registered field, while temporal differences and slopes carried most of the prospective information. Complete results and the threshold-calibration analysis are given in Supplementary Note 8.

### S1.9 Training intervention and repair-value decomposition

Let \(P_\alpha^{\mathrm{train}}\) be a family of training distributions and
\(W_\alpha\) the realized trained model under fixed architecture,
optimization budget, and paired randomness. The intervention target is the
chain

\[
P_\alpha^{\mathrm{train}}
\rightarrow W_\alpha
\rightarrow
\left(\partial\mathcal E_\alpha,b_\alpha,H_\alpha,
\mathcal Q_\alpha^{\mathrm{obs}}\right)
\rightarrow
\operatorname{SBT}_{\alpha,t}
\rightarrow R_{\alpha,t}
\rightarrow t_{\mathrm{cliff}}(\alpha).
\tag{SBT17}
\]

Only the first arrow is directly assigned by experiment. Common-stream
evaluation, remeasurement of \((b,H,Q)\), and paired transition accounting
identify successive links without attributing the result to a different
deployment realization. This is the theoretical status of Round 10 and Round
11C jointly. “Training composition affects generalization” is too weak a
description: the tested claim is that training distribution controls whether
future deployment transport forms a Cliff by reshaping first-order risk
direction, curvature, and observable geometry.

For repair set \(S\), define

\[
W_S
=
\mathcal A_{\mathrm{train}}(D_{\mathrm{train}}\cup S,\lambda,\xi).
\]

On a common deployment stream,

\[
\begin{aligned}
\mathcal V_T(S)
&=
R_W(T)-R_{W_S}(T)\\
&=
R_W(0)-R_{W_S}(0)
+
\sum_{t=0}^{T-1}
\left[
\operatorname{SBT}_{W,t}
-\operatorname{SBT}_{W_S,t}
\right].
\end{aligned}
\]

The first term is anchor-risk reduction; the second is cumulative flux suppression. A repair can work through either or both. Comparing models on separately sampled deployment streams would confound this decomposition, motivating the same-stream paired design.

For the frozen Round 10 common stream, the aggregate arithmetic decomposition is:

| Training support relative to baseline | Terminal-risk gain | Anchor-risk gain | Future cumulative-SBT suppression |
|---|---:|---:|---:|
| Support depleted | -0.05945 | -0.04685 | -0.01260 |
| Random broad, 5% | 0.32110 | 0.16715 | 0.15395 |
| Cliff-aware, 5% | 0.33310 | 0.17455 | 0.15855 |

Future SBT suppression was positive in all ten seed–path comparisons for both enrichment arms. The smaller Cliff-aware advantage over random broad, 0.01200 terminal risk, decomposes into 0.00740 anchor gain and 0.00460 additional future suppression. These are post hoc arithmetic decompositions of frozen ledgers; they introduce no new model fit or target sample.

Changing training also changes the observation geometry when model outputs or representations enter telemetry. A smaller \(\operatorname{tr}(Q)\) after repair is therefore not automatically harmful or beneficial. If \(b\) and \(H\) nearly vanish, the risk direction itself becomes weakly defined; the constructive result is low risk and reduced transport, not large information about a risk direction that no longer exists.

### S1.10 Coverage-before-duplication and its limits

Let \(n_g(S)\) count repair examples from deployment fragment \(g\). A separable diminishing-return approximation is

\[
\mathcal V(S)
\approx
\sum_g w_g G_g(n_g(S)),
\qquad
G_g(n+1)-G_g(n)
\text{ nonincreasing}.
\]

The marginal value of candidate \(x\in g\) is approximately

\[
\Delta\mathcal V(x\mid S)
\approx
w_g\left[G_g(n_g+1)-G_g(n_g)\right].
\]

Point hazard \(h(x)\) does not contain \(n_g(S)\) and therefore cannot account for redundancy. Under symmetric positive weights and a common concave \(G\), the first example in an uncovered fragment has value at least as large as a duplicate in a represented fragment. This gives the coverage-before-duplication exchange argument in the main text.

The conclusion fails without its assumptions. Coverage need not dominate when:

- one fragment carries nearly all future deployment exposure;
- some declared cells are risk neutral;
- a training algorithm needs several same-fragment examples before any response;
- cross-fragment interactions make the separable approximation invalid;
- the coverage map divides irrelevant variation rather than risk-relevant heterogeneity;
- acquisition costs differ materially across cells.

Round 12C therefore supports a bounded empirical principle, not a universal submodularity theorem. Its direct result is that the frozen coverage manipulation beats global hazard concentration under the same 5% budget and common acquisition pool. It does not identify the unique correct cells or prove that coverage always beats unstratified random sampling.

Round 13E tests the extension that Round 12C could not establish. In a much broader image-corruption field, both dangerous arms already span all 15 corruption families, all 10 classes, and all five first-crossing severities. The coverage arm occupies 731 fragments rather than 262, but its average boundary-pressure score is 1.4007 rather than 2.5504. Under the same 1,000-example budget, the more concentrated high-pressure arm performs better on endpoint error, risk area, and crossing fraction. Thus the first-example exchange argument is not violated mathematically; its symmetry and common-gain assumptions are empirically inappropriate once a broad support floor already exists.

**Post-target factorization hypothesis (not confirmed).** A synthesis generated after the cross-domain reversal, to be tested prospectively rather than claimed as a law, is

\[
\Delta\mathcal V(x\mid S)
\approx
\pi_{g(x)}\,\kappa(x)\,h(x)\,\phi_{g(x)}(n_{g(x)}(S)),
\tag{SBT19}
\]

where \(\pi_g\) is future deployment exposure, \(h(x)\) is boundary pressure or crossing depth, \(\kappa(x)\) is transfer to held-out boundary flow, and \(\phi_g\) decreases with within-fragment allocation. Coverage is represented by the occupancy-dependent factor \(\phi_g\); it is one component of marginal value, not the complete allocation objective. Round 13E generates this weighted support-allocation hypothesis but does not confirm its factors separately.


### S1.11 Claim hierarchy and theory-to-evidence dictionary

| Mathematical or evidential level | Supported statement | What is not inherited |
|---|---|---|
| Exact accounting | \(\Delta R_t=J_t^+-J_t^-\), gross-turnover resolution when \(J_t^+\) and \(J_t^-\) are retained, and boundary-indexed headroom exhaustion for a fixed deterministic classifier on paired identities | Observability, prediction, causal attribution or external safety meaning |
| Local smooth representation | Continuous surface current and the \((b,H)\) risk chart under stated regularity | Global smoothness, a unique physical coordinate or a guarantee that derivatives are estimable at every scale |
| Telemetry conditional | Auditability and warning under risk-sufficient sensing, target calibration, temporal information, finite coherence and adequate effective population | Universal warning from arbitrary unlabelled streams or schedule-independent calibration |
| Controlled intervention | Training support changes the learned boundary and future transport on a common stream | Broad architectural generality or a universal repair allocator |
| Post-target hypothesis | Exposure, pressure, redundancy and transfer are candidate factors in set allocation | A prospectively confirmed conditional allocation law |

| Statement | Status | Evidence or requirement |
|---|---|---|
| Paired risk change equals incident minus recovery mass | Exact accounting identity | Main-text Eq. (1); Supplementary Eqs. SBT1–SBT2 |
| The scalar SBT contains more scalar information than \(\Delta R_t\) | Rejected by definition | They are equal; added information comes only from the resolved ledger |
| Continuous risk change equals net task-boundary probability current | Exact under stated smoothness and regularity | Supplementary Note 1.2 |
| A cliff occurs when cumulative SBT exhausts declared headroom | Exact first-passage consequence for a fixed paired path and boundary | Main-text Eq. (2); Supplementary Eq. SBT2 |
| The ledger is independent of the chosen operational boundary | Exact | Threshold changes first passage, not the transition counts |
| Synchrony is necessary | Rejected | Round 11C and official CIFAR-10-C show distributed first crossings |
| Named physical coordinates are necessary for deployed readout | Rejected in the frozen TorchSig field | Round 9 control-blind chart |
| Time or current state alone explains CURE-OR warning | Rejected for the frozen readout | Supplementary Notes 7–8 |
| Training support changes future SBT after the anchor | Controlled-field evidence | Round 10 decomposition and Round 11C paired transport |
| Signed paired transport transfers beyond TorchSig | Supported in one second paired neural-network benchmark | Official CIFAR-10-C, three training seeds |
| Degradation or positive risk drift necessarily creates a cliff | Rejected within CIFAR-10-C | Noncrossing corruption families retain headroom or recovery |
| Coverage dominates hazard in the sparse controlled RF field | Bounded five-seed result | Round 12C; 4/5 seed directions, exact one-sided sign tail 0.1875 |
| Coverage universally dominates repair | Prospectively rejected | Round 13E; all three directional gates fail |
| A conditional coverage–pressure law is confirmed | Not established | Requires a prospectively frozen orthogonal factorial design |
| Formation–warning–guarded control closes on one ordered image-benchmark field | Preregistered same-benchmark confirmation | CURE-OR v2 H1–H3 under two-phase commitment |
| Hybrid25 is calibration free, population invariant or universally optimal | Not established | One target calibration, one 50-identity operating unit and one frozen readout |
| CURE-OR v2 directly measures \(b\), \(H\) or \(\mathcal Q^{\mathrm{obs}}\) | Not established | It measures risk, transitions, telemetry, alarms and guarded outcomes |
| Any benchmark-relative threshold is a safety boundary | Not established | Requires independent utility, engineering or regulatory justification |

### S1.12 Proof details for results invoked in the main text

**Exact paired accounting.** For each identity, \(e_{i,t}\in\{0,1\}\). Exhausting the four pairs \((e_{i,t},e_{i,t+1})\) gives

\[
e_{i,t+1}-e_{i,t}
=
\mathbf1\{0\rightarrow1\}
-
\mathbf1\{1\rightarrow0\}.
\]

Summing over identities and dividing by \(N\) proves Eq. SBT1. Summing Eq. SBT1 over time telescopes to Eq. SBT2. Comparing the telescoped sum with \(\beta-R_0\) gives the first-passage criterion.

**Continuous boundary current.** Under the continuity equation, substitute \(\partial_t p=-\nabla\cdot j\) into the risk integral over the fixed error region and apply the divergence theorem. The resulting surface integral is the signed normal probability current in Supplementary Note 1.2. The fixed-model condition is essential because otherwise boundary motion contributes an additional term.

**Linear least-observable pair.** For two local states separated by \(d=u_+-u_-\), linear risk separation requires \(b^\top d\ge2\gamma\). On \(\operatorname{range}(Q)\), generalized Cauchy–Schwarz gives

\[
(b^\top d)^2
\le
(b^\top Q^\dagger b)(d^\top Qd).
\]

Therefore \(d^\top Qd\ge4\gamma^2/(b^\top Q^\dagger b)\), with equality at \(d=2\gamma Q^\dagger b/(b^\top Q^\dagger b)\) when feasible. For two Gaussian means with common covariance \(\Sigma/n\), the equal-prior Bayes balanced accuracy is \(\Phi(\sqrt{nD_*^2}/2)\), yielding the expression in Supplementary Note 1.7.

**Structural blindness.** If a feasible \(n\in\ker(A)\) changes risk, states separated along \(n\) have identical noiseless telemetry distributions but different risk. No decision rule using that channel can distinguish the constructed pair better than chance uniformly over the pair.

**Quadratic fibre observability.** For \(n\in\ker(A)\), states \(u\) and \(u+n\) share the same noiseless observation. Their quadratic-risk difference is

\[
r(u+n)-r(u)
=b^\top n+u^\top Hn+\frac12 n^\top Hn.
\]

If \(Hn=0\) and \(b^\top n=0\), the difference vanishes on every fibre. Conversely, if it vanishes for every admissible \(u\), the coefficient of \(u\) gives \(Hn=0\), and the remaining constant term gives \(b^\top n=0\).

**Finite-horizon warning margin.** If \(\ddot R(s)\ge-a\) on \([t,t+h]\), two integrations give \(R(t+h)\ge R_t+hJ_t-ah^2/2\). Replacing \(R_t\) and \(J_t\) by lower confidence bounds yields the conservative margin in Supplementary Note 1.8. The implemented Q90 rule does not estimate every term and is therefore an operational test rather than a complete certificate.

## Supplementary Note 2 | TorchSig qualification, warning and training-support intervention


### S2.1 Global linear rejection and local linear identification

The first wide probe was designed to fail safely if a global linear bridge was not justified. It spanned a deployment-risk range of 0.475, but the risk fit had \(R^2=0.528\), below the frozen 0.70 gate. Its state-discrimination curve still appeared accurate (MAE 0.0239; Spearman 0.949), while the realized pair had an actual half-risk gap of only 0.00125. This establishes an important separation: an observation channel can distinguish states while the risk model assigning meaning to those states is wrong.

The subsequent local probe restricted the calibration domain and increased linear risk \(R^2\) to 0.955. An attempted amplitude-only blind channel was rejected rather than redefined: its risk-null ratio was 0.185 and a target-label oracle reached 1.0 accuracy at the largest batch size. The experiment therefore did not manufacture a structural null direction by changing the numerical rank threshold after reveal.

### S2.2 Replicated linear midpoint failure and quadratic repair

Two independently seeded formal linear replays each passed seven of eight gates:

| Run | Theory--empirical MAE | Spearman | Actual half-gap / \(\gamma\) | Midpoint drift | Failed gate |
|---|---:|---:|---:|---:|---|
| Linear v1 | 0.0231 | 0.9747 | 1.2375 | +0.01987 | One endpoint missed its frozen relative set |
| Linear v2 | 0.0205 | 0.8944 | 1.2050 | -0.01422 | One endpoint missed its frozen relative set |

Both candidate pairs were observationally distinguishable and had sufficient realized total separation. Their failure came from curvature moving the midpoint in opposite directions. This repeated pattern motivated an asymmetric quadratic pair rather than a larger target sample.

The independent quadratic replay passed all seven pre-target gates:

| Pre-target quantity | Result |
|---|---:|
| Quadratic training \(R^2\) | 0.9766 |
| Five-fold risk \(R^2\) | 0.9710 |
| Risk-relevant score \(R^2\) | 0.9853 |
| Risk-null ratio | 0.1123 |
| Pair constraint error | \(4.70\times10^{-15}\) |
| Minimum support slack | \(-1.58\times10^{-11}\) |
| Five-fold pair error | 0.00231 |

The fitted Hessian norm was 6.079 and the optimized asymmetric pair had \(D_Q^2=1.82387\). The theoretical and empirical information curves had MAE 0.02297 and Spearman correlation 0.9411. Both revealed point risks crossed their relative cutoffs, but the lower marginal Wilson interval missed by 0.000292. The retained status is therefore 12/13 with interval abstention. The result supports the fitted local candidate and information curve, not a global polynomial optimum or externally meaningful safety boundary.

### S2.3 Measurement-objective rejection and independent compression success

All 63 nonempty combinations of six observation groups were evaluated. At budget 27, the designated risk-directed criterion and \(\operatorname{tr}(Q)\) selected the same 25 dimensions at both centers: model outputs, complex moments, autocorrelation, and spectral summaries.

The frozen 1.05 objective-superiority gate therefore failed at 1.00 and target generation stopped. This is a negative result about unique objective superiority, not a negative result about compression.

The common subset was then frozen and replayed independently:

| Center | Cost fraction | Worst-pair information retained | Curve MAE | Spearman | Mean full-minus-25D accuracy |
|---|---:|---:|---:|---:|---:|
| Balanced | 0.463 | 0.8432 | 0.0288 | 0.9856 | -0.00056 |
| Phase heavy | 0.463 | 0.8911 | 0.0247 | 0.9856 | -0.00056 |

All compression gates passed. The supported conclusion is local information-preserving compression, not universal sensor optimality.

### S2.4 Sequential warning ladder and retained failures

The first independent warning replay used history \(H=6\), horizon \(L=5\), a calibration Q90 residual buffer, and the phase-heavy center. The 25D position-plus-velocity rule achieved 64.44% timely warning with median lead three, compared with 12.78% for contemporaneous risk. An entropy/margin trend achieved only 1.11% timely warning and generated 100% stationary false alarms. The 25D rule nevertheless exceeded the premature-warning limit by 1.67 percentage points and retained a 9/10 decision.

The next source attempt aborted pre-target because one direction could not satisfy the endpoint geometry. A later two-direction source passed warning gates, but an after-the-fact same-seed logging rerun changed sealed core hashes. That source was invalidated rather than used for confirmation.

The replacement source froze single-thread execution, all-time logging, and a two-process determinism check before fresh target generation. It passed all pre-target gates and all ten operational warning gates:

| Metric | 25D forecast | 25D current state | 54D forecast |
|---|---:|---:|---:|
| Timely warning | 0.6250 | 0.0583 | 0.6417 |
| Premature warning | 0.1250 | 0.0000 | 0.0833 |
| Stationary false alarm | 0.0417 | 0.0000 | 0.0417 |
| Median timely lead | 3 | 1 | 2 |

The gain over contemporaneous detection was 0.5667. This established a prospective signal to explain; it did not yet identify temporal accumulation as its source.

### S2.5 Fixed-terminal temporal-order intervention

Round 8 used 120 matched event replicates and 28,800 shuffled replays. Every shuffle preserved the complete risk-coordinate multiset and the final pre-crossing coordinate. The primary continuous estimands passed:

| Mechanism estimand | Result |
|---|---:|
| Integrated slope difference | 0.005398 |
| Slope bootstrap interval | [0.004982, 0.005815] |
| Integrated forecast difference | 0.030075 |
| Forecast bootstrap interval | [0.027460, 0.032763] |
| Ordered pair superiority | 0.9833 |
| Positive-slope fraction difference | 0.3703 |
| Any-window alarm difference | 0.2672 |
| Sudden-proxy pre-warning | 0.0083 |

Ordered mean integrated slope was 0.006322 versus 0.000924 after shuffling, so destroying order removed 85.4% of the trend magnitude while retaining the observed state inventory and final pre-crossing state. Both frozen directions had positive effects. The result identifies the order-specific contribution used by the warning functional; it does not imply that the counterfactual shuffled sequences are physically realizable deployment paths.

### S2.6 Control-blind readout and training-reference null result

Round 9 withheld named physical controls from the deployed predictor. Calibration still used labelled environments, but online prediction received only outcome-blind batch telemetry.

| Readout | Timely warning | Premature | Stationary false alarm | Median lead | Position MAE |
|---|---:|---:|---:|---:|---:|
| 25D mean + log variance | 0.7250 | 0.0000 | 0.0000 | 2.0 | 0.007345 |
| 25D training relative | 0.7167 | 0.0000 | 0.0000 | 2.5 | 0.007355 |
| Oracle physical coordinate | 0.7750 | 0.0583 | 0.0000 | 3.0 | 0.008822 |
| Wrong-reference placebo | 0.0000 | 0.0000 | 0.0000 | -- | 0.159031 |

Relative to the matched moment chart, the training-relative timely-warning difference was -0.00833 with paired interval [-0.025, 0]. Its position-MAE difference also spanned zero. The formal state was 14/16; both failed gates concerned incremental training-reference value.

The wrong-reference placebo establishes coordinate-origin sensitivity after a reference-based chart has been fitted. It does not establish that the correct training set contains additional predictive information. The result therefore separates two claims:

- named physical coordinates are unnecessary for the deployed readout;
- a fixed training reference is not a privileged extra warning channel once representation capacity is matched.

### S2.7 Common-stream training-support intervention

Round 10 crossed four equal-size training supports with five fresh training seeds. Model random state, class balance, learner, calibration cache, and deployment cache were paired within seed. A 20% pilot was retained as a saturation failure because both enrichment strategies reached the risk floor. The formal dose was frozen at 5%.

| Training support | Start risk | End risk | Path area | Shared-boundary crossing fraction |
|---|---:|---:|---:|---:|
| Support depleted | 0.2226 | 0.3936 | 0.2938 | 1.00 |
| Baseline | 0.1758 | 0.3341 | 0.2401 | 1.00 |
| Random broad, 5% | 0.0086 | 0.0130 | 0.00945 | 0.00 |
| Cliff-aware, 5% | 0.0012 | 0.0010 | 0.00110 | 0.00 |

The predeclared continuous contrasts were:

| Paired contrast | Estimate | Descriptive five-seed cluster-bootstrap range |
|---|---:|---:|
| Cliff-aware minus baseline, terminal risk | -0.33310 | [-0.36830, -0.29045] |
| Cliff-aware minus baseline, path area | -0.23899 | [-0.27026, -0.20198] |
| Cliff-aware minus random broad, terminal risk | -0.01200 | [-0.02625, -0.00350] |
| Cliff-aware minus random broad, path area | -0.00835 | [-0.01745, -0.00265] |

Both enrichment arms removed all ten common-boundary crossings. The targeted advantage is therefore supported only through continuous endpoints and is substantially smaller than the broad enrichment effect. After averaging the two paths within seed, all five seed-level Cliff-aware contrasts were positive relative to baseline and random broad; the exact one-sided sign tail for an all-positive five-seed comparison is \(1/32\). This is small-sample directional evidence, not a population-level guarantee.

The local geometry changed as follows:

| Quantity | Baseline | Cliff-aware | Ratio |
|---|---:|---:|---:|
| \(\tau\) | 0.219891 | 0.001220 | 0.00555 |
| \(\|b\|_2\) | 1.055517 | 0.005255 | 0.00498 |
| \(\|H\|_F\) | 2.420089 | 0.092255 | 0.03812 |
| \(\operatorname{tr}(Q)\) | 155.7854 | 134.7827 | 0.86518 |

The gradient and Hessian collapse supports local risk-field flattening. The reduction in \(\operatorname{tr}(Q)\) is not itself an improvement: once the risk field is almost flat, large information about its former direction is no longer the constructive objective.


### S2.8 Frozen-output diagnostics on the frozen Round 10 outputs

The terminal-risk benefit was decomposed exactly into anchor and post-anchor components:

| Regime relative to baseline | Terminal gain | Anchor gain | Future cumulative-SBT suppression | Fraction of terminal gain from future suppression |
|---|---:|---:|---:|---:|
| Random broad | 0.32110 | 0.16715 | 0.15395 | 47.9% |
| Cliff-aware | 0.33310 | 0.17455 | 0.15855 | 47.6% |
| Cliff-aware relative to random broad | 0.01200 | 0.00740 | 0.00460 | 38.3% |

Future suppression was positive in all ten seed–path comparisons for both enrichment regimes. This establishes that the effect is not reducible to better deployment-anchor accuracy.

A separate boundary sensitivity recomputed the same two-consecutive-window first passage without changing any model, trajectory or transport count:

| Boundary \(\beta\) | Support depleted | Baseline | Random broad | Cliff-aware |
|---:|---:|---:|---:|---:|
| 0.15 | 10/10 | 10/10 | 0/10 | 0/10 |
| 0.20 | 10/10 | 10/10 | 0/10 | 0/10 |
| 0.25 | 10/10 | 10/10 | 0/10 | 0/10 |
| 0.30 | 10/10 | 8/10 | 0/10 | 0/10 |
| 0.35 | 10/10 | 6/10 | 0/10 | 0/10 |

The registered boundary remains primary. The diagnostic shows that the enrichment ordering is not produced by a uniquely tuned line, while cliff incidence and time remain correctly threshold dependent. Complete threshold rows are summarized in Supplementary Note 8 and released in the frozen-output diagnostic archive.

### S2.9 Evidence-status ledger through the training intervention


| Stage | Frozen outcome | Role in the final theory |
|---|---|---|
| Wide linear bridge | STOP | State discrimination cannot rescue risk misspecification. |
| Local linear probe | Advance | Identifies a valid local domain. |
| Two formal linear replays | 7/8 each | Repeated midpoint drift motivates curvature. |
| Quadratic asymmetric replay | 12/13 | Supports local quadratic auditability; interval abstention retained. |
| Measurement-objective comparison | STOP | Rejects unique objective superiority. |
| Independent 25D compression | PASS | Shows local observation compression. |
| First warning formal | 9/10 | Finds prospective signal; premature gate retained. |
| Fresh warning source | PASS | Establishes a signal to explain. |
| Temporal-order knockout | PASS | Identifies ordered accumulation in the frozen rule. |
| Blind-\(u\) readout | 14/16 | Supports control-blind warning; rejects training-reference increment. |
| Training-support intervention | 16/16 | Establishes upstream control of the risk field. |

## Supplementary Note 3 | Distribution-level margin transport under unpaired field shift


### S3.1 Purpose and claim level

The Covertype study was not designed as a second paired replication of the TorchSig mechanism. It asked a more basic external question: when deployment samples move relative to training support in a public real-data field, which description of that motion tracks model-risk change? The study progressively rejected unsigned distance, globally stable low-order geometry, and synchronized crossing as general explanations. Its terminal supported statement is:

> In the balanced Covertype protocol, risk changes were associated with robust signed transport of probability mass in the trained model's margin space, rather than with unsigned displacement from the training distribution alone.

Because different examples populate different elevation windows, this is distribution-level, model-conditional evidence. It does not identify longitudinal sample trajectories, literal first-crossing times, or an unlabelled deployment monitor.

### S3.2 Field selection and source construction

Two natural-time candidates were rejected before Covertype was advanced. Elec2 produced no repeated ExtraTrees source and only one repeated MLP source. Wild-Time Yearbook produced a repeatable crossing only from its 1970 start. These failures show that chronological drift alone is not sufficient to construct a multi-source Cliff field.

Covertype retained its two largest forest-cover classes. Elevation 2670--3360 m was divided into 46 fixed 15-metre windows. Every window contained 256 examples from each class, fixing class balance, batch size, and sample count. Both increasing- and decreasing-elevation directions were tested from three frozen starts. Training and calibration preceded six held-out deployment windows. A crossing required both error and Brier score to exceed their calibration Q90 boundaries for two consecutive windows.

The source smoke produced repeated crossings in 5/6 sources for both an MLP and ExtraTrees. Removing elevation from the MLP inputs did not remove the phenomenon: 23/30 source–seed cases crossed, and 4/6 sources crossed for all five fresh seeds. Elevation therefore constructed the path but was not a sufficient direct-threshold explanation.

### S3.3 Bounded support-intervention pilot

For each of six sources and five fresh seeds, a 5% data budget compared baseline, random broad coverage, and path-local coverage on exactly the same held-out stream.

| Arm | Crossings | Mean target Brier | Mean target error |
|---|---:|---:|---:|
| Baseline | 27/30 | 0.22674 | 0.36349 |
| Random broad, 5% | 27/30 | 0.22432 | 0.35462 |
| Path local, 5% | 15/30 | 0.20797 | 0.32884 |

Path-local training beat baseline and random broad coverage in 30/30 Brier pairs. Its source-clustered Brier differences were -0.01877 (bootstrap range [-0.02491, -0.01360]) versus baseline and -0.01635 ([-0.02308, -0.01042]) versus random broad coverage. This was a bounded pilot using labelled samples from the future path; it supported training-support sensitivity but did not itself solve privileged-coordinate-free acquisition. Round 12C later addresses that question only inside the controlled TorchSig field.

### S3.4 Registered failures that changed the mechanism

| Stage | Decision | Scientific consequence |
|---|---|---|
| Fine-scale smooth geometry | STOP | Stable global (b,H) were not supported; harmful motion concentrated late. |
| Blind support geometry | STOP | Training-support gap correlated with risk in some panels but was not risk sufficient. |
| 15-metre flow forecast | STOP/REDESIGN | Risk levels were reliable, but derivatives and path directions were not. |
| Scale audit | Diagnostic | Signs became reproducible around 45--60 m; 45 m was selected post-target and is not universal. |
| 45-metre mean-flow formal | STOP | Mean transport failed fresh-panel bootstrap and advantage gates. |
| Panel crossover | Diagnostic | Failure localized to a model-fit by deployment-panel interaction. |
| Shoulder synchronization | STOP/REJECT | Velocity-shuffle controls were stronger; monotone unpaired coupling made apparent synchrony degenerate. |
| Robust-transport holdout | 9/10 STOP | Robust signed transport replicated, but strict superiority over mean transport was not certified. |

The decisive conceptual change was to describe deployment in the trained model's signed-margin space. If (q_t(m)) is the margin distribution at window (t), risk is its mass on (m\le0). Unsigned displacement can be large while mass moves parallel to the boundary. Conversely, modest global displacement can change risk when a dense shoulder moves across zero. Forward transport can also be offset by recovery or depletion of susceptible mass, producing plateaus or reversals after crossing.

### S3.5 Terminal fresh holdout

The last holdout froze 30 newly seeded MLP fits and 60 transitions. The primary robust statistic multiplied boundary density by the negative median equal-quantile signed-margin velocity.

| Terminal quantity | Result |
|---|---:|
| Spearman with error change | 0.746 |
| Spearman with Brier change | 0.821 |
| Increasing-elevation correlation | 0.779 |
| Decreasing-elevation correlation | 0.746 |
| Unsigned hidden-space distance correlation | -0.029 |
| Coordinate-permuted boundary q95 | 0.269 |
| Two-way source/seed cluster range | [0.485, 0.879] |
| Robust-minus-mean point advantage | 0.314 |
| Robust-minus-mean cluster interval | [-0.052, 0.678] |

The final gate failed because the last interval crossed zero. The result therefore supports signed margin transport and true-boundary specificity but not strict universal superiority of the robust statistic over ordinary mean-margin transport.

### S3.6 Covertype claim boundary

Supported interpretation:

- repeated operational crossings in a class-balanced real-data path;
- scale dependence of risk derivatives;
- model-relative signed transport is more informative than unsigned training distance in the terminal holdout;
- the trained boundary outperforms coordinate-permuted boundary placebos;
- the mechanism evidence is distribution level and model conditional.

Outside the present evidence:

- universal first- or second-order risk geometry;
- a universal 45-metre physical scale;
- longitudinal sample flow or synchronized crossing;
- causal proof that robust median transport generates every Cliff;
- strict robust-over-mean superiority;
- unlabelled deployment use;
- a complete formal pass.

## Supplementary Note 4 | Persistent paired transport and rejection of synchronized crossing


### S4.1 Rejected synchronized-crossing hypothesis (Round 11A)

The first paired TorchSig smoke preserved each latent signal across deployment windows. It froze one new model seed, 192 paired examples per path, baseline and Cliff-aware training, exact forward/recovery accounting, a class-conditional velocity shuffle, and random probability-space boundaries. Six of seven gates passed.

| Quantity | Result |
|---|---:|
| Maximum flux-accounting error | \(3.82\times10^{-17}\) |
| Minimum baseline terminal-risk increase | 0.1250 |
| Minimum baseline incident-crossing fraction | 0.1302 |
| True boundaries above random-boundary q95 | 2/2 |
| Mean Cliff-aware terminal-risk reduction | 0.2734 |
| Mean Cliff-aware incident-crossing reduction | 0.1458 |
| Paths above velocity-shuffle q95 | 0/2 |

The frozen decision was `SMOKE_STOP_REDESIGN`. The failed gate rejected the claim that samples nearer the boundary receive an additional specially harmful velocity.

### S4.2 Declared post-target diagnosis

The diagnosis used 4,096 class-conditional velocity permutations per active transition. Only one of 22 active transitions exceeded its shuffle q95; under a 5% null, the one-sided binomial probability of at least one exceedance was 0.6765. First crossings were broad and persistent:

| Path | Incident fraction | Three-window incident share | Time IQR | Normalized entropy | Endpoint persistence |
|---|---:|---:|---:|---:|---:|
| Noise | 0.1302 | 0.0573 | 4.0 | 0.8349 | 0.9600 |
| Mixed gradient | 0.1771 | 0.0677 | 5.5 | 0.8436 | 0.9412 |

This diagnosis did not relabel the failed smoke. It motivated a new hypothesis: a Cliff can form when temporally distributed, mostly irreversible net boundary flux accumulates beyond risk headroom; a synchronized pulse is a possible subclass, not a necessary condition.

### S4.3 Fresh distributed-flux replacement probe (Round 11B)

Three new model seeds and a fresh panel of 320 signals per path were generated after the replacement hypothesis, seeds, and nine gates were frozen. All nine passed.

| Quantity | Result |
|---|---:|
| Baseline relative-cliff fraction | 1.0000 |
| Maximum accounting error | \(2.26\times10^{-17}\) |
| Median endpoint persistence | 0.9590 |
| Median crossing-time entropy | 0.8274 |
| Median largest three-window share | 0.3798 |
| Extra velocity-coupling pairs | 0/6 |
| True-boundary-specific pairs | 6/6 |
| Mean terminal-risk reduction | 0.2224 |
| Mean incident-crossing reduction | 0.0828 |

Because this probe used an approximate NumPy signal kernel, its decision explicitly required a source- or package-faithful confirmation.

### S4.4 Official-source paired transport confirmation (Round 11C)

Round 11C transcribed the exact TorchSig `v2.1.1` algorithms exercised by the frozen configuration: constellation maps, SRRC pulse shaping, the unit-rate constellation modulator, nonlinear amplification, carrier phase noise, and AWGN. The configuration's ideal resampling factor is exactly one. Five fresh training seeds paired baseline and Cliff-aware models; 512 fresh latent signals were followed along each of two paths. All 12 frozen gates passed.

| Quantity | Result |
|---|---:|
| Baseline relative-cliff fraction | 1.0000 |
| Maximum accounting error | 0.0000 |
| Median endpoint persistence | 0.9525 |
| Median crossing-time entropy | 0.8789 |
| Median largest three-window share | 0.3827 |
| Extra velocity-coupling pairs | 0/10 |
| True-boundary-specific pairs | 9/10 |
| Mean Cliff-aware terminal-risk reduction | 0.27285 |
| Descriptive five-seed cluster-bootstrap range | [0.21543, 0.33496] |
| Mean Cliff-aware incident-crossing reduction | 0.12148 |
| Descriptive five-seed cluster-bootstrap range | [0.08574, 0.15176] |
| Minimum pathwise median terminal reduction | 0.22852 |
| Minimum pathwise median incident reduction | 0.11914 |

The five model seeds are the intervention clusters; identities, windows and paths are not independent model replications. The displayed ranges are therefore descriptive small-cluster summaries.

The pre-target manifest digest is `e4bb820525969d856da9d2a700a46172915bd3cfc18159526a138dc27e03eddb`; the paired panel digest is `d360fd48701e7d4539be9943fc3dbb9491c9aeb841136450e673fbc318c23178`.

### S4.5 Runtime scope

The runtime is `torchsig_2.1.1_official_source_numpy_execution`, based on TorchSig tag `v2.1.1`, source commit `d9abfe1af2b0216d2bacc31c677407ed31878086`. The published wheel SHA-256 reserved for future package-runtime confirmation is `2e6ea54df639028b4914fee4daea0ed7e87ef53d15c4dddd939d0d867e24d2e1`.

This removes the approximate signal-kernel confound but is not an installed PyTorch/TorchSig package execution. The retained decision is `OFFICIAL_SOURCE_NUMPY_CONFIRMATION_PASS_PACKAGE_RUNTIME_REQUIRED`.

## Supplementary Note 5 | Repair-set selection under sparse support (Rounds 12A–C)


### S5.1 Why mechanism activity did not immediately yield a repair selector

Round 11C identifies persistent decision-boundary transport as the state that assembles aggregate risk. The next question is prescriptive: which 5% of observed deployment examples should be added to training? A point with a clear crossing can be mechanistically informative yet redundant with other selected points. Retraining acts on a set, so pointwise crossing incidence or one-at-a-time influence need not equal marginal set value.

The control-free ladder retained every failed selector:

| Stage | Selector | Absolute repair | Comparator result | Decision |
|---|---|---:|---|---|
| 12A-v1 | Unlabelled persistent switch | Terminal risk -0.1875 | Random better by 0.0371 | 12/13, STOP |
| 12A-v2 | Predicted-class-stratified switch | Incident crossing -0.1035 | Random better in terminal risk by 0.2324; anchor risk rises | 11/14, STOP |
| 12A-v3 | Uncertainty query, then signed flux | Not evaluated | Query contains only true classes 0/1 | Pre-target acquisition abort |
| 12A-v4 | Common-query signed flux | Terminal risk -0.2656 | Random/uncertainty better by 0.0176/0.0195 | 13/16, STOP |
| 12B | One-example counterfactual influence | Terminal risk -0.2188 | Random/uncertainty/direct flux better by 0.0508/0.0332/0.0117 | 13/16, STOP |

These results reject the claim that a more accurate pointwise hazard score is automatically a better small repair set. They motivate a different experimental factor--coverage--rather than another renamed ranking score.

### S5.2 Sparse-field equal-budget design (Round 12C)

Five fresh training seeds each generate a 320-example baseline training set, one common acquisition stream, and one independent evaluation stream. An outcome-blind random query takes equal quotas from each path-by-anchor-predicted-class stratum. After the 64 queried labels are revealed, every trajectory contributes candidates in early windows 7--13 and late windows 14--20. The repair budget is 16 examples, exactly 5% of training.

The deployment field is partitioned into 16 cells: queried true class (4) by deployment path (2) by early/late phase (2). The four arms are:

1. global random selection;
2. global concentration on the highest phase-local hazard scores;
3. one random unique trajectory per deployment cell;
4. one highest-hazard unique trajectory per deployment cell.

Training size, base data, model random state, acquisition pool, and fresh evaluation samples are paired within seed. The primary outcome is baseline-minus-repaired terminal risk, averaged over both paths. The pre-target primary contrast is coverage-random minus hazard-concentrated reduction.

### S5.3 Coverage–hazard manipulation and primary result (Round 12C)

| Repair selection | Cell coverage | Local hazard score | True incident hit rate | Terminal-risk reduction | Incident-crossing reduction | Path-area reduction |
|---|---:|---:|---:|---:|---:|---:|
| Random unstratified | 0.5875 | 1.0478 | 0.1625 | 0.21875 | 0.13125 | 0.16396 |
| Hazard concentrated | 0.4375 | 4.8575 | 0.7875 | 0.15195 | 0.09922 | 0.10930 |
| Coverage random | 1.0000 | 0.5240 | 0.0750 | 0.24609 | 0.13594 | 0.19396 |
| Coverage plus hazard | 1.0000 | 2.5692 | 0.4125 | 0.24648 | 0.13359 | 0.19615 |

The manipulation is deliberately adversarial to the claim: hazard concentration has a 4.3336 higher hazard score and 71.25 percentage-point higher true-incident hit rate than coverage-random. Nevertheless, coverage-random has 0.09414 greater terminal-risk reduction.

| Training seed | Coverage-random minus hazard-concentrated reduction |
|---:|---:|
| 20261261 | 0.13867 |
| 20261262 | 0.08008 |
| 20261263 | 0.17773 |
| 20261264 | 0.08984 |
| 20261265 | -0.01563 |
| Mean | 0.09414 |
| Descriptive five-seed cluster-bootstrap range | [0.03633, 0.14648] |

All 14 frozen gates pass. Four of five seed-level primary contrasts are positive, corresponding to an exact one-sided sign tail of 0.1875; the result is therefore a controlled-field effect rather than population-level evidence. Baseline models cross on 10/10 seed–path pairs. Coverage-random has positive median terminal-risk reduction on both paths, and no repair arm loses validation accuracy relative to its paired baseline. Adding coverage to high hazard contributes 0.09453 mean reduction. Adding hazard ranking after full coverage contributes 0.00039, with a post hoc interval [-0.01914, 0.01797].

Coverage-random exceeds unstratified random by 0.02734 on average, but the five-seed interval [-0.03125, 0.10898] spans zero. Round 12C therefore supports coverage over hazard concentration in this frozen sparse field, not universal superiority over every random realization.

### S5.4 Implementation history and claim boundary

The first execution attempt stopped during acquisition-table assembly because a seed column was inserted twice. The second stopped during validation-set construction because a nested validation mapping was omitted. Neither generated an evaluation stream. Both records are retained, the implementation-only fixes changed no seed, selector, endpoint, or gate, and the final run used pre-target digest 853e7cc450db493cd8e4bad4079bff1758e14e1fb0ee482b822927cde6b32090.

Supported interpretation:

- under the frozen 5% TorchSig budget, spanning all queried class-by-path-by-phase cells is more effective than concentrating on the highest local hazards;
- local boundary hazard is a mechanism-state measurement but not a sufficient set-selection objective;
- physical impairment coordinates are unnecessary for this bounded repair-selection rule.

Outside the present evidence:

- universal superiority of coverage over arbitrary random sampling;
- fully unlabelled repair, because the common query pool is labelled before selection;
- automatic discovery of path or phase cells;
- necessity of every individual coverage axis;
- transfer beyond the frozen ExtraTrees/TorchSig paths.

Round 13E is the prospective external stress test of the boldest possible extension of this result. Its opposite directional outcome leaves every Round 12C number and gate intact, but narrows the supported rule: coverage can dominate severe concentration in a sparse field; it is not sufficient as a universal allocation objective after broad domain support has already been established.

## Supplementary Note 6 | Paired image-domain formation and prospective repair reversal


### S6.1 Identity-pairing qualification and CURE-TSR abstention (Round 13A)

The target of Round 13 was not merely another corruption-accuracy table. It was a second neural-network domain in which the *same identities* could be followed through ordered degradation, so that correct-to-error and error-to-correct transitions were sample paired. CURE-TSR was screened first because it offers realistic challenge conditions. The official public description, however, defines an index as a different instance within a condition and does not certify that the same index across challenge levels is the same underlying frame. The formal files also required a license form and were unavailable during execution. The frozen eligibility audit therefore returned `DATA_UNAVAILABLE`; no CURE-TSR pairing, risk, or mechanism claim was made. The preregistered mini CURE-OR v2 confirmation in Supplementary Note 7 is a distinct object-recognition dataset and protocol.

The fallback ladder separated qualification from confirmation. Round 13B generated three controlled corruptions from the same CIFAR-10 test identities. Round 13C expanded seeds, paths, and identities. Only after both stages passed was the official CIFAR-10-C archive opened for Round 13D. Round 13E then used fresh training seeds and a sealed identity split to test repair allocation. The stage decisions were:

| Stage | Evidence target | Design | Frozen decision |
|---|---|---|---|
| 13A | Eligibility of real-world paired challenge data | CURE-TSR access and cross-level identity audit | Hold CURE-TSR claims |
| 13B | Minimal paired image-domain signal | One CNN, three generated paths, 2,000 identities | `ADVANCE_TO_ONE_HOUR_PILOT`, 8/8 |
| 13C | Seed and path robustness before official data | Three fresh CNN seeds, six paths, 10,000 identities | `ADVANCE_TO_OFFICIAL_BENCHMARK`, 9/9 |
| 13D | Official second-domain mechanism test | CIFAR-10-C, 15 families, five severities, 10,000 paired identities | `MECHANISM_DOMAIN_CONFIRMED`, 12/12 |
| 13E | Equal-budget repair allocation | Three fresh seeds, four arms, 8,000 sealed identities | `PARTIAL_OR_STOP`, 6/9 |

### S6.2 Minimal paired image probe (Round 13B)

A `SmallCNN` was trained for four epochs on 10,000 CIFAR-10 training identities with seed 13. The same 2,000 test identities were evaluated at clean level 0 and five increasing levels of Gaussian noise, Gaussian blur, and contrast loss. The operational boundary was fixed as clean error plus 0.15. For identity \(i\), adjacent-level incident and recovery indicators were

\[
I_{i,s}=\mathbf 1\{e_{i,s}=0,e_{i,s+1}=1\},
\qquad
B_{i,s}=\mathbf 1\{e_{i,s}=1,e_{i,s+1}=0\}.
\]

All eight frozen gates passed: model competence, identity preservation, exact paired accounting, material risk growth, headroom exhaustion, endpoint persistence, distributed first crossings, and boundary-local ordering. Runtime was 22.9 seconds. This stage supported expansion only; it was not described as official CIFAR-10-C evidence.

### S6.3 Three-seed controlled qualification (Round 13C)

Before any new result was revealed, Round 13C froze training seeds 31, 47, and 61; a common stratified set of 20,000 training identities; all 10,000 test identities; eight training epochs; six paths; and 10,000 training-seed-cluster bootstrap draws. Each SmallCNN used four convolutional blocks with 32, 64, 96 and 128 channels; the final three convolutions used stride 2, each convolution was followed by batch normalization and ReLU, adaptive average pooling fed a 128-to-10 linear classifier, and training used cross-entropy with AdamW (learning rate \(10^{-3}\), weight decay \(10^{-4}\)), cosine annealing, batch size 256 and deterministic algorithms. The six same-identity paths were Gaussian noise, Gaussian blur, contrast loss, darkening, pixelation, and central occlusion. A fixed incorrect pseudo-label boundary was added as a specificity placebo. For each identity, an independent integer offset from 1 to 9 was generated with the frozen protocol seed and added modulo ten to the true class; the resulting wrong label was fixed across models, corruptions and severities.

Clean accuracy was 0.6762, 0.6653, and 0.6651. All 18 seed-by-path cells had material positive endpoint risk growth and exhausted the frozen headroom. The principal cluster summaries were:

| Quantity | Estimate | Descriptive three-seed cluster-bootstrap range |
|---|---:|---:|
| Endpoint risk increase | 0.4200 | [0.4087, 0.4291] |
| Endpoint persistence | 0.9534 | [0.9503, 0.9594] |
| Normalized first-crossing entropy | 0.9105 | [0.9075, 0.9160] |
| Placebo-minus-true accounting RMSE | 0.1106 | [0.1086, 0.1121] |

All nine gates passed: competence, exact true-boundary accounting, material growth, headroom exhaustion, persistence, distributed crossing times, boundary-local margin ordering, pseudo-boundary specificity, and seed-direction consistency. The result justified advancing to the official benchmark but did not itself support a claim about standard corruptions.

### S6.4 Official data integrity and frozen design (Round 13D)

Round 13D used the official CIFAR-10-C archive. Its MD5 was verified as `56bf5dcef84df0e2308c6dcbcbbd8499`. A format audit conducted before any official inference found that `labels.npy` contains 50,000 entries, five repeated 10,000-label severity blocks, rather than a single 10,000 vector. The v1 protocol and source hash were retained, the schema fact and one-line shape correction were logged, and a v2 pre-target hash was frozen. No scientific gate, model, corruption family, threshold, or endpoint changed.

The three Round 13C models were frozen. Fifteen official corruption families, five severities, and all 10,000 identities were evaluated. The threshold for each seed remained clean error plus 0.15. The measured object was adjacent-severity task-boundary transport, with cumulative net flux compared against headroom. Confidence intervals resampled the three independent training-seed clusters; they do not treat the 45 seed-by-family cells as independent models.

### S6.5 Official paired formation and internal controls (Round 13D)

All 12 gates passed. The complete gate set was archive integrity, official identity layout, model competence, exact task-boundary accounting, reproducible headroom exhaustion, exact first-crossing timing, persistent crossings, distributed crossings, boundary-local ordering, pseudo-boundary specificity, seed-direction consistency, and cross-family heterogeneity.

| Quantity | Estimate | Descriptive three-seed cluster-bootstrap range |
|---|---:|---:|
| Endpoint risk increase | 0.26858 | [0.25089, 0.27885] |
| Endpoint persistence among crossing cells | 0.88056 | [0.87349, 0.88543] |
| Normalized first-crossing entropy | 0.91901 | [0.90417, 0.93170] |
| Prior true-margin separation | 1.23488 | [1.16775, 1.27827] |
| Placebo-minus-true accounting RMSE | 0.07487 | [0.06846, 0.07882] |

The maximum adjacent-step accounting error was \(9.72\times10^{-17}\). Endpoint risk increased for all three seeds in all 15 families. Eleven families exhausted headroom for all three seeds; pixelation did so for two seeds; brightness, elastic transform, and JPEG compression did so for none. Median Cliff levels occupied four distinct severities.

These within-domain noncrossing families are important controls. Brightness raised endpoint risk by about 0.10--0.11 but retained the 0.15 headroom. Elastic transform raised risk by about 0.12 while also producing substantial recovery flow. JPEG compression raised risk by about 0.04--0.07. Therefore ordered corruption, positive risk growth, and a general accuracy reduction are not equivalent to a Cliff. A Cliff requires cumulative *net* flow to exhaust the operational headroom.

Recovery was not negligible in the crossing families either. Summed recovery across five transitions was about 0.42 for glass blur and 0.41--0.44 for snow; they crossed because incident flow was larger. Elastic transform accumulated approximately 0.41--0.42 recovery and did not cross. Raw incident counts are therefore not the conserved risk increment. The exact quantity is incident minus recovery through the trained task boundary.

Round 13D supports a second paired neural-network mechanism domain. It does not support natural temporal drift, image-domain warning, CURE-TSR identity pairing, or a universal claim across architectures and training regimes.

### S6.6 Prospective repair hypothesis and manipulation (Round 13E)

Round 13E prospectively froze a deliberately strong test: with the same 1,000-image budget and a 100% ensemble-confirmed dangerous-hit rate, broader corruption-by-class-by-first-severity coverage should outperform deeper concentration on the most severe first crossings. The selection ensemble comprised the frozen Round 13D seeds 31, 47, and 61. An example was a confirmed first crossing when at least two models changed from correct to incorrect at the same first severity. The 10,000 identities were split into 2,000 calibration identities and 8,000 sealed holdout identities.

The four arms were clean-only baseline, uniform official-domain random augmentation, the 1,000 deepest confirmed first crossings, and a round-robin selection over corruption-by-true-class-by-first-severity fragments with hazard used only within a fragment. Models were retrained from scratch for eight epochs using fresh seeds 71, 83, and 97. Each seed's common operational threshold was its baseline clean error plus 0.15. Model checkpoints, paired outputs, fit summaries, and continuation state were saved after each fit.

| Arm | Budget | Confirmed dangerous-hit rate | Unique fragments | Families | Classes | Severities | Mean pressure |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 1,000 | 0.075 | 556 | 15 | 10 | 5 | not defined |
| Hazard | 1,000 | 1.000 | 262 | 15 | 10 | 5 | 2.5504 |
| Coverage | 1,000 | 1.000 | 731 | 15 | 10 | 5 | 1.4007 |

This is a valid coverage manipulation but not an extreme narrow-versus-global contrast. The hazard arm already has a broad domain floor: every family, class, and first-crossing severity is represented. Coverage expands the occupied fragment count by 2.79-fold while reducing mean pressure.

### S6.7 Prospective repair outcomes (Round 13E)

Lower is better in the following table. Values average the three fresh seeds and 15 corruption families on the sealed 8,000-identity holdout.

| Arm | Clean error | Endpoint error | Risk area | Crossing fraction | Endpoint net flux |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.35654 | 0.60235 | 0.51465 | 0.77778 | 0.24581 |
| Random | 0.36479 | 0.56257 | 0.48614 | 0.73333 | 0.19777 |
| Coverage | 0.36004 | 0.55494 | 0.47818 | 0.66667 | 0.19490 |
| Hazard | 0.36508 | 0.54172 | 0.46972 | 0.62222 | 0.17664 |

Coverage produced a real repair relative to the baseline and random arms. Yet the three preregistered directional coverage-versus-hazard endpoints all moved in the opposite direction:

| Pre-target contrast | Estimate | Descriptive three-seed cluster-bootstrap range | Result |
|---|---:|---:|---|
| Coverage minus hazard, endpoint error | +0.01322 | [+0.01041, +0.01611] | Coverage worse |
| Coverage minus hazard, risk area | +0.00846 | [+0.00645, +0.00993] | Coverage worse |
| Coverage minus hazard, crossing fraction | +0.04444 | [0.00000, +0.06667] | Coverage not better |
| Coverage minus baseline, endpoint error | -0.04741 | [-0.05912, -0.03596] | Coverage repairs baseline |
| Coverage minus random, endpoint error | -0.00763 | [-0.00882, -0.00697] | Coverage beats random |
| Coverage minus random, risk area | -0.00797 | [-0.00929, -0.00674] | Coverage beats random |
| Coverage minus hazard, clean error | -0.00504 | [-0.00513, -0.00500] | No clean-error explanation |

The endpoint coverage-minus-hazard contrasts were +0.01313, +0.01041, and +0.01611 for seeds 71, 83, and 97. An all-positive direction across three seeds has an exact one-sided sign tail of 0.125, so the cluster-bootstrap ranges are reported as descriptive effect ranges rather than population-level confidence guarantees. The risk-area contrast was positive for all three seeds, and both contrasts were positive for 11 of 15 families after seed averaging. Hazard's corrupted-domain advantage therefore cannot be attributed to one failed seed or to better clean accuracy; hazard actually has slightly worse clean error.

### S6.8 Frozen gates and decision integrity

| Frozen gate | Result |
|---|---|
| Equal 1,000-example budgets and 100% dangerous-hit rates | PASS |
| Coverage occupies at least twice as many fragments and at least 14 families | PASS |
| Every fitted model has clean accuracy at least 45% | PASS |
| Paired incident-minus-recovery accounting error at most \(10^{-12}\) | PASS |
| Coverage beats hazard on endpoint error | **FAIL** |
| Coverage beats hazard on risk area | **FAIL** |
| Coverage reduces crossing fraction relative to hazard | **FAIL** |
| Coverage beats baseline on endpoint error | PASS |
| Coverage-hazard clean-error cost at most 0.02 | PASS |

The formal machine decision is `PARTIAL_OR_STOP (6/9)`. The maximum independently recomputed paired-accounting error over all 12 fits was below \(9.72\times10^{-17}\). The protocol, source, test, and pre-target hash files are retained unchanged. The result is not relabelled as a preregistered confirmation merely because it is scientifically informative.

### S6.9 What was confirmed, falsified, and generated

The temporal status of each statement is:

1. **Pre-target confirmation retained:** official-domain augmentation helps; coverage beats the clean-only baseline and random augmentation.
2. **Pre-target strong claim falsified:** conditional on equal budgets and confirmed dangerous hits, greater fragment coverage does not beat hazard concentration here.
3. **Prospective counterdirection observed:** hazard is better on all three fresh seeds and on all three directional endpoints.
4. **Post-target hypothesis generated:** a coverage floor is followed by allocation according to deployment exposure, boundary pressure, redundancy, and held-out transfer, as summarized by Eq. SBT19.

This ordering preserves preregistration value. Preregistration is doing the work precisely because it prevents the post-target weighted law from being presented as though it had been predicted and confirmed. The next warranted confirmatory experiment is a frozen factorial manipulation of fragment breadth and pressure distribution, with deployment-mass and net-flux weighting separately varied or matched.

### S6.10 Image-domain evidence integrity (Rounds 13A–E)

The Round 13 artifact contains the A--E protocols, v1 and v2 pre-target snapshots, correction log, source, tests, download/checksum scripts, models, raw paired outputs, selection ledger, per-fit summaries, aggregate summaries, figures, and independent audits. The official CIFAR-10-C source images are not redistributed; the verified archive checksum and download procedure are supplied. The internal manifest contains 103 hashed evidence entries. The Round 13 mechanism audit passes 10/10 checks and the Round 13E independent audit passes 15/15 checks.

## Supplementary Note 7 | Preregistered serial formation–warning–control confirmation


### S7.1 Serial claim and authoritative evidence chain

CURE-OR v2 [27,28] tests one serial claim rather than three independent experiments:

\[
H_{\mathrm{full}}=H_{\mathrm{formation}}\wedge
H_{\mathrm{warning}}\wedge H_{\mathrm{repair}}.
\tag{SC1}
\]

No downstream module rescues an upstream failure, and no endpoint, threshold, exclusion, or bootstrap unit changes after outcome reveal. The authoritative chain is:

- frozen registration: <https://osf.io/c6ygf>;
- Phase 1 blind commitment: <https://osf.io/nm3ex/files/x8vmb>;
- Phase 2 confirmatory evidence: <https://osf.io/nj3hp>.

The Phase 1 archive SHA-256 is `a883b29fecfc0ab26d3338373a3ddd080782e2ce7dff911e40de5f18a60f955a`; the blind-commit SHA-256 is `4a888493531dfa797efa765adc2057e73564eacc59a9e42c86b8ef2b27c1b237`; and the Phase 2 archive SHA-256 is `7287ccf37bc1d2dd191d59601614d052b35c1830d49d3ef693e36149fd9a9e60`.

### S7.2 Models, identities, schedules, and inferential unit

The frozen ImageNet ConvNeXt-Tiny backbone has expected weight SHA-256 `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`. Five ridge classifier heads use seeds 113, 127, 139, 151, and 163. Fifty class-balanced identities are used for calibration and 50 disjoint class-balanced identities for confirmation. Ten challenge families (2, 6, 11--18), three schedules (211, 223, 227), and 13 windows yield

\[
5\times3\times10=150
\]

confirmation paths. Schedules change only the frozen asynchronous order in which paired identities advance through severity. They expand temporal stress coverage but are not independent inferential units. The five classifier-head seeds are the inferential clusters; paths, families, schedules, identities, and windows are not resampled independently.

The confirmation uses the same public mini CURE-OR benchmark, identities, and challenge families. “Held-out-head prospective confirmation” means that the classifier-head seeds were never used to train the frozen detector and were bound before reveal. It does not mean fresh data, fresh identities, a new dataset, or a new domain.

### S7.3 H1: formation and exact paired closure

The same base identity is followed through all 13 windows. For error indicator $E_i(t)$, let $F_t$ count correct-to-error transitions and $G_t$ error-to-correct transitions. Then

\[
R_{t+1}-R_t=\frac{F_t-G_t}{50}=J_t^+-J_t^-.
\tag{SC2}
\]

Risk is the fraction of misclassified identities. The registered boundary is \(\tau=0.50\), and a persistent Cliff begins at the earliest window whose risk is at least 0.50 and remains at least 0.50 thereafter. Baseline-ineligible paths remain in raw tables but do not enter Cliff-onset or warning denominators.

H1 requires at least 30 persistent-Cliff paths, at least 30 controls, at least one Cliff for every seed, closure error at most $10^{-12}$, positive median endpoint-minus-baseline risk on Cliff paths, and passing identity, pairing, phase-order, and integrity checks. The result is:

| Formation quantity | Result |
|---|---:|
| Eligible paths | 150 |
| Persistent Cliffs | 72 |
| Non-Cliff controls | 78 |
| Maximum absolute closure error | $1.1102230246251565\times10^{-16}$ |
| Median endpoint-minus-baseline risk on Cliff paths | 0.54 |
| H1 | PASS |

This verifies exact SBT closure on every observed path and shows that cumulative harmful transport exhausts the registered headroom on the persistent-Cliff subset. It does not turn the 150 paths into 150 independent repetitions.

### S7.4 H2: frozen outcome-blind Hybrid25 warning

Hybrid25 records 25 outcome-blind batch channels per window. The 11 active channels are: (1) total departure from each identity's baseline predicted class; (2–4) current prediction-margin quantiles at 0.10, 0.50 and 0.90; (5–7) change-from-baseline margin quantiles at 0.10, 0.50 and 0.90; (8) new departure minus return to the baseline prediction; (9) persistent departure; (10) near-boundary occupancy, defined as the fraction with margin below 0.10; and (11) mean frozen-representation norm. Define \(\widetilde J_t^+\) as the fraction newly leaving the baseline predicted class and \(\widetilde J_t^-\) as the fraction returning to it. The signed channel

\[
\widetilde{\operatorname{SBT}}_t=\widetilde J_t^+-\widetilde J_t^-
\tag{SC2a}
\]

is an identity-anchored prediction-state transport proxy. It uses no confirmation labels, but it is not equal by definition to the label-based \(J_t^+-J_t^-\). The 14 committed diagnostics are the ten class-specific departure masses, current-margin quantiles at 0.25 and 0.75, new departure and recovery. These diagnostics have zero weight in the warning readout but preserve the declared 25-channel telemetry budget.

The frozen readout uses the active 11-vector, its one-window difference, its two-window slope and normalized time, for 34 inputs. A class-balanced logistic readout with \(C=0.3\), its scaler, coefficients and intercept were selected on 100 calibration paths using the registered procedure and then frozen without refitting on the five confirmation heads. The alarm threshold is 0.87, persistence is one window and the earliest possible alarm is window 1.

An alarm is timely only when the first trigger strictly precedes the persistent Cliff. A trigger on an eligible non-Cliff path is false, and a post-Cliff trigger is not timely. H2 requires pooled timely rate at least 75%, pooled false-alarm rate at most 20%, median lead at least two windows, and prespecified per-seed timely and false-alarm gates. All confirmation alarms, scores, and predictions are committed before confirmation labels are scored.

| Seed | Timely / Cliffs | False / controls | Median lead |
|---:|---:|---:|---:|
| 113 | 15/15 | 0/15 | 2 |
| 127 | 12/12 | 3/18 | 2 |
| 139 | 15/15 | 0/15 | 3 |
| 151 | 15/15 | 0/15 | 3 |
| 163 | 14/15 | 0/15 | 4 |
| **Pooled** | **71/72** | **3/78** | **3** |

The pooled timely rate is 98.61% and the false-alarm rate is 3.85%. Every seed-level warning gate passes. The allowed calibration diagnostic is 59/69 timely with 6/81 false alarms and median lead four; it is not substituted for the confirmation result.

During registered model selection on calibration paths, leave-one-family-out diagnostics achieved 72.7% timely warning with 14.3% false alarms, and leave-one-schedule-out diagnostics achieved 86.4% timely warning with 10.7% false alarms. These are internal calibration diagnostics used before confirmation; they are not independent target-domain tests and do not replace the frozen confirmation result.


### S7.5 Shortcut and group-dependence diagnostics for the frozen warning readout

After the registered analysis, mean-substitution diagnostics were applied to the committed 34-input readout. No coefficient, scaler, intercept, threshold, path or alarm rule was refitted. Removing normalized time retained 70 of 72 timely warnings and all three false alarms, whereas time alone warned none and current state alone warned one. History-only channels warned 51, and current state plus the two-window slope warned 47. These analyses measure the reliance of the registered readout; they do not estimate the best separately trained detector available from each subset.

Performance was similar across the three frozen schedules. All three false alarms occurred in challenge family 17 under classifier-head seed 127, identifying a localized family-by-seed calibration weakness rather than a schedule clock. The complete ablation table and schedule, seed and family slices are reported once in Sections S8.4–S8.5.

### S7.6 Calibration sensitivity to the operational boundary

The registered boundary \(\beta=0.50\) remains primary. A post hoc diagnostic recomputed persistent first-passage events at alternative boundaries while retaining the committed alarms and models. Timely-warning rate was 84.7% at \(\beta=0.425\), 98.6% at the registered boundary and 98.3% at \(\beta=0.55\); the corresponding false-alarm rates were 3.85%, 3.85% and 16.7%. Lower boundaries converted some committed alarms into late detections, whereas higher boundaries reclassified previously alarming paths as controls. The SBT ledger was unchanged because only the event definition moved. This supports a calibration-conditional rather than threshold-invariant interpretation. The complete boundary grid and path-level repair sensitivity are reported once in Section S8.6.

### S7.7 H3: calibration-gated full-coverage repair

The repair pool contains 100 family-by-class fragments. Five frozen examples per fragment produce a fixed 500-example full-coverage budget. The candidate update is a baseline-anchored delta-ridge head with trust penalty 1000. A seed-level repair is deployed only if calibration shows all four of the following: at least one family-level Cliff disappears, no family acquires a new Cliff, no calibration path has higher mean risk, and aggregate calibration mean-risk gain is at least 0.05. A failed gate retains the unchanged baseline.

H3 requires at least three eligible model repairs, at least two removed model–family Cliffs, zero new model–family Cliffs, at least two seeds with a removed Cliff, a model-seed cluster-bootstrap 95% interval for mean risk gain strictly above zero, and positive mean event-time gain with no Cliff coded as window 13. Twenty-thousand bootstrap draws use NumPy seed 21313.

| Seed | Repair deployed | Mean risk gain | Mean event-time gain |
|---:|:---:|---:|---:|
| 113 | yes | 0.0904615 | 2.0667 |
| 127 | no; baseline retained | 0 | 0 |
| 139 | yes | 0.0597949 | 1.0667 |
| 151 | yes | 0.0745128 | 1.3333 |
| 163 | yes | 0.0853846 | 1.2333 |

Four of five model updates are eligible. Five model–family Cliffs are removed (`113:17`, `139:17`, `151:13`, `151:17`, and `163:17`), none is introduced, and all four eligible seeds remove at least one Cliff. Mean deployment-risk gain is 0.0620308 with the registered five-seed cluster-bootstrap interval [0.0298051, 0.0852410]. Repeated bootstrap draws do not create additional independent classifier-head or backbone seeds; the interval is interpreted together with the five complete seed-level effects. Mean event-time gain is 1.14 windows with interval [0.4933333, 1.7000].

Seed 127 is a registered safety-gate abstention, not a failed replacement or excluded seed. Its deployed head remains the baseline and its zero effect remains in pooled inference. This is evidence that the no-harm guard can prevent an unsupported update.

### S7.8 Two-phase outcome separation and audit

Phase 1 trains heads, applies calibration, computes confirmation predictions, logs Hybrid25 telemetry and alarms, applies the repair gate, and commits all output hashes while `confirmation_labels_scored=false`. Phase 2 verifies the package binding and blind commitment before reading confirmation truth and evaluating H1--H3.

| Integrity item | Result |
|---|---:|
| Package audit | PASS |
| Phase 1 independent audit | 22/22 PASS |
| Phase 2 independent audit | 16/16 PASS |
| Confirmation labels scored in Phase 1 | false |
| Blind commitment preserved | true |
| Phase order preserved | true |
| Overall machine decision | `FORMATION_WARNING_REPAIR_CONFIRMED` |

### S7.9 Supported interpretation

The exact layer is the paired identity

\[
\Delta R_t=J_t^+-J_t^-.
\]

CURE-OR v2 is high-precision empirical confirmation of that accounting and of headroom exhaustion across the registered field; it is not required for the mathematical identity to hold. At the conditional observation layer, the result shows that future persistent SBT is warnable from the frozen Hybrid25 history under target calibration, a 50-identity operating population, ordered schedules, paired identity anchors, and the registered event rule. Frozen ablations rule out time-only and current-state-only explanations for this readout, while alternative-boundary sensitivity shows that performance remains calibration conditional. At the control layer, the result shows that one fixed, calibration-gated, 500-example full-coverage update can reduce risk and remove or delay Cliffs on the same future field.

The experiment does not directly estimate $b$, $H$, or \(\mathcal Q^{\mathrm{obs}}\). The repair result is consistent with the chain

\[
D_{\mathrm{train}}\rightarrow W\rightarrow
(\partial\mathcal E,b,H,\mathcal Q^{\mathrm{obs}})
\rightarrow \operatorname{SBT}\rightarrow R\rightarrow t_{\mathrm{cliff}},
\]

but it cannot be described as direct measurement of curvature change. It also does not compare coverage with pressure, hazard, or another allocation rule.

### S7.10 Scope boundaries

The result must not be described as:

- fresh-data, fresh-identity, cross-dataset, or cross-domain confirmation;
- natural longitudinal drift;
- 150 independent replications;
- a universal 25-channel warning sensor;
- a fully label-free system;
- proof that coverage is generally superior to hazard or pressure;
- validation of an external safety threshold;
- a universal repair theorem for arbitrary architectures;
- direct estimation of $b$, $H$, or \(\mathcal Q^{\mathrm{obs}}\).

The correct summary is: five held-out classifier-head seeds are the inferential units; three schedules extend temporal stress coverage; the field, identities, families, telemetry, alarm rule, and repair protocol remain benchmark specific.

### S7.11 Reproducibility package and primary files

The standalone archive `CURE_OR_V2_COMPLETE_REPRODUCIBILITY_c6ygf.zip` has SHA-256 `e3c3508eb800859baace03f6d1259cde6c4d49fdc453a902fc91d61bd196d143`. It contains the frozen source and configuration, official metadata and frozen tables, Phase 1 arrays, Phase 2 raw tables, independent audits, and final manifests. Primary analysis files are:

- `raw_outputs/path_level_results.csv` for formation and warning;
- `raw_outputs/repair_path_results.csv` for paired repair trajectories;
- `raw_outputs/seed_level_results.csv` for the five inferential clusters;
- `raw_outputs/blind_predictions.npz` for committed predictions;
- `raw_outputs/features.npz` for the frozen numerical representation;
- `raw_outputs/results.json` for the normative machine-readable decision.
- `Cliff_NMI_frozen_output_diagnostics.zip` for post hoc threshold, ablation and decomposition ledgers; these files do not alter the registered decision.

The same evidence is integrated into `Cliff_boundary_transport_code_v6.zip`, SHA-256 `751e2b1f5a10e6bd1201087475742cd338f6e4f647030db2ab83af33ab8c73e5`. The paper package does not duplicate code, raw data, models, or result archives.

## Supplementary Note 8 | Sensitivity and reliance diagnostics on committed outputs


### S8.1 Scope and temporal status

Every analysis in this note was added after completion of the primary studies. The inputs are committed prediction, risk, telemetry or selection arrays from the released evidence repository. No classifier, warning model or repair model was retrained; no new target identity or corruption realization was generated; no registered or frozen primary endpoint was replaced. The files are collected in `Cliff_NMI_frozen_output_diagnostics.zip`.

The purpose is diagnostic: separate boundary-independent transport from boundary-relative first passage, test time/current-state shortcut explanations, distinguish anchor accuracy from future transport suppression and make small-cluster interpretation explicit.

### S8.2 TorchSig first-passage sensitivity with the transport ledger fixed

The two-consecutive-window crossing rule was recomputed across alternative boundaries. The final column reports the mean first-passage window for the baseline, with noncrossing trajectories censored at 21. Boundaries below some starting risks produce window-0 events and are included only to show the full arithmetic sensitivity.

| Boundary | Support depleted | Baseline | Random broad | Cliff-aware | Baseline mean first-passage window |
|---|---|---|---|---|---|
| 0.1 | 10/10 | 10/10 | 0/10 | 0/10 | 0.00 |
| 0.125 | 10/10 | 10/10 | 0/10 | 0/10 | 0.00 |
| 0.15 | 10/10 | 10/10 | 0/10 | 0/10 | 3.00 |
| 0.175 | 10/10 | 10/10 | 0/10 | 0/10 | 4.10 |
| 0.2 | 10/10 | 10/10 | 0/10 | 0/10 | 7.00 |
| 0.225 | 10/10 | 10/10 | 0/10 | 0/10 | 10.30 |
| 0.25 | 10/10 | 10/10 | 0/10 | 0/10 | 12.30 |
| 0.275 | 10/10 | 8/10 | 0/10 | 0/10 | 14.30 |
| 0.3 | 10/10 | 8/10 | 0/10 | 0/10 | 15.60 |
| 0.325 | 10/10 | 7/10 | 0/10 | 0/10 | 16.60 |
| 0.35 | 10/10 | 6/10 | 0/10 | 0/10 | 18.10 |

The endpoint transport sum is unchanged across boundaries. Support-depleted models crossed throughout the grid. Random-broad and Cliff-aware models did not cross at any tested boundary. Baseline incidence decreased only as the line moved above the endpoints of some paths. This supports robustness of the intervention ordering while preserving the fact that cliff incidence and time are boundary relative.

### S8.3 Exact anchor-versus-future-transport decomposition

The aggregate decomposition is:

| Regime relative to baseline | Terminal gain | Anchor gain | Future SBT suppression |
|---|---|---|---|
| Support depleted | -0.05945 | -0.04685 | -0.01260 |
| Random broad | 0.32110 | 0.16715 | 0.15395 |
| Cliff-aware | 0.33310 | 0.17455 | 0.15855 |

The complete seed–path ledger is:

| Regime | Seed | Path | Terminal gain | Anchor gain | Future SBT suppression |
|---|---|---|---|---|---|
| Random-broad | 20261011 | mixed_gradient | 0.3570 | 0.1800 | 0.1770 |
| Random-broad | 20261011 | noise | 0.3675 | 0.1845 | 0.1830 |
| Random-broad | 20261012 | mixed_gradient | 0.2480 | 0.1195 | 0.1285 |
| Random-broad | 20261012 | noise | 0.2530 | 0.1235 | 0.1295 |
| Random-broad | 20261013 | mixed_gradient | 0.3395 | 0.1955 | 0.1440 |
| Random-broad | 20261013 | noise | 0.3535 | 0.1960 | 0.1575 |
| Random-broad | 20261014 | mixed_gradient | 0.3705 | 0.2015 | 0.1690 |
| Random-broad | 20261014 | noise | 0.3775 | 0.2020 | 0.1755 |
| Random-broad | 20261015 | mixed_gradient | 0.2695 | 0.1390 | 0.1305 |
| Random-broad | 20261015 | noise | 0.2750 | 0.1300 | 0.1450 |
| Cliff-aware | 20261011 | mixed_gradient | 0.3645 | 0.1840 | 0.1805 |
| Cliff-aware | 20261011 | noise | 0.3745 | 0.1905 | 0.1840 |
| Cliff-aware | 20261012 | mixed_gradient | 0.2550 | 0.1240 | 0.1310 |
| Cliff-aware | 20261012 | noise | 0.2590 | 0.1325 | 0.1265 |
| Cliff-aware | 20261013 | mixed_gradient | 0.3450 | 0.1980 | 0.1470 |
| Cliff-aware | 20261013 | noise | 0.3590 | 0.2015 | 0.1575 |
| Cliff-aware | 20261014 | mixed_gradient | 0.3720 | 0.2030 | 0.1690 |
| Cliff-aware | 20261014 | noise | 0.3785 | 0.2035 | 0.1750 |
| Cliff-aware | 20261015 | mixed_gradient | 0.3080 | 0.1575 | 0.1505 |
| Cliff-aware | 20261015 | noise | 0.3155 | 0.1510 | 0.1645 |

Future suppression is positive in all ten rows of each enrichment arm. Across the Cliff-aware rows it ranges from 0.1265 to 0.1840; across random-broad rows it ranges from 0.1285 to 0.1830. The decomposition is a telescoping identity applied to frozen risks, not a fitted mediation model.

### S8.4 Frozen Hybrid25 mean-substitution ablations

Removed feature groups were replaced by frozen calibration means, while the original scaler, coefficients, intercept, alarm threshold, earliest alarm time and persistence rule were retained.

| Variant | Timely | Rate | False / controls | False rate | Median lead | Any alarm |
|---|---|---|---|---|---|---|
| Full Hybrid25 | 71/72 | 0.986 | 3/78 | 0.038 | 3 | 75 |
| Full without time | 70/72 | 0.972 | 3/78 | 0.038 | 3 | 75 |
| History only | 51/72 | 0.708 | 2/78 | 0.026 | 4 | 60 |
| Current + slope | 47/72 | 0.653 | 3/78 | 0.038 | 4 | 70 |
| Current + difference | 18/72 | 0.250 | 0/78 | 0.000 | 1 | 44 |
| Current + time | 3/72 | 0.042 | 0/78 | 0.000 | 1 | 21 |
| Current only | 1/72 | 0.014 | 0/78 | 0.000 | 1 | 26 |
| Time only | 0/72 | 0.000 | 0/78 | 0.000 | — | 0 |

The `history only` variant retains differences and slopes but removes current state and normalized time. `Current + slope` retains the active current vector and two-window slope. Because these are no-refit ablations, a low value means that the registered detector relied on omitted channels; it does not prove that no separately trained detector could use the retained subset better.

### S8.5 Schedule, model-seed and challenge-family slices

| Schedule | Cliffs | Timely rate | Controls | False rate | Median lead |
|---|---|---|---|---|---|
| 211 | 24 | 0.958 | 26 | 0.038 | 3 |
| 223 | 24 | 1.000 | 26 | 0.038 | 3 |
| 227 | 24 | 1.000 | 26 | 0.038 | 2.5 |

| Classifier-head seed | Cliffs | Timely rate | Controls | False rate | Median lead |
|---|---|---|---|---|---|
| 113 | 15 | 1.000 | 15 | 0.000 | 2 |
| 127 | 12 | 1.000 | 18 | 0.167 | 2 |
| 139 | 15 | 1.000 | 15 | 0.000 | 3 |
| 151 | 15 | 1.000 | 15 | 0.000 | 3 |
| 163 | 15 | 0.933 | 15 | 0.000 | 4 |

| Family | Cliffs | Timely rate | Controls | False rate | Median lead |
|---|---|---|---|---|---|
| 2 | 0 | — | 15 | 0.000 | — |
| 6 | 0 | — | 15 | 0.000 | — |
| 11 | 0 | — | 15 | 0.000 | — |
| 12 | 0 | — | 15 | 0.000 | — |
| 13 | 15 | 1.000 | 0 | — | 3 |
| 14 | 15 | 1.000 | 0 | — | 1 |
| 15 | 0 | — | 15 | 0.000 | — |
| 16 | 15 | 1.000 | 0 | — | 5 |
| 17 | 12 | 1.000 | 3 | 1.000 | 8 |
| 18 | 15 | 0.933 | 0 | — | 2 |

The schedule slices are uniformly strong and therefore inconsistent with one schedule alone driving performance. The three false alarms are all family-17 controls under seed 127; the family table should not be interpreted as ten independent family experiments because family event status is highly structured in this frozen benchmark field.

### S8.6 CURE-OR boundary sensitivity with alarms fixed

| Boundary | Cliffs | Controls | Timely rate | False rate | Median lead |
|---|---|---|---|---|---|
| 0.35 | 84 | 66 | 0.393 | 0.000 | 1 |
| 0.375 | 78 | 72 | 0.538 | 0.000 | 2 |
| 0.4 | 78 | 72 | 0.667 | 0.000 | 2 |
| 0.425 | 72 | 78 | 0.847 | 0.038 | 2 |
| 0.45 | 72 | 78 | 0.944 | 0.038 | 2 |
| 0.475 | 72 | 78 | 0.972 | 0.038 | 2 |
| 0.5 | 72 | 78 | 0.986 | 0.038 | 3 |
| 0.525 | 69 | 81 | 0.986 | 0.074 | 4 |
| 0.55 | 60 | 90 | 0.983 | 0.167 | 3 |
| 0.575 | 51 | 99 | 1.000 | 0.242 | 3 |
| 0.6 | 51 | 99 | 1.000 | 0.242 | 3 |
| 0.625 | 45 | 105 | 1.000 | 0.286 | 5 |
| 0.65 | 45 | 105 | 1.000 | 0.286 | 5 |
| 0.675 | 45 | 105 | 1.000 | 0.286 | 6 |
| 0.7 | 45 | 105 | 1.000 | 0.286 | 6 |

The registered detector is calibrated to \(\beta=0.50\). Lower boundaries turn some committed alarms into late detections; higher boundaries reclassify previously alarming paths as controls and increase false alarms. This is the expected behaviour of a target-calibrated first-passage predictor and directly supports the paper's calibration-conditional, rather than threshold-invariant, warning claim.

For completeness, the deployed repair paths were also rescored at alternative boundaries:

| Boundary | Baseline path events | Repaired path events | Removed path events | New path events | Mean path-level event gain |
|---|---|---|---|---|---|
| 0.35 | 84 | 84 | 0 | 0 | 1.595 |
| 0.375 | 78 | 78 | 0 | 0 | 1.615 |
| 0.4 | 78 | 78 | 0 | 0 | 1.795 |
| 0.425 | 72 | 66 | 6 | 0 | 2.236 |
| 0.45 | 72 | 63 | 9 | 0 | 2.069 |
| 0.475 | 72 | 60 | 12 | 0 | 1.986 |
| 0.5 | 72 | 57 | 15 | 0 | 2.375 |
| 0.525 | 69 | 51 | 18 | 0 | 2.420 |
| 0.55 | 60 | 51 | 9 | 0 | 2.550 |
| 0.575 | 51 | 45 | 6 | 0 | 2.765 |
| 0.6 | 51 | 39 | 12 | 0 | 2.667 |
| 0.625 | 45 | 39 | 6 | 0 | 2.289 |
| 0.65 | 45 | 33 | 12 | 0 | 2.400 |
| 0.675 | 45 | 33 | 12 | 0 | 2.311 |
| 0.7 | 45 | 33 | 12 | 0 | 1.956 |

These are **path-level** event counts over 150 repeated stress conditions, not the registered H3 model–family endpoint. At the registered boundary, the table counts 15 removed path events across schedules, whereas the primary result counts five removed model–family cliffs after aggregation. No new path event is introduced anywhere in the grid.

### S8.7 Small-cluster interpretation

The independent intervention units are five training seeds for the principal TorchSig interventions, three training seeds for the official CIFAR repair study and five classifier-head seeds for CURE-OR. Bootstrap repetition does not increase these counts. Accordingly:

- all-positive effects over five seeds have an exact one-sided sign tail of \(1/32=0.03125\);
- the Round 12C coverage-versus-hazard result has 4/5 positive seed effects, exact tail 0.1875;
- the CIFAR coverage-versus-hazard reversal has 3/3 positive endpoint and area effects, exact tail 0.125;
- CURE-OR path rates describe performance on the frozen field, while seed-level consistency supplies replication evidence.

These sign tails are descriptive checks added after the original multiplicity families. They do not replace additional independent model or backbone seeds.

### S8.8 Matched-calibration warning diagnostics, nested prediction-state transport proxies and operating-budget sensitivity

This analysis was added after the registered H1–H3 evaluation and uses only frozen features, identities, schedules and predictions. It does not alter the registered Hybrid25 model or decision. Each comparator was fitted on the 50 calibration identities across five classifier heads, three schedules and ten families. Logistic regularization was selected by leave-one-family-out calibration. After fitting on all calibration identities, the alarm threshold maximized timely warning subject to a calibration false-alarm rate no greater than 7.5%, matching the registered calibration diagnostic of 6/81. The fitted rule was applied without parameter or threshold tuning on the disjoint 50 confirmation identities. The comparison questions were nevertheless specified after confirmation reveal, so the results are post hoc diagnostics on disjoint identities rather than independent held-out confirmation.

Feature sets were:

- **time only:** normalized window index;
- **static current telemetry:** nine current channels excluding one-step transition proxies, explicit differences, slopes and time;
- **one-step augmented telemetry:** static channels plus the two baseline-prediction transition proxies;
- **entropy–margin trend:** entropy and probability-margin levels with one- and two-step changes;
- **unsigned shift:** probability-mean displacement, probability-variance displacement, representation-norm displacement, total class-departure mass and one-step changes;
- **estimated risk:** a ridge-calibrated current-risk estimate from the active state;
- **estimated risk plus slope:** the risk estimate plus a six-window extrapolation from its two-step slope;
- **risk-proxy CUSUM:** a one-sided cumulative sum of changes in the estimated risk;
- **full temporal refit:** a newly fitted logistic model on the 34 temporal inputs used by the registered detector.

| Method | Calibration timely | Calibration cliffs | Calibration false | Calibration controls | Calibration median lead | Confirmation timely | Confirmation cliffs | Confirmation false | Confirmation controls | Confirmation median lead |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Registered Hybrid25 | 59 | 69 | 6 | 81 | 4 | 71 | 72 | 3 | 78 | 3 |
| Time only (refit) | 0 | 69 | 0 | 81 | — | 0 | 72 | 0 | 78 | — |
| Static current telemetry (refit) | 61 | 69 | 6 | 81 | 5 | 62 | 72 | 3 | 78 | 3 |
| One-step augmented telemetry (refit) | 59 | 69 | 6 | 81 | 5 | 70 | 72 | 3 | 78 | 3 |
| Entropy–margin trend (refit) | 45 | 69 | 6 | 81 | 5 | 57 | 72 | 20 | 78 | 2 |
| Unsigned shift (refit) | 52 | 69 | 6 | 81 | 3.5 | 64 | 72 | 21 | 78 | 3 |
| Hybrid25 full temporal (refit) | 58 | 69 | 6 | 81 | 5 | 61 | 72 | 1 | 78 | 3 |
| Estimated current risk | 27 | 69 | 4 | 81 | 3 | 28 | 72 | 6 | 78 | 2 |
| Estimated risk + slope | 22 | 69 | 6 | 81 | 6 | 57 | 72 | 14 | 78 | 2 |
| Risk-proxy CUSUM | 29 | 69 | 6 | 81 | 2 | 49 | 72 | 27 | 78 | 3 |

The registered detector has the highest timely count, whereas the refitted full temporal model has the fewest confirmation false alarms; no scalar utility was prespecified to rank this trade-off, and the table is not an architecture-level competition. Time alone cannot meet the matched calibration false-alarm budget while warning a cliff. Estimated current risk warns 28/72 confirmation cliffs, and generic entropy–margin and unsigned-shift comparators transfer with 20/78 and 21/78 false alarms. The reported CUSUM is univariate on a calibrated risk proxy and is not the strongest possible multivariate change-point detector over the full task-oriented telemetry.

A strictly nested comparison used the same nine static channels and added either \(\widetilde{\operatorname{SBT}}\), persistent departure or both:

| Augmented-state feature set | Confirmation timely | Confirmation false / controls | Median lead | Timely gain versus static |
|:---|---:|---:|---:|---:|
| Nine static current channels | 62/72 | 3/78 | 3 | 0 |
| Static + net prediction-state transport proxy | 71/72 | 3/78 | 3 | +9 |
| Static + persistent departure | 62/72 | 3/78 | 3 | 0 |
| Static + both proxy channels | 70/72 | 3/78 | 3 | +8 |

Differences of one event among the leading rules are within the single-event granularity of the field and are not interpreted as superiority, equivalence or harm. The net-proxy gain is distributed across classifier-head seeds:

| Seed | Static | + net proxy | + persistence | + both proxies |
|---:|---:|---:|---:|---:|
| 113 | 12/15 | 15/15 | 12/15 | 14/15 |
| 127 | 11/12 | 12/12 | 11/12 | 12/12 |
| 139 | 15/15 | 15/15 | 15/15 | 15/15 |
| 151 | 13/15 | 14/15 | 13/15 | 14/15 |
| 163 | 11/15 | 15/15 | 11/15 | 15/15 |

Relative to static telemetry, the net-proxy timely-rate difference is 0.125 with a descriptive complete-seed bootstrap range of 0.042–0.213; the both-proxy difference is 0.111 with range 0.040–0.194. These ranges do not create representation-level replication, because all heads share one backbone.

To assess dependence on the single 7.5% calibration operating point, fitted score models were held fixed and thresholds were recalibrated on calibration scores across several false-alarm budgets. Each cell reports confirmation timely /72; false /78:

| Calibration FPR budget | Static | + net proxy | + both proxies | Full temporal |
|---:|---:|---:|---:|---:|
| 5.0% | 20/72; 0/78 | 31/72; 1/78 | 32/72; 1/78 | 36/72; 0/78 |
| 7.5% | 62/72; 3/78 | 71/72; 3/78 | 70/72; 3/78 | 61/72; 1/78 |
| 10.0% | 62/72; 3/78 | 72/72; 3/78 | 72/72; 3/78 | 70/72; 3/78 |
| 15.0% | 62/72; 3/78 | 72/72; 5/78 | 72/72; 6/78 | 70/72; 3/78 |
| 20.0% | 62/72; 3/78 | 72/72; 5/78 | 72/72; 6/78 | 70/72; 3/78 |

The net-proxy state retains a timely-warning advantage over static telemetry across the 5–20% budget grid, while more permissive budgets increase false alarms. This is a post hoc operating-point sensitivity analysis, not a prospectively frozen receiver-operating comparison. The empirical result motivates dynamic sufficiency but does not prove the conditional-independence criterion in Eq. SBT18a.

### S8.9 Cliff-difficulty and calibration–confirmation regime analysis

The registered warning result on confirmation (71/72 timely; 3/78 false) is stronger than the allowed calibration diagnostic (59/69 timely; 6/81 false). To test whether confirmation contained only deeper and easier cliffs, endpoint overshoot was defined as \(R_T-0.50\), endpoint delta as \(R_T-R_0\), initial headroom as \(0.50-R_0\), and onset as the persistent-cliff window. Calibration tertiles were frozen before confirmation summaries. Endpoint-overshoot cutpoints were 0.12 and 0.38; onset cutpoints were windows 3 and 7.

| role         | stratifier         | stratum   |   n |   timely |   timely_rate |   median_lead |   median_value |
|:-------------|:-------------------|:----------|----:|---------:|--------------:|--------------:|---------------:|
| calibration  | endpoint_overshoot | low       |  24 |       22 |        0.9167 |        5.0000 |         0.0500 |
| calibration  | endpoint_overshoot | mid       |  39 |       34 |        0.8718 |        3.5000 |         0.3600 |
| calibration  | endpoint_overshoot | high      |   6 |        3 |        0.5000 |        1.0000 |         0.4000 |
| calibration  | event              | low       |  26 |       18 |        0.6923 |        1.0000 |         2.0000 |
| calibration  | event              | mid       |  27 |       25 |        0.9259 |        5.0000 |         6.0000 |
| calibration  | event              | high      |  16 |       16 |        1.0000 |        8.0000 |         9.0000 |
| calibration  | pre_event_slope    | low       |  25 |       17 |        0.6800 |        7.0000 |         0.0000 |
| calibration  | pre_event_slope    | mid       |  23 |       22 |        0.9565 |        3.0000 |         0.0300 |
| calibration  | pre_event_slope    | high      |  21 |       20 |        0.9524 |        2.0000 |         0.0700 |
| confirmation | endpoint_overshoot | low       |  27 |       27 |        1.0000 |        3.0000 |         0.0600 |
| confirmation | endpoint_overshoot | mid       |  33 |       32 |        0.9697 |        2.0000 |         0.3600 |
| confirmation | endpoint_overshoot | high      |  12 |       12 |        1.0000 |        2.0000 |         0.4000 |
| confirmation | event              | low       |  29 |       28 |        0.9655 |        1.0000 |         2.0000 |
| confirmation | event              | mid       |  25 |       25 |        1.0000 |        3.0000 |         6.0000 |
| confirmation | event              | high      |  18 |       18 |        1.0000 |        7.0000 |         8.5000 |
| confirmation | pre_event_slope    | low       |  10 |       10 |        1.0000 |        5.0000 |         0.0100 |
| confirmation | pre_event_slope    | mid       |  24 |       24 |        1.0000 |        5.0000 |         0.0400 |
| confirmation | pre_event_slope    | high      |  38 |       37 |        0.9737 |        1.0000 |         0.1050 |

The shallow endpoint-overshoot stratum contains 27 confirmation cliffs and all 27 are warned; the single missed confirmation cliff lies in the middle overshoot stratum and begins at window 3. All ten confirmation cliffs in the calibration-defined low pre-event-slope stratum were warned. Thus the confirmation rate is not explained solely by deep endpoint events. The operating regimes nevertheless differ. Calibration cliff paths have median baseline risk 0.36, initial headroom 0.14 and pre-event risk slope 0.03. Confirmation cliff paths have median baseline risk 0.28, initial headroom 0.22 and pre-event slope 0.06. Confirmation therefore begins farther below the boundary but often accumulates more sharply. The study establishes performance over these observed strata; it does not identify a minimum detectable overshoot, slope, history length or effective population.

### S8.10 Trained peer-boundary effect sizes and cluster-aware false alarms

Random and incorrect partitions are weak specificity nulls because they are not learned decision boundaries. We therefore used committed predictions to construct a trained peer-boundary placebo. For each focal CIFAR seed–corruption cell and each CURE-OR seed–schedule–family path, the peer model with closest anchor risk was selected. The peer model’s incident-minus-recovery series was used to reconstruct the focal model’s risk increments. Exact focal closure is guaranteed by the accounting identity, so inference is based on the peer error magnitude, its normalization to focal risk changes and its position between incorrect and focal boundaries rather than on a self-pass/peer-fail label.

| Domain | Subset | n | Incorrect-partition RMSE mean | Trained-peer RMSE mean | Trained-peer RMSE median | Focal-self RMSE mean | Median peer NRMSE / focal RMS increment | Median anchor-risk gap |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10-C | all | 45 | 0.074870 | 0.015389 | 0.009711 | \(3.1\times10^{-17}\) | 0.165 | 0.0002 |
| CIFAR-10-C | cliff | 35 | 0.087107 | 0.017377 | 0.012147 | \(3.4\times10^{-17}\) | 0.164 | 0.0002 |
| CURE-OR | active | 135 | — | 0.027502 | 0.026458 | 0 | 0.858 | 0.0200 |
| CURE-OR | cliff | 72 | — | 0.034334 | 0.032913 | 0 | 0.707 | 0.0200 |

On the common 45-cell CIFAR field, mean reconstruction error exhibits a graded null hierarchy:

\[
0.07487\;\text{(incorrect partition)}
>
0.01539\;\text{(closest-risk trained peer)}
\gg
3.1\times10^{-17}\;\text{(focal self)}.
\]

The trained-peer error is not trivial relative to the focal increments: its median ratio to the cell-specific root-mean-square risk increment is 0.165 over all cells and 0.164 over cliff cells. Using mean-absolute increments gives corresponding median ratios of 0.188 and 0.180. In CURE-OR, peer reconstruction error is larger relative to the coarser 50-identity risk steps, with median NRMSE 0.707 on cliff paths. These domain-specific effect sizes show that both a task-agnostic incorrect partition and a closely matched trained peer fail to preserve the focal pathwise ledger. They should not be read as a universal monotone law because the nulls differ in construction. Peer mismatch is larger relative to focal increments in CURE-OR than in CIFAR-10-C, but anchor-risk separation, increment scale and head parameterization vary simultaneously; the causal source of this cross-domain difference is unresolved. The CURE-OR field has no prespecified incorrect-partition comparator, so no cross-domain absolute RMSE ladder is asserted. Per-focal-seed summaries in the released v7 diagnostic table show that the mismatch is not confined to one CNN seed or classifier head.

The three CURE-OR false alarms all occur under classifier-head seed 127 and challenge family 17. Seed-level control counts are:

|   seed |   false |   controls |
|-------:|--------:|-----------:|
|    113 |       0 |         15 |
|    127 |       3 |         18 |
|    139 |       0 |         15 |
|    151 |       0 |         15 |
|    163 |       0 |         15 |

Resampling the five classifier-head seeds as complete clusters gives a descriptive 95% percentile range of 0–0.1071 for the pooled false-alarm fraction. The field rate 3/78 remains the operational benchmark result; the cluster range reflects uncertainty about replication across classifier heads.

## Supplementary Note 9 | Evidence hierarchy and open boundaries

### S9.1 End-to-end evidence hierarchy

The theory-to-evidence chain is

\[
\begin{aligned}
D_{\mathrm{train}}
&\longrightarrow W,\partial\mathcal E_W,(b,H,\mathcal Q^{\mathrm{obs}}),\\
(P_t,v_t)\times\partial\mathcal E_W
&\longrightarrow \mathcal L_t
\longrightarrow \operatorname{SBT}_t
\longrightarrow \text{headroom depletion}
\longrightarrow \text{cliff},\\
\text{identity-anchored outcome-blind telemetry}
&\longrightarrow \widehat R_t,\widetilde{\operatorname{SBT}}_t,\widehat{C_{t,L}^{\mathcal P}}
\longrightarrow \text{advance warning or abstention},\\
\text{training-support intervention}
&\longrightarrow D'_{\mathrm{train}}
\longrightarrow \text{changed anchor risk and future SBT}.
\end{aligned}
\]

Here \(\mathcal L_t\) is the resolved ledger. Scalar SBT closes accounting, while incident/recovery, turnover, fragments, timing and path-conditioned persistence provide mechanism resolution.

| Evidence | Role | Supported conclusion | Scope limit |
|---|---|---|---|
| Exact paired identity | Mathematical accounting | Risk increments equal incident minus recovery for fixed deterministic classifiers | No prediction or causal mechanism from closure alone |
| Covertype field exploration | Unpaired distribution-level falsification | Signed margin transport is more informative than unsigned distance in the terminal holdout | No longitudinal identity flow or complete formal pass |
| TorchSig temporal-order intervention | Warning-mechanism identification | Ordered history adds prospective information beyond a preserved state inventory and terminal state | Counterfactual histories need not be physically realizable |
| TorchSig control-blind chart | Observation test | Named physical coordinates are unnecessary in the frozen field | Labelled calibration remains required |
| TorchSig common-stream intervention | Upstream control | Training support changes local risk geometry, anchor risk and future cumulative SBT | Five seeds and two controlled paths |
| TorchSig paired transport | Sample-resolved formation | Temporally distributed focal-boundary transport assembles crossing under ordered paths and is reduced by training | ExtraTrees and one synthetic generator; persistence is path conditioned |
| Official CIFAR-10-C transport | Second paired neural field | Focal-boundary transitions resolve incident, recovery, timing and headroom exhaustion | Three small CNN seeds; ordered severity, not natural time |
| Trained peer-boundary effect sizes | Specificity audit | Incorrect partitions > closest-risk peers > focal self in CIFAR; normalized peer error remains material in CURE-OR | Post hoc, domain-specific analysis |
| Refitted warning baselines | Comparative observation audit | Task-oriented telemetry outperforms time-only, risk-only and generic shift proxies | Post hoc; current active state nearly matches registered temporal readout |
| Cliff-difficulty stratification | Operating-regime audit | Confirmation performance is strong in shallow-overshoot, low-slope and early-onset strata | Calibration and confirmation regimes differ |
| TorchSig sparse-field repair | Bounded set intervention | Coverage can beat severe hazard concentration when declared support is sparse | 4/5 seed directions; no universal allocation result |
| CIFAR repair reversal | Prospective falsification | Coverage is useful but not universally sufficient | Moderator remains confounded and unconfirmed |
| CURE-OR two-phase confirmation | Preregistered serial loop | Formation, warning and calibration-gated control pass on one field | Five head seeds conditional on one frozen representation and one 50-identity unit |
| Anchor/SBT decomposition | Intervention audit | A substantial share of enrichment gain occurs after the anchor through future SBT suppression | Arithmetic decomposition, not broad mediation |

The evidential peaks are asymmetric. **Formation** has the strongest status: exact paired accounting plus temporally distributed, focal-boundary-specific and intervention-responsive ledgers, with persistence conditioned on the ordered paths studied. **Warning** is conditional: temporal-order intervention identifies one prospective component in TorchSig, while CURE-OR confirms task-oriented outcome-blind telemetry at one calibration, horizon and 50-identity unit. Its strictly nested augmented-state comparison shows that one-step net departure minus return to the baseline prediction can make the fitted state approach dynamic sufficiency, so explicit multi-window history is not uniquely necessary in that field. This channel is an outcome-blind proxy, not exact task-error SBT. **Control** is intervention-supported but domain bounded. **Allocation** remains unresolved: the two repair studies falsify simple universal rules without identifying the replacement moderator.

### S9.2 Established findings, abstentions and open boundaries

| Status | Claim or question | Evidence status in v7 | Consequence for interpretation |
|---|---|---|---|
| Established exactly | Paired zero–one risk accounting | Exact for fixed deterministic classifiers; numerically closed in all paired fields | Exact within fixed-classifier, paired-identity scope |
| Supported in tested paired fields | Cliff formation | Temporally distributed, focal-boundary-specific transport with path-conditioned persistence exhausts declared headroom | Supports formation without synchrony or risk discontinuity under tested paths |
| Conditionally supported | Outcome-blind warning | Supported for frozen TorchSig and CURE-OR telemetry, calibrations, histories and units | No universal sensor, history requirement or threshold invariance |
| Supported within tested interventions | Training support controls future transport | Common-stream retraining changes anchor risk, geometry and post-anchor cumulative SBT | No universal retraining efficacy or targeting law |
| Falsified | Universal coverage-first repair | Prospective CIFAR reversal contradicts universal extension of sparse RF result | Replacement factors remain hypotheses |
| Abstention retained | CURE-OR seed 127 update | Frozen no-harm gate retained baseline | Abstention is intended guarded-control output |
| Open | Natural-time and fresh-identity deployment | No such confirmation is present | Requires new prospective data |
| Open | Representation-level CURE-OR replication | Five classifier heads share one frozen ConvNeXt-Tiny representation | Requires independent backbones or representation seeds |
| Open | Non-monotone persistence | Principal paired paths are ordered challenge ladders | Requires dwell–reversal or round-trip paths |
| Open | General losses and moving/adaptive classifiers | Exact discrete law is for fixed deterministic zero–one classification | Requires expanded loss and boundary-motion theory |
| Open | Universal sensor or minimum effective population | Warning was tested at fixed operating units | Requires a prospective phase diagram |
| Open | Coverage-by-pressure factorization | Orthogonal factorial has not been run | Must not be presented as a confirmed allocation law |

The open rows are future tests, not hidden premises of established or conditionally supported claims.

## Supplementary Note 10 | Reproducibility artifacts and evidence map


### S10.1 Artifact and repository map

The reader-facing paper package contains only the manuscript and Supplementary Information. Code, raw outputs, models and complete audit records remain in the separate evidence archives listed below.

**Top-level archives**

- Unified code-and-evidence repository: `Cliff_boundary_transport_code_v6.zip`
- Standalone CURE-OR v2 reproducibility archive: `CURE_OR_V2_COMPLETE_REPRODUCIBILITY_c6ygf.zip`
- Frozen-output diagnostic archive: `Cliff_NMI_frozen_output_diagnostics.zip`
- v6 committed-output comparative diagnostics: `Cliff_NMI_v6_posthoc_diagnostics.zip` (SHA-256 `07d83a591f1728a12e08e375057883e7785112aad71c905e287366e1ac61352d`)
- v7 post hoc diagnostic addendum: `Cliff_NMI_v7_posthoc_diagnostics.zip` (SHA-256 `5a36bd5577ac1b03bc302bfbe580ffb1af69d707ccc3b5332112a7574b099730`)
- v7 executable diagnostic script: `build_v7_diagnostics.py`
- Fair warning comparison: `warning_fair_baselines.csv`
- Strictly nested current-state comparison: `warning_nested_channel_ablation.csv`
- Cliff-difficulty rows and summaries: `cure_or_cliff_difficulty_rows.csv`, `cure_or_cliff_difficulty_summary.csv`
- Trained peer-boundary placebo: `trained_peer_boundary_placebo_rows.csv`, `trained_peer_boundary_placebo_summary.csv`
- Domain-normalized peer-boundary effects: `trained_peer_boundary_normalized_rows.csv`, `trained_peer_boundary_normalized_summary.csv`, `cifar_boundary_specificity_gradient.csv`
- Cluster-aware false alarms: `registered_false_alarm_by_seed.csv`, `registered_false_alarm_cluster_range.json`
- Diagnostic reports and guide: `V6_DIAGNOSTIC_REPORT.md`, `V6_DIAGNOSTIC_ADDENDUM.md`, `README.md`
- v7 nested-proxy seed summaries: `warning_nested_channel_by_seed.csv`, `warning_nested_channel_seed_cluster_ranges.csv`, `warning_nested_channel_path_discordance.csv`
- v7 operating-budget sensitivity: `warning_false_budget_sensitivity.csv`
- v7 per-seed trained-peer summaries: `trained_peer_boundary_by_seed.csv`, `V7_DIAGNOSTIC_ADDENDUM.md`

**TorchSig and Covertype evidence**

- Covertype evidence root: `covertype/`
- TorchSig qualification, warning and intervention root: `torchsig/`
- Paired transport synthesis: `ROUND11_TORCHSIG_PAIRED_BOUNDARY_FLUX.md`
- Official-source paired transport report: `ROUND11C_TORCHSIG_OFFICIAL_SOURCE_FLUX.md`
- Repair-selector ledger: `ROUND12_CONTROL_FREE_REPAIR_SMOKE_LEDGER.md`
- Sparse-field repair protocol and report: `ROUND12C_COVERAGE_VS_HAZARD_PILOT_PROTOCOL.md`, `ROUND12C_COVERAGE_VS_HAZARD_PILOT.md`
- Covertype synthesis, ledger and claim boundary: `SCIENTIFIC_SYNTHESIS_EN.md`, `EXPERIMENT_LEDGER.md`, `CLAIM_BOUNDARIES.md`

**Image-domain evidence**

- Round 13 root: `round13_second_domain/`
- Qualification and pilot protocols: `ROUND13_PROTOCOL.md`, `ROUND13C_PILOT_PROTOCOL.md`
- Official benchmark protocol and schema correction: `ROUND13D_OFFICIAL_PROTOCOL.md`, `ROUND13D_SCHEMA_CORRECTION.md`
- Repair protocol and reports: `ROUND13E_REPAIR_PROTOCOL.md`, `ROUND13_SERIES_REPORT.md`, `ROUND13E_REPAIR_REPORT.md`
- Official formation summary and paired output: `results/cifar10c_official_v1/summary.json`, `paired_outputs.npz`
- Repair summary, selections and independent audit: `results/round13e_formal_v1/summary.json`, `selections.json`, `independent_audit.json`

**CURE-OR evidence**

- Complete protocol and conclusion report: `docs/EXPERIMENT_PROTOCOL.md`, `docs/RESULTS_AND_CONCLUSIONS.md`
- Normative machine-readable result: `raw_outputs/results.json`
- Path, repair and seed tables: `raw_outputs/path_level_results.csv`, `repair_path_results.csv`, `seed_level_results.csv`
- Phase audits: `audit/PHASE1_AUDIT.json`, `audit/PHASE2_AUDIT.json`
- OSF registration and two-phase evidence chain: <https://osf.io/c6ygf>, <https://osf.io/nm3ex/files/x8vmb>, <https://osf.io/nj3hp>

**Frozen-output diagnostic ledgers**

- Threshold analyses: `round10_threshold_summary.csv`, `round10_threshold_rows.csv`, `cure_or_threshold_sensitivity_exact.csv`, `cure_or_repair_threshold_sensitivity.csv`
- Warning ablations and slices: `cure_or_warning_ablation_summary.csv`, `cure_or_warning_ablation_rows.csv`, `cure_or_warning_group_slices.csv`
- Anchor/transport decomposition: `round10_anchor_transport_decomposition.csv`, `round10_anchor_transport_decomposition_seed_path.csv`

Stopped decisions, invalidations, source files, model checkpoints and SHA-256 manifests remain in the evidence archives. Source benchmark images are not redistributed where unnecessary; checksum-verified download procedures are provided instead.
