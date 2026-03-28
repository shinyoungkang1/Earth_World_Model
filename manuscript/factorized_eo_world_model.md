# Factorized EO World Model

## Working Title

Factorized Latent JEPA for Earth Observation: Persistent Scene Latents, Temporal State Latents, and Transition-Aware Forecasting

## One-Sentence Pitch

We propose a JEPA-style Earth observation world model that explicitly factorizes a sequence into a persistent scene latent `z`, a time-varying state latent `z_t`, and an optional transition latent `delta_t`, enabling better representation learning, forecasting, and modular decoding of downstream geophysical variables.

## Abstract

Earth observation sequences contain at least two distinct kinds of information: persistent scene structure and time-varying environmental state. Standard self-supervised predictive models typically entangle these factors inside a single latent space, making it difficult to support both robust semantic representation learning and temporally faithful forecasting. We propose a factorized latent Joint-Embedding Predictive Architecture (JEPA) for Earth observation that separates persistent scene identity from temporal state evolution. Given a multi-sensor sequence, the model learns a global scene latent `z` that captures stable spatial semantics, a sequence of state latents `z_t` that capture time-specific environmental conditions, and an optional transition latent `delta_t` that represents change between timesteps. The model is trained to predict teacher latents for masked observations rather than reconstruct pixels, preserving the JEPA philosophy while introducing stronger temporal inductive bias. A transition module rolls state latents forward under irregular time gaps, and lightweight sensor or task-specific decoders can be attached to forecast future environmental targets such as land-surface temperature or other variables derived from external Earth observation and geospatial products. This framing turns the model from a regularizer-driven representation learner into a structured world model for multi-sensor Earth observation.

## 1. Motivation

### 1.1 Problem

Earth observation data is temporally structured but not fully video-like. It has:

- irregular revisit intervals
- missing observations
- multiple sensors with different resolutions and noise models
- persistent scene content mixed with transient environmental dynamics

A single latent representation is often forced to encode:

- what the place is
- what is happening at this time
- how the state is changing

That is a poor inductive bias for Earth observation.

### 1.2 Thesis

The latent state of an EO sequence should be decomposed into:

- `z`: a persistent scene latent
- `z_t`: a time-varying environmental state latent
- `delta_t`: an optional transition latent or change code

This decomposition should improve:

- semantic retrieval and representation quality
- temporal forecasting
- robustness under missing sensors
- modular decoding of downstream variables

## 2. High-Level Model

Given an input sequence

`x = {x_t^(m)} for t = 1..T and sensors m in M`

the model produces:

- teacher token targets for masked predictive learning
- a global scene latent `z`
- per-timestep state latents `z_t`
- optional transition latents `delta_t`

The world model is trained with a JEPA-style objective:

- the student sees only visible observations
- the teacher sees the full sequence
- the student predicts teacher latents for masked parts of the sequence

The key difference from standard JEPA is that prediction is factorized through structured latents rather than being treated as one undifferentiated embedding space.

## 3. Latent Variables

### 3.1 Persistent Scene Latent `z`

`z` should capture stable properties of a location:

- spatial layout
- land-cover composition
- terrain-conditioned appearance
- long-lived infrastructure or field geometry

`z` should be relatively invariant across:

- nearby timesteps
- partial cloud differences
- moderate seasonal variation
- sensor changes when observing the same place

### 3.2 Temporal State Latent `z_t`

`z_t` captures time-specific environmental condition:

- vegetation phase
- moisture or thermal condition
- transient weather-related appearance
- recent disturbance or recovery state

`z_t` should vary over time and should be the main carrier of predictive temporal information.

### 3.3 Transition Latent `delta_t`

Optional:

- `delta_t = h(z_t, z_{t+1}, Delta t, meta_t)`

or predicted forward:

- `z_{t+1} = z_t + delta_t`

`delta_t` is useful when the paper wants an explicit change representation rather than an implicit recurrent update.

## 4. Architecture

### 4.1 Sensor Adapters

Each sensor has a lightweight adapter that maps raw observations into a shared backbone space.

Examples:

- Sentinel-2 adapter
- Sentinel-1 adapter
- HLS adapter
- future Planet adapter

The backbone should not need to be retrained from scratch whenever a new sensor is added. New sensors should be attachable as observation adapters.

### 4.1.1 Missing Sensor Robustness

The model should explicitly support cases where one or more sensors are unavailable at inference time.

Examples:

- Sentinel-2 is present but Sentinel-1 is missing
- Sentinel-1 is present but Sentinel-2 is missing
- only HLS is available for a region
- a sensor is available for only a subset of timesteps

This should not be handled as an afterthought. It should be a training-time design goal.

Recommended mechanism:

- separate per-sensor adapters
- explicit sensor presence flags
- missing-modality embeddings
- sensor dropout during training

Sensor dropout should be applied to the student branch while the teacher still sees the full available observation set. This teaches the student to infer `z` and `z_t` from incomplete evidence while keeping the teacher latent target strong.

Recommended dropout schedules:

- sequence-level dropout:
  - drop S1 for the whole clip
  - drop S2 for the whole clip
  - occasionally keep both
- optional timestep-level dropout:
  - simulate partial availability across time

This gives a clean claim:

- the world model is robust to missing known sensors
- sensor absence is treated as structured uncertainty, not just corruption

### 4.1.2 Adding New Sensors After Pretraining

Missing an already-known sensor and adding a brand-new sensor are different problems.

For a new sensor `m_new`, we should not require full retraining of the world model. Instead, the model should be decomposed into:

- sensor adapter `e_m`
- latent world model `f`
- optional sensor decoder `d_m`

Then new-sensor integration becomes:

1. keep most of the latent world model fixed
2. add a new adapter `e_m_new`
3. align it using co-located and co-temporal data
4. optionally add a decoder `d_m_new`

This is the right foundation-model story:

- the world model learns sensor-agnostic latent state
- new sensors become interfaces into the latent state
- downstream tasks become lightweight decoder problems

Examples:

- add HLS after S1/S2 pretraining
- add a thermal sensor later
- add a future Planet sensor family later
- add meteorological or GEE-derived products as decoder targets

### 4.1.3 Current Implementation vs Long-Term Version

Current practical version:

- S2 adapter
- S1 adapter
- HLS adapter
- cross-attention fusion for paired S1/S2

Long-term version:

- a sensor-set encoder rather than a fixed pairwise fusion block
- one adapter per sensor family
- sensor metadata tokens for:
  - sensor id
  - time offset
  - resolution
  - uncertainty / quality
- set attention or graph attention over sensor evidence

The long-term design is what makes the model extensible beyond a fixed S1/S2 training setup.

### 4.2 Spatiotemporal Backbone

The backbone processes the multi-sensor sequence into token-level embeddings.

This remains close to the current EWM design:

- per-sensor spatial tokenization
- temporal metadata encoding
- temporal transformer or factorized temporal block
- dense or pooled token handling

### 4.3 Scene Head

A scene pooling head aggregates the sequence into a persistent latent:

- `z = Pool_scene(H_1, ..., H_T)`

Possible implementations:

- temporal attention pooling
- masked mean pooling with learned query
- persistence-aware pooling over high-confidence timesteps

### 4.4 State Head

A state head produces timestep-level latent states:

- `z_t = f_state(H_t, z, time_meta_t)`

`z_t` may be derived from:

- pooled timestep tokens
- cross-attention from a learned state query
- a recurrent state update over timeline features

### 4.5 Transition Module

A transition module predicts the future state:

- `z_hat_{t+1} = g_trans(z_t, Delta t, sensor_meta_t, z)`

or

- `delta_hat_t = g_delta(z_t, Delta t, sensor_meta_t, z)`
- `z_hat_{t+1} = z_t + delta_hat_t`

This can be:

- GRU-style latent dynamics
- MLP transition with time-gap conditioning
- state-space or linear dynamical module

### 4.6 Masked State Predictor

The student predicts masked timestep states from visible context:

- visible states
- scene latent `z`
- time-gap metadata
- sensor timing metadata

This is the structured analogue of standard JEPA masked latent prediction.

### 4.7 Masked Token Predictor

Masked teacher token embeddings are predicted from:

- persistent scene latent `z`
- predicted masked state latent `z_t`
- position / time / sensor query metadata

This preserves the JEPA principle:

- predict latent targets
- do not reconstruct pixels

### 4.8 Optional Decoders

Task-specific decoders map latent states to downstream variables:

- temperature
- soil moisture
- vegetation stress
- methane proxy
- biomass-related targets
- change maps

These decoders operate on either:

- current `z_t`
- future rolled-out `z_hat_{t+k}`
- a pair `(z, z_t)` if persistent context matters

## 5. Why This Is Still JEPA-Like

Yes, this remains JEPA-like as long as the training target is a teacher latent rather than pixel reconstruction.

The model is still JEPA if:

- the student sees partial context
- the teacher sees the full sequence
- the student predicts teacher embeddings / states / tokens in latent space

The proposal is not a return to autoencoding. It is a more structured predictive latent architecture.

## 6. Formal Objective

Let:

- `y_t^teacher` be teacher token targets for masked regions
- `s_t^teacher` be teacher timestep state targets
- `z^teacher` be a teacher scene latent

The total training objective is:

`L = lambda_tok * L_mask_token`
`  + lambda_state * L_mask_state`
`  + lambda_dyn * L_dynamics`
`  + lambda_scene * L_scene_consistency`
`  + lambda_sensor * L_sensor_consistency`
`  + lambda_delta * L_delta_regularization`
`  + lambda_sup * L_decoder_supervision`

### 6.1 Masked Token Latent Prediction

`L_mask_token`

- predicts masked teacher token embeddings
- analogous to the existing JEPA-style target

Recommended loss:

- cosine or normalized MSE in latent space

### 6.2 Masked State Prediction

`L_mask_state`

- predicts masked teacher timestep states `s_t^teacher`
- enforces temporal latent quality directly

This is one of the key departures from the current setup and should be one of the paper's main ideas.

### 6.3 Dynamics Loss

`L_dynamics`

- encourages rolled-out future states to match teacher future states
- handles irregular revisit via explicit `Delta t` conditioning

Example:

- `L_dynamics = sum_k || z_hat_{t+k} - z_{t+k}^teacher ||`

### 6.4 Scene Consistency

`L_scene_consistency`

The scene latent should be stable under:

- different visible masks
- subclips of the same sequence
- different sensors of the same place

Example:

- consistency between scene latents from overlapping subclips
- consistency between S1-only and S2-only views when both are available

### 6.5 Sensor Consistency

`L_sensor_consistency`

Encourages the persistent representation to remain compatible across sensors while allowing state-specific variation.

This is better framed as a representation consistency term than a generic decorrelation penalty.

### 6.6 Transition Sparsity or Energy

`L_delta_regularization`

If `delta_t` is explicit, we may want:

- small deltas when little changes
- robust large deltas when real events occur

Candidate forms:

- L1 penalty on `delta_t`
- Huber penalty
- change-aware weighting if supervision exists

### 6.7 Decoder Supervision

`L_decoder_supervision`

Optional supervised targets decoded from current or future latent states.

Examples:

- decode land surface temperature from `z_t`
- decode future temperature from `z_hat_{t+k}`
- decode geophysical products from external EO datasets

## 7. Downstream Variable Decoding

This model naturally supports a second-stage program:

1. pretrain a strong EO world model
2. freeze or partially tune the backbone
3. attach lightweight heads for specific targets

Examples of targets:

- land surface temperature
- vegetation indices
- evapotranspiration proxies
- methane proxy targets
- weather-conditioned surface variables

This can use:

- Planet products
- public EO products
- GEE-derived supervisory variables
- reanalysis products aligned to EO footprints

### 7.1 Why This Is Important

This reframes the contribution away from:

- "we made JEPA train better"

and toward:

- "we learned a reusable world model whose latent state supports modular environmental prediction"

That is much more compelling for a top conference.

## 8. Forecasting vs Planning

### 8.1 What We Have Without Actions

Without an action variable, this model is:

- a world model
- a forecasting model
- a counterfactual latent simulator under observed temporal conditions

It is not true planning in the V-JEPA 2 action-conditioned sense.

### 8.1.1 Stronger Near-Term Claim

The strongest honest near-term claim is:

- structured latent world modeling
- future state forecasting
- counterfactual rollout under different observation schedules
- active observation value estimation

This is already meaningful and publishable without overstating the method as full planning.

### 8.2 What Would Be Needed for Planning

To support real planning, we need:

- an action or control variable `a_t`
- a transition model `p(z_{t+1} | z_t, z, a_t, Delta t)`
- a goal or utility in latent or decoded space
- an optimizer or search procedure over action sequences

Then we can do:

- goal-conditioned rollout
- latent model predictive control
- sensor scheduling / active acquisition planning
- intervention planning if action labels exist

### 8.2.1 Two Different Kinds of Actions

There are two distinct planning settings.

#### A. Observation Planning

The action does not change the Earth state directly. It changes what we choose to observe.

Examples:

- which sensor to query next
- when to revisit next
- which area to observe next
- what observation budget to spend

This is the most realistic first planning setting for EO.

Formally:

- latent transition:
  - `z_{t+1} = f(z, z_t, Delta t)`
- observation policy / utility:
  - `u(a_t | z, z_t) = planner(z, z_t, a_t, Delta t, budget_t)`

where `a_t` could encode:

- candidate sensor
- revisit gap
- location or tile id
- cost or acquisition budget

This is better described as:

- active sensing
- observation scheduling
- acquisition planning

#### B. Intervention Planning

The action changes the world state.

Examples:

- irrigation decisions
- reservoir release
- forest treatment
- agricultural management

Formally:

- `z_{t+1} = f(z, z_t, a_t, Delta t)`

This is true control, but it requires action-outcome supervision. It is a later extension, not the current paper's main claim.

### 8.2.2 Action Representation

For action-conditioned observation planning, the action token should include:

- sensor identity embedding
- proposed time delta
- optional budget / cost scalar
- optional region or mission metadata

This enables a planning head to score candidate future observations:

- `score(a_t) = g_plan(z, z_t, a_t, Delta t, cost_t)`

The model can then:

1. predict or roll out a future latent state
2. score a set of candidate observation actions
3. choose the highest-utility action under a budget

### 8.2.3 What the Planner Optimizes

Candidate utility functions:

- expected reduction in state uncertainty
- expected improvement in downstream decoder accuracy
- expected improvement in future forecast quality
- change-detection sensitivity per unit cost

This is the key connection between a world model and a planning story. The planner is useful when it can trade off information gain against acquisition cost.

### 8.3 Honest Claim

The right current claim is:

- structured latent forecasting
- not full planning

If we later add action channels, then planning becomes a real extension.

### 8.4 Recommended Progression

The recommended research progression is:

1. train the factorized world model without actions
2. make it robust to missing sensors via sensor dropout
3. support post-hoc sensor adapters for new sensors
4. add an action-conditioned observation-planning head
5. only later move to intervention planning if action labels exist

## 9. What Makes This Publishable

### 9.1 Stronger Core Claim

A stronger paper claim is:

"Earth observation requires separating persistent scene identity from temporal environmental state; factorized latent JEPA improves both predictive world modeling and downstream variable decoding."

This is stronger than:

- a loss-function paper
- a regularization ablation paper

### 9.2 Stronger Empirical Story

We can evaluate:

- semantic retrieval from `z`
- forecasting from `z_t`
- change sensitivity from `delta_t`
- cross-sensor robustness
- supervised decoding from future latent rollouts

## 10. Critical Ablations

Main ablations should be:

1. Single-latent JEPA baseline
2. `z + z_t`
3. `z + z_t + dynamics`
4. `z + z_t + delta_t`
5. No scene consistency loss
6. No dynamics loss
7. No time-gap conditioning
8. No cross-sensor conditioning
9. Decoder-only downstream adaptation vs full fine-tuning

Important reviewer-facing controls:

10. Same parameter count control
11. Same training budget control
12. Same decoder budget control

## 11. Reviewer Questions We Must Be Ready For

### Q1. Why can one latent not do this already?

Answer:

- a single latent is forced to simultaneously encode static semantics and temporal change
- EO has stronger persistence than ordinary video
- explicit factorization matches the data-generating structure better

### Q2. What prevents `z` and `z_t` from collapsing into the same thing?

Answer:

- `z` is trained with persistence and sensor-consistency constraints
- `z_t` is trained with masked state prediction and dynamics consistency
- downstream supervision further separates stable vs time-varying information

### Q3. Is this still JEPA?

Answer:

- yes, because the target remains a teacher latent
- we do not reconstruct pixels
- the student predicts masked latent targets from partial context

### Q4. Is `delta_t` really necessary?

Answer:

- not necessarily
- it should be introduced as an optional extension or later ablation
- the main method should likely be `z + z_t + dynamics`

### Q5. Is there planning?

Answer:

- not in the strict action-conditioned sense
- the current model supports forecasting and counterfactual rollout
- planning requires action variables and goal optimization

## 12. Proposed Implementation in This Repo

### 12.1 Reuse

Keep:

- current spatiotemporal backbone
- current temporal metadata handling
- current JEPA-style student/teacher training loop
- current latent rollout scaffolding

Relevant files:

- `earth_world_model/src/ewm/models/world_model.py`
- `earth_world_model/train_tpu.py`

### 12.2 New Modules

Add:

- `scene_pool_head`
- `state_head`
- `masked_state_predictor`
- `state_to_token_decoder`
- optional `delta_transition_head`
- `sensor_adapter_registry`
- `sensor_dropout` augmentation in training
- optional `observation_planning_head`
- optional `action_conditioned_transition_head`
- optional supervised task decoders

### 12.3 Training Refactor

Refactor the current training loop so that:

- the primary method is the factorized latent JEPA
- regularization is auxiliary, not the main contribution
- dynamics is a first-class part of the architecture

### 12.4 Missing-Sensor Training

Add a training-only sensor dropout module that:

- drops S1 and/or S2 on the student branch
- optionally drops HLS when present
- preserves teacher access to full available sensors
- logs the realized dropout pattern

This should be configurable by:

- sequence-level dropout probability per sensor
- timestep-level dropout probability per sensor
- whether at least one sensor must remain visible

### 12.5 Sensor Adapter Interface

Refactor the encoder so that sensors are handled through an adapter interface rather than hardcoded pairwise logic only.

Minimum interface:

- adapter name
- input channels
- supported tokenizer modes
- projection module(s)
- optional missing-modality embedding

This should cover current sensors:

- `s2`
- `s1`
- `hls`

and leave room for future sensors:

- `planet`
- `thermal`
- `meteo`

### 12.6 Observation Planning Extension

Add an optional planning head that scores observation actions from latent state.

Input:

- scene latent `z`
- latest or queried state latent `z_t`
- candidate action token:
  - sensor id
  - time delta
  - optional cost

Outputs:

- utility score per candidate action
- optional action-conditioned future state prediction

This allows a later paper stage to ask:

- which sensor should we query next?
- when should we revisit?
- which observation has the highest value under a cost budget?

## 13. Minimal Version for Paper Submission

To keep the first submission focused, the minimal paper should be:

- a factorized latent JEPA
- persistent scene latent `z`
- temporal state latent `z_t`
- dynamics rollout on `z_t`
- optional downstream temperature decoding as an application

I would keep `delta_t` as:

- a lightweight extension
- or a late ablation

not the main flagship idea

## 14. Recommended Positioning

Best positioning for a top conference:

- not "better self-supervised regularization"
- not "another JEPA loss tweak"
- but "a structured latent world model for multi-sensor Earth observation"

That is the most defensible and ambitious version of the idea.

## 15. Immediate Next Steps

1. Define the exact latent factorization:
   - `z`
   - `z_t`
   - optional `delta_t`
2. Add missing-sensor training through student-only sensor dropout.
3. Add a sensor adapter interface so new sensors can be attached later.
4. Add an action-conditioned observation-planning head.
5. Write the method equations cleanly.
6. Decide the minimal submission version:
   - likely `z + z_t + dynamics`
7. Implement the latent heads and state prediction path in the codebase.
8. Design the first downstream decoder task:
   - temperature is a good candidate if aligned supervision is available.

## 16. Planned Training Runs

For the `yearly_10000 + ssl4eo_50000` study, the intended run order is:

1. `ema_single_latent`
   - EMA JEPA baseline
   - no factorized latents
   - no dynamics, no delta, no planning

2. `factorized_z_zt_v1`
   - scene latent `z`
   - per-timestep state latent `z_t`
   - masked token loss, masked state loss, scene consistency, latent rollout

3. `factorized_z_zt_delta_v2`
   - `z + z_t` plus explicit random-horizon `delta_t`
   - one to four step delta supervision

4. `factorized_z_zt_delta_v2_sensor_dropout`
   - run `#3` plus student-only sensor dropout
   - intended to make the latent state robust to missing `S1` or `S2`

5. `factorized_z_zt_delta_v2_sensor_dropout_observation_planning`
   - run `#4` plus an action-conditioned observation-planning auxiliary loss
   - the action is a candidate future observation defined by:
     - sensor identity
     - time delta
     - acquisition cost
   - the model predicts the future latent state for the realized future observation and learns a planning score for candidate actions

The observation-planning training signal is self-supervised. For a sampled horizon, the model uses the current latent state and scene latent, then:

- predicts the future latent state under the realized future observation action
- trains the transition head against the teacher future latent
- trains the planning score head against a utility target built from:
  - whether the candidate sensor is actually available at that future step
  - the predicted latent quality for that action
  - the configured acquisition cost

This is not yet intervention planning. It is observation planning / active sensing pretraining.
