"""Build the full process record from existing artifacts; never trains a model."""
import csv
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
ART = PROJECT / 'artifacts'
DOC = ART / 'documentation'
NAMES = ['DataSense', 'Edge-IIoTset', 'CICIoT2023', 'X-IIoTID']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table(headers, rows):
    def text(value):
        if isinstance(value, float):
            return f'{value:.8g}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(text, row)) + ' |' for row in rows])


def csv_file(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    DOC.mkdir(exist_ok=True)
    splits = {n: read(ART/'native_splits'/f'{n}.json') for n in NAMES}
    prep = {n: read(ART/'encoded_inputs'/n/'preprocessing.json') for n in NAMES}
    experiment = ART/'adapted_experiment'
    config = read(experiment/'configuration.json')
    baseline = read(experiment/'baseline.json')
    mixture = read(ART/'mixed_training/manifest.json')
    environment = read(DOC/'environment.json')
    verification = read(experiment/'verification.json')
    loader = read(experiment/'baseline_loader_verified.json')
    histories = {c['model']: read(experiment/c['model']/'history.json') for c in baseline['candidates']}
    selected_epochs = {k: max(v, key=lambda r:(r['mean_validation_macro_f1'],r['mean_validation_ap'])) for k,v in histories.items()}
    parameter_record = dict(
        recorded_date='2026-10-31', shared_configuration=config,
        adam_defaults=environment['adam_defaults'],
        actual_runs=[dict(**c,selected_checkpoint_epoch=selected_epochs[c['model']]['epoch'])
                     for c in baseline['candidates']],
        validation_rule=dict(frequency='After every complete epoch',
            primary='Unweighted mean of the four validation macro-F1 scores',
            checkpoint_comparison='Strict lexicographic improvement in (mean macro-F1, mean average precision)',
            minimum_improvement='No explicit min_delta; strict greater-than comparison',
            patience=3, baseline_tie_break='Mean validation AP, then fewer trainable parameters',
            threshold=0.5,test_data_used_for_selection=False),
        scheduler=None,gradient_clipping=None,class_weights=None,
        label_smoothing=None,explicit_data_augmentation=None,
        note='Recorded settings describe the completed adapted experiment; no new training performed')
    (DOC/'training_parameters.json').write_text(json.dumps(parameter_record,indent=2),encoding='utf-8')
    parameter_text = '''# Training parameters and validation rule

This reference records the completed five-model adapted mixed-domain experiment.
It summarizes the actual code and saved configuration; it is not a new run.

## Shared training settings

'''+table(['Parameter','Actual setting'],[
        ['Optimizer','Adam'],['Initial and fixed learning rate',config['learning_rate']],
        ['Adam beta1 / beta2',environment['adam_defaults']['betas']],
        ['Adam epsilon',environment['adam_defaults']['eps']],
        ['Weight decay',environment['adam_defaults']['weight_decay']],
        ['AMSGrad',environment['adam_defaults']['amsgrad']],
        ['Dropout probability',config['dropout']],
        ['Maximum training epochs',config['epochs']],
        ['Early-stopping patience',config['early_stopping_patience']],
        ['Effective batch size',config['batch_size']],
        ['Initial physical microbatch size',config['microbatch_size']],
        ['Loss','Unweighted BCEWithLogitsLoss'],
        ['Positive class','attack = 1; benign = 0'],
        ['Prediction threshold','Attack probability >= 0.5'],
        ['Python / NumPy / Torch seed',config['seed']],
        ['Training observations per completed epoch',mixture['rows']],
        ['Sampling','Natural domain proportions; shuffled domain-homogeneous batches'],
        ['Class weights / resampling','None'],
        ['Learning-rate scheduler','None; no learning-rate decay configured'],
        ['Gradient clipping','None configured'],
        ['Label smoothing / data augmentation','None configured'],
        ['Training device',config['gpu']],
        ['Precision',config['precision']],
        ['CPU threads',4],['cuDNN deterministic / benchmark','True / False']])+'''

All Adam defaults, including foreach/fused/capturable/differentiable flags, are
recorded in [training_parameters.json](artifacts/documentation/training_parameters.json).
Null foreach/fused values mean that no explicit choice was made; library behavior
follows the recorded PyTorch build. They do not prove a particular kernel was used.

Physical microbatches halve if the implemented GPU out-of-memory fallback is
triggered; the effective batch remains 8192 through weighted gradient accumulation.
Final partial batches contain the remaining rows. No rows are dropped to force
a full batch. A fallback is implemented, not claimed as an observed final event.

## Dropout and model dimensions

- Encoder: learned per-feature affine map to width 8, tanh, and fixed mean pooling
  into eight groups. No encoder dropout. Every candidate has four encoders and
  one shared classifier; the resulting representation is [batch, 8, 8].
- MLP/DNN: flatten to 64; hidden layers 128, 64, 32, ReLU and dropout 0.3 after
  each hidden layer; one output logit.
- CNN1D: channels 8->128->64->32, kernels 3, padding 1. Dropout 0.3 after the
  first two convolution/ReLU stages; adaptive average pooling. Head 32->64->32->1,
  with ReLU/dropout 0.3 after head widths 64 and 32.
- LSTM and GRU: input width 8, hidden width 128, two recurrent layers, dropout 0.3
  between recurrent layers. Head 128->64->32->1 with ReLU/dropout after 64 and 32.
- BiLSTM: same recurrent depth/width per direction; concatenate both final hidden
  states into 256 values. Head 256->64->32->1 with the same head dropout.

Dropout is disabled by model.eval() for validation/test inference. Encoder weights
and biases initialize from normal noise scaled by 0.02; other modules use their
installed PyTorch initialization defaults under seed 42. Shared settings do not
mean identical parameter counts or identical numbers of dropout layers.
Recurrent inputs are feature-group sequences, not chronological traffic.

## Exact validation and stopping rule

After every completed epoch, predict each of the four validation sets separately
using threshold 0.5. Compute binary macro-F1 and attack average precision (AP)
for each dataset. Let F_d and A_d be these values for the four datasets:

```
F = (F_DataSense + F_Edge + F_CIC + F_X) / 4
A = (A_DataSense + A_Edge + A_CIC + A_X) / 4
```

Each dataset has equal weight in validation selection, even though training uses
natural row proportions. Compare `(F, A)` lexicographically to the best recorded
pair. A higher F improves the checkpoint; exactly equal F and higher A also
improves it. No separate min_delta is configured. Improvement saves best.pt and
resets the non-improvement counter to zero. Otherwise increment the counter.
Stop after three consecutive non-improvements, or at the maximum of 20 epochs.
This is not early stopping on validation loss or on test accuracy.

After every candidate finishes, rank models by highest F, then highest A, then
fewest trainable parameters. Restore the best checkpoint, not automatically the
last epoch. Freeze the baseline choice before test evaluation. Test thresholds
are not optimized. The final baseline is provisional within this fixed-seed,
dataset-specific-encoder experiment.

## Actual completed runs

'''+table(['Model','Parameters','Epoch limit','Actual epochs','Best checkpoint epoch','Best mean validation macro-F1','Associated mean AP'],[
        [c['model'],c['parameters'],config['epochs'],c['epochs_run'],selected_epochs[c['model']]['epoch'],
         c['validation_macro_f1'],c['validation_ap']] for c in baseline['candidates']])+f'''

Selected baseline: **{baseline['selected']['model']}**, with mean validation
macro-F1 **{baseline['selected']['validation_macro_f1']:.8f}**. The MLP ran 20
epochs but its selected checkpoint is epoch {selected_epochs['MLP']['epoch']}.

## Input preprocessing and evaluation contract

Per-domain numeric parsing -> signed log1p -> fixed zero imputation -> per-domain
standard scaling fitted on training rows only. Feature counts are 64, 25, 39 and
25 for DataSense, Edge-IIoTset, CICIoT2023 and X-IIoTID, respectively. Labels,
timestamps and raw identities are not predictors; derived identity counts may
be present in the saved native feature lists. Domain identity routes the encoder.
This is not a common physical flow-feature representation.

All four test sets are evaluated separately for every candidate (20 evaluations).
Confusion counts, both-class precision/recall/F1/support, macro/weighted F1,
accuracy, balanced accuracy, MCC, specificity, false-positive/negative rates,
ROC-AUC, AP, log loss and Brier score are saved. AP is not trapezoidal PR area.
Inference batches use at most 8192 rows in the trainer. Model-only timing is a
separate experiment at batches 1/512 with five warmups and 20 repetitions.

## Sources and full procedure

- [Saved run configuration](artifacts/adapted_experiment/configuration.json)
- [Recorded optimizer defaults and machine environment](artifacts/documentation/environment.json)
- [Machine-readable parameter record](artifacts/documentation/training_parameters.json)
- [Training implementation](train_adapted.py) and [architectures](adapted_models.py)
- [Full process documentation](FULL_PROCESS_DOCUMENTATION.md)
- [Complete results](EXPERIMENT_REPORT.md)

The batch size, encoders and recurrent/CNN extensions differ from the paper.
No 2-D/3-D spatial variants, adversarial defenses or end-to-end edge timing are
claimed by these training results. See the full procedure for all limitations.
'''
    (PROJECT/'TRAINING_PARAMETERS.md').write_text(parameter_text,encoding='utf-8')
    test = []
    for candidate in baseline['candidates']:
        for name in NAMES:
            test.append(dict(model=candidate['model'],dataset=name,
                             **read(experiment/candidate['model']/'test'/name/'metrics.json')))
    with (ART/'report/inference_latency.csv').open(encoding='utf-8') as stream:
        timings = list(csv.DictReader(stream))
    features = []
    epoch_rows = []
    for name in NAMES:
        data = prep[name]
        for i, feature in enumerate(data['features']):
            features.append(dict(dataset=name,index=i,feature=feature,
                group=config['feature_group_assignment'][name][feature],
                train_missing=int(data['train_missing_per_feature'][i]),
                train_missing_fraction=data['train_missing_per_feature'][i]/splits[name]['counts']['train']['rows'],
                scaler_mean=data['scaler_mean'][i],scaler_scale=data['scaler_scale'][i]))
    for model, history in histories.items():
        for r in history:
            row = dict(model=model,epoch=r['epoch'],training_rows=r['training_rows'],
                       train_loss=r['train_loss'],mean_validation_macro_f1=r['mean_validation_macro_f1'],
                       mean_validation_ap=r['mean_validation_ap'],elapsed_seconds=r['elapsed_seconds'])
            for name in NAMES:
                row[f'{name}_validation_macro_f1'] = r['validation'][name]['macro_f1']
                row[f'{name}_validation_ap'] = r['validation'][name]['average_precision']
            epoch_rows.append(row)
    csv_file(DOC/'feature_catalog.csv',features)
    csv_file(DOC/'training_history.csv',epoch_rows)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PROJECT.glob('*.py'))}
    (DOC/'source_code_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    sections = []
    def add(title, content):
        sections.append(f'## {title}\n\n{content.strip()}\n')

    add('1. Scope, completion and evidence', '''
This records the current four-dataset experiment and its development, audited on
2026-10-31 using the user's Asia/Seoul client date. The project was rebuilt after
the earlier reset request; removed earlier experiments are not evidence for this run.

Completed: native-record binary labels, independent stratified splits, mixed
training membership, dataset-specific numerical preprocessing, five final neural
models, validation-only baseline selection, 20 separate model/test evaluations,
80 warm workstation inference timing configurations, saved predictions, plots,
checkpoints and verification. Original dataset files and papers were preserved.

This is an **authorized adapted mixed-domain classification study**, not a
completed real-time, explainable, adversarially robust edge IDS, and not a faithful
reproduction of the paper's common-flow experiment. No PCAP-to-flow extraction
was performed for these final results.

Evidence comes from the current source files and saved JSON/CSV/NPZ/NPY/PT
artifacts. Development events also use the conversation record. No complete
timestamped terminal transcript or versioned historical source tree was saved;
the chronology below is a reconstruction, not an invented command-by-command log.
''')
    add('2. Paper consulted and decisions authorized', '''
The local [TAN-IDS paper](../../journal.pone.0346801.pdf), particularly sections
4.1-4.5 and Tables 5-6, was consulted. Its mixed-domain scenario trains on
compatible NetFlow features at natural dataset proportions and tests each domain
separately. Its eight attributes are flow duration, total packets, total bytes,
source port, destination port, protocol, TCP flags and packet rate.

Reference: Dung Ha Thanh (2026), *A transfer-aware, deployment-oriented evaluation
framework for NetFlow-based intrusion detection systems (TAN-IDS)*, PLOS One
21(4): e0346801. [DOI](https://doi.org/10.1371/journal.pone.0346801).

The paper's source datasets are UNSW-NB15 and NF-CSE-CIC-IDS2018, not the four
datasets here. Its mapping cannot reconstruct flows from device-window summaries.
Deterministically encoding a connection state also does not establish equivalence
to a TCP packet-flag bitmask. The proposed exact four-dataset flow representation
was therefore not claimed to exist.

Decision sequence: inspect the paper and local schemas; explain the flow/window
mismatch; prepare independent native-record splits with pooling initially deferred
at the user's request; combine training membership when requested; then obtain
explicit authorization for dataset-specific encoders feeding a shared classifier.
The final training follows that last authorization. The eight encoder groups are
**not** the paper's eight flow attributes.

Retained principles: binary labels, 70/15/15 stratification, natural training
proportions, training-only preprocessing statistics, separate test sets and saved
artifacts. Adaptations: native observation units, per-domain features/scalers and
encoders, feature-group CNN/recurrent models, effective batch 8192 rather than
the paper's 512, mixed precision, early stopping and mean validation macro-F1
selection. CNN and recurrent candidates are additions to the paper's model set.
''')
    source_rows = []
    units = {'DataSense':'One-second device/network summaries',
             'Edge-IIoTset':'Selected packet/protocol CSV observations',
             'CICIoT2023':'Supplied traffic-statistic observations',
             'X-IIoTID':'Supplied connection records'}
    for n in NAMES:
        source_rows.append([n,units[n],len(splits[n]['sources']),prep[n]['rows'],len(prep[n]['features']),splits[n]['label_source']])
    add('3. Dataset inventory, release selection and labels', table(
        ['Dataset','Native observation','Selected CSV sources/members','Rows','Features','Binary label provenance'],source_rows)+'''

Labels are benign=0 and attack=1. String labels are stripped and lowercased;
unknown labels fail instead of being silently assigned a class.

- DataSense: select only matched `attack_samples_1sec.csv` and
  `benign_samples_1sec.csv` nested inside `all_attack_benign_samples.tar.xz`.
  `label1` supplies benign/attack. Other 1-10 second releases are not concatenated,
  avoiding repeated representations of the same traffic at different resolutions.
  The README explicitly describes time-window features. Raw captures were not
  available in the inspected local release.
- Edge-IIoTset: use only `Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv`.
  Map `Attack_label` 0/0.0 to benign and 1/1.0 to attack. The overlapping ML release
  and raw traffic CSVs are excluded from this training population. The 24 local
  PCAPs were found but were not processed into flows for this experiment.
- CICIoT2023: use all 309 supplied CSV files in scenario folders. No PCAP files
  were found; names ending `.pcap.csv` are CSV exports. The inspected schemas lack
  embedded labels and endpoint ports/flow identifiers. `Benign_Final` means
  benign; all other supplied scenario folders mean attack. This is scenario-level
  label provenance, not verified per-flow or per-packet attack attribution. The
  README's general embedded-label description does not match these inspected
  files, so the actual folder-based rule is recorded explicitly. Adding unrelated
  folders later is unsafe: the current code labels every non-Benign_Final folder
  as attack and would need review.
- X-IIoTID: use `X-IIoTID dataset.csv`; map `class3` Normal to benign and Attack to
  attack. `class1` and `class2` are excluded predictors. Network connection fields
  are retained; host resources, logs and alert outputs are not used.

No rows were deduplicated, resampled or removed in this stage. No synthetic or
adversarial rows were added. Native records are not relabelled as common flows.
The native split manifests list ordered file paths, archive members, offsets and
row counts; ordinary CSV sources also record sizes and modification times.
''')
    split_rows = []
    for n in NAMES:
        for s in ['train','validation','test']:
            c = splits[n]['counts'][s]
            split_rows.append([n,s,c['rows'],c['benign'],c['attack'],c['attack']/c['rows']])
    add('4. Exact independent splits', '''
For each dataset, `train_test_split` first reserves 30% using `stratify=y` and
`random_state=42`; a second stratified split divides the holdout equally into
validation and test using the same seed. Integer rounding explains one-row
differences. Every selected record receives exactly one split code: 0=train,
1=validation, 2=test. Splits preserve binary class proportions, not attack-subtype,
device, session or capture proportions.

'''+table(['Dataset','Split','Rows','Benign','Attack','Attack fraction'],split_rows)+'''

Original rows are not copied into multiple feature CSV exports. NPZ sidecars
store binary labels and split assignments in source order. JSON stores offsets
and SHA-256 digests of the label and split arrays. Array digests are not hashes
of all original data files, and source-row uniqueness does not prove that record
contents are duplicate-free.
''')
    mix_rows = [[n,mixture['datasets'][n]['dataset_id'],mixture['datasets'][n]['rows'],
                 mixture['datasets'][n]['benign'],mixture['datasets'][n]['attack'],
                 100*mixture['datasets'][n]['fraction']] for n in NAMES]
    add('5. Mixed training membership and actual sampling',table(
        ['Dataset','Domain ID','Training rows','Benign','Attack','Percent of mixture'],mix_rows)+f'''

The mixed index contains **{mixture['rows']:,}** training records:
{mixture['benign']:,} benign and {mixture['attack']:,} attack. No validation or
test rows are included. CICIoT2023 contributes about 93.47% of training rows.

`combine_training.py` concatenates structured entries with `dataset_id:uint8`,
`source_row:uint32` and `label:uint8`, then shuffles using NumPy RNG seed 42.
`source_row` is zero-based within the dataset's ordered source records, excluding
CSV headers. Use source offsets to locate a particular file/member and local row.
The source-backed index preserves provenance; it is not a shared feature matrix.

**The trainer does not stream the saved mixed index directly.** It reconstructs
the same training memberships from each domain's split arrays, shuffles each
domain's training indices with NumPy RNG seed `42 + epoch` (epoch is zero-based),
makes domain-homogeneous batches, and randomly interleaves those batches. Every
training observation appears once per completed epoch. Thus the saved global
index order is not the SGD order. Natural proportions refer to record inclusion;
batch-mean optimizer steps, including a few partial final batches per domain, are
not mathematically identical to globally IID mixed batches or exact uniform
per-record optimizer weighting. No class weights or domain balancing are applied.

![Training composition](artifacts/report/training_composition.png)
''')
    add('6. Numerical preprocessing and leakage controls', '''
`prepare_features.py` streams ordinary CSVs in chunks of 100,000 rows. DataSense
is read through nested streaming XZ tar archives and converted in blocks up to
50,000 rows; it is not unpacked into a second complete dataset. CICIoT2023 numeric
CSV parsing uses float32 directly for speed. Other native fields use string
conversion where required.

Processing order for every selected feature:

1. Convert Boolean text TRUE/FALSE to 1/0 where applicable; parse numeric hex
   strings such as TCP flag values; coerce other numeric text to float32.
2. Treat unparseable values and nonfinite numeric values as missing (malformed hex
   parsing can fail rather than being silently accepted).
3. Apply the fixed transform `z = sign(x) * log(1 + abs(x))`.
4. Replace missing transformed values by zero. This is fixed imputation, not a
   fitted mean/median policy. Missingness indicators are not added. Imputation
   can therefore conflate missing information with a true transformed zero.
5. For each domain independently, fit `StandardScaler.partial_fit` only on its
   training rows after fixed imputation. Fit means/scales over the complete
   training population, not a pilot sample.
6. Apply that same fitted transformation to all rows, including validation and
   test, without refitting; save standardized float32 arrays using memory maps.
   No clipping is implemented. Constant-feature scaling follows StandardScaler.

The standardized feature is `(z - training_mean) / training_scale`. Exact means,
scales, missing counts, feature names/order and split/label digests are saved in
each `preprocessing.json`. The appendix table is rounded for readability; JSON
and the feature catalog CSV retain Python's saved numeric precision.

Labels, scenario names, timestamps and raw identities are excluded as predictors.
DataSense derived distinct-IP/MAC/port/protocol counts **are retained**; exclusion
of raw identities does not eliminate all possible identity/scenario shortcuts.
X-IIoTID Protocol, Service and Conn_state categories are not encoded: they are
excluded from the explicit 25-field input. No cross-domain protocol dictionary,
byte-unit harmonization or eight-feature NetFlow conversion was performed.

During this documentation audit, the broad `excluded` descriptions in the four
preprocessing manifests were clarified to match the explicit feature lists. In
particular, an older DataSense description incorrectly suggested identity counts
were excluded. Only this descriptive text changed; feature arrays, feature
selection/order, labels, splits, means, scales and checkpoints were preserved.
See [description corrections](artifacts/documentation/metadata_description_corrections.json).
''')
    add('7. Feature selection, missingness and exact catalog',table(
        ['Dataset','Selected features','Training missing cells','Numeric feature matrix'],
        [[n,len(prep[n]['features']),sum(prep[n]['train_missing_per_feature']),
          f"{prep[n]['rows']:,} x {len(prep[n]['features'])}"] for n in NAMES])+'''

DataSense retains numeric `network_` fields while excluding raw identity/port/
protocol lists and all sensor/log fields. Edge and X-IIoTID use fixed explicit
column lists selected by source schema and field type, not a fitted feature
selection algorithm. CICIoT2023 uses the supplied 39 numeric fields. No SHAP,
permutation importance, statistical ranking or target-test-based feature selection
was performed. The grouping/selection is engineering judgment, not a validated
semantic equivalence test. Missing-cell totals are counts across selected training
features, not numbers of missing rows.

[Complete feature catalog](artifacts/documentation/feature_catalog.csv) contains
all 153 domain-specific fields, exact order, encoder group, training missing
count/fraction and recorded scaling mean/scale. Appendix A reproduces the list.
This is a column inventory, not a verified physical-unit dictionary. Per-feature
units, traffic direction and extraction semantics were not comprehensively
validated or reconciled across the four native exporters.
''')
    add('8. Dataset-specific encoders and tensor meaning', '''
Each candidate has four separate trainable encoders and one shared classifier.
Domain identity is supplied to route a row to the correct encoder; it is not
inferred from traffic. This is known-domain mixed training, not zero-shot transfer
to a new dataset. An unknown schema has no trained encoder.

For feature j in domain d, create an 8-dimensional token:
`t[d,j] = tanh(x[d,j] * W[d,j] + b[d,j])`, with learned 8-vectors W and b.
W and b initialize from normal noise scaled by 0.02. Mean-pool tokens within each
saved group to obtain shape `[batch, 8 groups, 8 embedding components]`. Empty
groups produce zero latent tokens; those zeros are not observed network counts.
Pooling masks are registered model buffers and saved with checkpoints. The
encoders and shared classifier are optimized jointly; each domain batch updates
its encoder and the shared classifier.

Group order: ports, packet_counts, size_statistics, timing_rates, flags,
protocol_attributes, headers_topology, other. `group_for()` uses deterministic
name heuristics with priority rules. Group names organize native attributes,
not common physical definitions. One saved example is `tcp.connection.synack`
in `other`, because the final boundary-matching rule does not recognize the
combined SYNACK token. The recorded assignment is reported, not silently changed
after training. These heuristics have not undergone a comprehensive semantic
ontology audit.

MLP flattens the 8 x 8 representation to 64 values. CNN1D treats the eight groups
as its feature axis and the eight embedding components as channels. LSTM/GRU/
BiLSTM process the fixed feature-group order, not chronological observations.
This does not test temporal traffic learning. No adjacent shuffled rows are
fabricated into a time series. No 2-D/3-D models were run because meaningful
spatial axes were unavailable. Repacking a vector alone would not establish them.

```mermaid
flowchart LR
    D[Four native datasets] --> S[Independent 70/15/15 splits]
    S --> T[Training memberships only]
    T --> P[Per-domain training-fitted preprocessing]
    P --> E[Four dataset-specific encoders]
    E --> G[Eight group tokens]
    G --> C[One shared classifier for each candidate]
    S --> V[Four validation sets]
    C --> R[Validation ranking and frozen checkpoints]
    V --> R
    S --> Q[Four separate test sets]
    R --> F[Frozen models]
    F --> M[Per-dataset predictions and metrics]
    Q --> M
```
''')
    add('9. Exact model architectures', '''
All candidates output one attack logit and use sigmoid for probabilities. Dropout
is disabled for evaluation. MLP and DNN name the same candidate, not two runs.

| Candidate | Shared backbone after the domain encoder | Shared output head |
|---|---|---|
| MLP | Flatten 8x8; Linear 64->128; ReLU; Dropout 0.3 | Linear 128->64, ReLU, Dropout; Linear 64->32, ReLU, Dropout; Linear 32->1 |
| CNN1D | Conv1d 8->128, k=3,pad=1,ReLU,Dropout; Conv1d 128->64,k=3,pad=1,ReLU,Dropout; Conv1d 64->32,k=3,pad=1,ReLU; adaptive average pool to 1; flatten | Linear 32->64, ReLU, Dropout; Linear 64->32, ReLU, Dropout; Linear 32->1 |
| LSTM | Input width 8, hidden width 128, two layers, batch_first, recurrent dropout 0.3; final top-layer hidden state | Linear 128->64, ReLU, Dropout; Linear 64->32, ReLU, Dropout; Linear 32->1 |
| GRU | Same dimensions/layers as LSTM, using GRU cells | Same 128->64->32->1 head |
| BiLSTM | Two-layer bidirectional LSTM, hidden width 128 per direction; concatenate final forward/backward hidden states (256 values) | Linear 256->64, ReLU, Dropout; Linear 64->32, ReLU, Dropout; Linear 32->1 |

Head dropout is 0.3. ReLU and dropout are applied after the intermediate head
layers as shown. Recurrent cell nonlinearities follow the PyTorch cell definition.
Model-specific cell/conv parameters and parameter counts differ; 'same parameters'
means common comparable training settings, not mathematically identical models.
''')
    defaults = environment['adam_defaults']
    add('10. Shared training settings and execution',table(
        ['Setting','Value'],[['Seed',42],['Optimizer','Adam'],['Learning rate',0.001],
        ['Adam betas',defaults['betas']],['Adam epsilon',defaults['eps']],
        ['Weight decay',defaults['weight_decay']],['Dropout',0.3],
        ['Loss','Unweighted BCEWithLogitsLoss'],['Effective batch',8192],
        ['Initial physical microbatch',8192],['Maximum epochs',20],
        ['Early-stopping patience',3],['Probability threshold',0.5],
        ['CUDA precision','float16 autocast; GradScaler; float32 master weights'],
        ['Torch CPU threads',4],['cuDNN deterministic',True],['cuDNN benchmark',False]])+'''

Python, NumPy and Torch RNGs are seeded for each candidate; CUDA RNGs are seeded
where available. No learning-rate scheduler, explicit gradient clipping, class
weighting, focal loss or data augmentation is implemented. Other Adam options
use the captured installed defaults in the environment JSON. Reproducibility
across different devices/library versions is not claimed to be bit-for-bit.

For a batch, clear gradients, forward the native rows through their domain encoder
and shared model, compute BCE, scale gradients, backpropagate and step Adam. When
microbatches are needed, each micro-loss is weighted by its share of the batch.
If CUDA runs out of memory, physical microbatch size halves and the batch is
retried with cleared gradients; the effective batch remains unchanged. This
fallback exists in code; it is not claimed to have been needed in the final run.
Nonfinite batch loss raises an error. Progress is periodically persisted.

Every epoch ends with predictions on each validation set. History records training
BCE, per-domain validation macro-F1/AP, mean scores, rows processed and an epoch
timer. Timer capture occurs before final history/checkpoint/plot writes and does
not include environment installation, initial feature preparation, development
trials, model initialization or full end-to-end project wall time.
''')
    model_rows = []
    for c in baseline['candidates']:
        k = c['model']
        model_rows.append([k,c['parameters'],c['epochs_run'],selected_epochs[k]['epoch'],
                           sum(h['elapsed_seconds'] for h in histories[k]),
                           c['validation_macro_f1'],c['validation_ap']])
    add('11. Completed final runs and baseline selection',table(
        ['Model','Trainable parameters','Epochs run','Selected epoch','Sum epoch timers (s)',
         'Best mean validation macro-F1','Associated mean validation AP'],model_rows)+f'''

Total final recorded epochs: **{len(epoch_rows)}**. Total native training-record
presentations over those epochs: **{sum(r['training_rows'] for r in epoch_rows):,}**.
This is repeated exposure to the same training population, not additional unique
data. Early-stopped models retain the best checkpoint, not their last checkpoint.

After each epoch, compute the unweighted mean of the four validation macro-F1
values. Improve a candidate checkpoint when `(mean macro-F1, mean AP)` improves
lexicographically. Three consecutive non-improvements trigger early stopping.
Choose the final candidate by mean validation macro-F1, then mean validation AP,
then fewer trainable parameters. Threshold remains 0.5 and is not optimized on
test data. No test metric determines the baseline.

Selected baseline: **{baseline['selected']['model']}**, checkpoint epoch
**{selected_epochs[baseline['selected']['model']]['epoch']}**, validation mean
macro-F1 **{baseline['selected']['validation_macro_f1']:.8f}**. It is the best
of the five final grouped-encoder candidates for this seed and rule, not a claim
of universal superiority or the best result among every discarded pilot.
The small lead over CNN1D has not been tested for statistical significance.

[Baseline selection record](artifacts/adapted_experiment/baseline.json).
[All recorded epochs](artifacts/documentation/training_history.csv).
Appendix B lists every final epoch's aggregate scores and runtime.
''')
    valrows=[]
    for k,r in selected_epochs.items():
        for n in NAMES:
            valrows.append([k,r['epoch'],n,r['validation'][n]['macro_f1'],r['validation'][n]['average_precision']])
    add('12. Validation results at selected checkpoints',table(
        ['Model','Checkpoint epoch','Validation dataset','Macro-F1','Average precision'],valrows))
    add('13. Held-out evaluation protocol and saved predictions', '''
The baseline decision was written before any final model's test results were
computed. Each candidate's best checkpoint is restored and evaluated on all four
test sets separately, producing 5 x 4 = 20 evaluations. Rows are processed in
their saved native test order, in inference batches up to 8192. GPU inference
uses autocast; logits are converted to float32 before sigmoid.

For each model/domain, save `predictions.npz` with `labels`, `probability` and
`predicted`; `source_rows.npy` identifies original native test rows; `metrics.json`
stores scalar results; `confusion_matrix.png` and `roc_pr.png` provide plots.
The report supplements benign-class metrics and renders confusion counts plus
row-normalized matrices from the saved counts. ROC has a diagonal reference;
PR has a test-prevalence reference. Learning curves show training loss and mean
validation macro-F1. A separate all-model plot compares test macro-F1 by domain.

No adversarial samples or test-time adaptation are involved. This is within-known-
domain held-out testing after training on all four domains, not source-only
cross-domain transfer. Tests remain separate even though validation ranking
averages the four domains.
''')
    add('14. Metric definitions and interpretation', '''
Positive class is attack=1. Confusion rows are actual [benign,attack], columns are
predicted [benign,attack], so the matrix is `[[TN,FP],[FN,TP]]`.

| Metric | Definition/implementation |
|---|---|
| Accuracy | (TP+TN)/N |
| Attack precision | TP/(TP+FP) |
| Attack recall / detection rate | TP/(TP+FN) |
| Attack F1 | 2TP/(2TP+FP+FN) |
| Benign precision | TN/(TN+FN) |
| Benign recall / specificity | TN/(TN+FP) |
| Benign F1 | 2TN/(2TN+FP+FN) |
| Macro-F1 | Mean of benign F1 and attack F1 |
| Weighted F1 | Class F1 weighted by actual class supports |
| Balanced accuracy | Mean of the two class recalls |
| False-positive rate | FP/(TN+FP), the benign false-alarm fraction |
| False-negative rate | FN/(TP+FN), the attack miss fraction |
| MCC | sklearn Matthews correlation coefficient |
| ROC-AUC | sklearn roc_auc_score on attack probabilities |
| Average precision (AP) | sklearn average_precision_score; not trapezoidal PR area |
| Log loss | sklearn binary log_loss; probability clipping follows installed sklearn precision |
| Brier score | Mean squared difference between probability and binary label |
| Support | Number of actual records of each class |

Precision/recall/F1 use zero_division=0. Undefined single-class ROC-AUC/AP are
recorded as null; both classes occur in the actual selected test sets. Rate
denominators absent in generic evaluation return null where implemented.
High accuracy/AP on a heavily attack-dominated domain does not describe its
benign false-alarm performance. Read the per-class rates and supports as well.
There are no confidence intervals, bootstrap tests, multi-seed averages, calibrated
probability claims or significance tests in this run.
''')
    add('15. All final confusion counts and supports',table(
        ['Model','Test dataset','N','TN','FP','FN','TP','Benign support','Attack support'],
        [[r[k] for k in ['model','dataset','n','tn','fp','fn','tp','support_benign','support_attack']] for r in test]))
    add('16. All final class-specific precision, recall and F1',table(
        ['Model','Dataset','Benign precision','Benign recall','Benign F1','Attack precision','Attack recall','Attack F1','Macro-F1','Weighted F1'],
        [[r[k] for k in ['model','dataset','precision_benign','recall_benign','f1_benign','precision_attack','recall_attack','f1_attack','f1_macro','f1_weighted']] for r in test]))
    add('17. All final summary, ranking and probability metrics',table(
        ['Model','Dataset','Accuracy','Balanced accuracy','MCC','Specificity','FPR','FNR','ROC-AUC','AP','Log loss','Brier'],
        [[r[k] for k in ['model','dataset','accuracy','balanced_accuracy','mcc','specificity','false_positive_rate','false_negative_rate','roc_auc','average_precision','log_loss','brier_score']] for r in test])+'''

[Machine-readable full metric table](artifacts/report/test_metrics.csv).

![All final candidates](artifacts/report/test_comparison.png)
''')
    chosen=baseline['selected']['model']
    chosen_rows={r['dataset']:r for r in test if r['model']==chosen}
    add('18. Baseline errors and figures',f'''
The selected {chosen} misses **{100*chosen_rows['DataSense']['false_negative_rate']:.2f}%**
of DataSense attacks and falsely flags **{100*chosen_rows['CICIoT2023']['false_positive_rate']:.2f}%**
of CICIoT2023 benign test observations. These are substantial remaining errors,
despite high aggregate accuracy in some domains. The label provenance and split
limitations below must also constrain interpretation.

The frozen threshold is not retuned after observing these errors. This is a
provisional experimental baseline, not a certified deployment-ready IDS.

'''+ '\n\n'.join(f"### {n}\n\n![{chosen} {n} confusion](artifacts/adapted_experiment/{chosen}/test/{n}/confusion_matrix.png)\n\n![{chosen} {n} ROC and PR](artifacts/adapted_experiment/{chosen}/test/{n}/roc_pr.png)" for n in NAMES))
    reference=[]
    for n in NAMES:
        r=read(experiment/'domain_prior_reference'/n/'metrics.json')
        c=splits[n]['counts']['train']
        reference.append([n,c['attack']/c['rows'],r['accuracy'],r['f1_macro'],r['roc_auc'],r['average_precision']])
    add('19. Domain-prior reference', '''
For each domain, predict its training attack prevalence for every test row,
without inspecting any traffic features. Use the same threshold 0.5. This
reference uses domain identity and training labels only. It is not a neural
candidate or part of baseline selection. It makes the benefit of class imbalance
and known-domain routing visible.

'''+table(['Dataset','Training attack prior','Test accuracy','Macro-F1','ROC-AUC','AP'],reference)+'''

[Reference metrics](artifacts/report/domain_prior_metrics.csv). No paired
significance analysis against the neural candidates was performed.
''')
    add('20. Workstation inference timing: all measurements', '''
Use the first 512 training records of each domain as timing inputs, not additional
test-label tuning. Restore each model's best checkpoint. Measure CPU (four Torch
threads, float32) and GPU (autocast) at batch sizes 1 and 512. Inputs are already
preprocessed and resident on the measured device before timing. Each case uses
five warmups and 20 timed repetitions. CUDA is synchronized before and after each
timed forward pass. Time the model plus sigmoid, then report median, p95 and
batch_size/median_seconds throughput. Twenty repetitions give an empirical p95,
not a tail-latency guarantee. Raw repetition-level durations were not saved, so
the reported medians/p95 cannot be independently reconstructed from the CSV alone.

This is warm, cached-input **workstation model inference**, not edge-device or
end-to-end latency. Excluded: PCAP/packet extraction, CSV parsing, feature/window
construction, imputation/scaling, host-to-device transfer, queueing, SHAP/IG and
window collection. DataSense's one-second observation window is not included
in these millisecond model timings. No live deployment or throughput stress test
was performed. CPU/GPU precision differs as disclosed; timings are not a pure
hardware-only comparison.

'''+table(['Model','Dataset','Device','Batch','Repetitions','Median ms','p95 ms','Observations/s'],
        [[r[k] for k in ['model','dataset','device','batch_size','repetitions','median_ms','p95_ms','observations_per_second']] for r in timings])+'''

[Timing CSV](artifacts/report/inference_latency.csv).
[Exact timing scope](artifacts/report/inference_latency_scope.json).
''')
    add('21. Verification actually performed', '''
Checks performed during development/finalization:

- Split generation checked binary labels, disjoint assignment, total coverage and
  per-class proportions within integer-rounding tolerance.
- Saved split labels/counts/membership and recorded SHA-256 digests were checked.
- Mixed-index construction checked exact per-domain training membership,
  uniqueness of source-row references, labels and exclusion of held-out rows.
- Python syntax checks were run. Synthetic known predictions checked confusion
  counts (1,1,1,1), accuracy 0.5, ROC-AUC 0.75 and plot generation.
- Synthetic forward/backward checks exercised all five grouped architectures and
  all four actual input widths with CUDA mixed precision. Targeted grouping
  assertions checked packet counts, size deviations, timing, flags, TTL, protocol
  attributes and ports after correcting substring ambiguity.
- Final `verify_results.py` checked every model/domain's exact original test rows,
  labels, valid probability range, threshold predictions, confusion counts,
  accuracy and required plots; all epoch row totals and baseline ranking were
  checked. This is an artifact audit, not independent re-inference of every test
  record, a mathematical proof of uniqueness inside the training loop, or a
  test of duplicate-free record contents.
- A loader test re-inferred 32 saved test records per domain; maximum probability
  difference was 0.0 for all four sets. It validates 128 sampled probabilities,
  not independent recomputation of all predictions.
- Final CSV checks confirmed 20 model/test evaluations, per-class supports,
  macro-F1 consistency and 80 positive-latency timing configurations.
- The completed experiment was rerun idempotently: completed candidates and their
  test inference were reused, checkpoints were restored, and the completion run
  returned exit code 0. Domain-prior outputs/report were refreshed by that rerun.
- The notebook was executed with nbclient and validated with nbformat; no cell
  errors remained. Comparison and confusion figures were visually inspected;
  the comparison legend was moved outside the bars, and confusion annotation
  contrast plus row normalization were added for readability.

'''+f"Saved final audit status: **{verification['status']}** for {len(verification['evaluations'])} evaluations.\n\n"+table(
        ['Loader dataset','Sampled rows','Maximum probability difference'],
        [[n,loader[n]['checked_rows'],loader[n]['max_probability_difference']] for n in NAMES])+'''

[Artifact audit](artifacts/adapted_experiment/verification.json).
[Loader audit](artifacts/adapted_experiment/baseline_loader_verified.json).
[Architecture preflight](artifacts/model_preflight.json).
No full original-file cryptographic audit, independently implemented ROC/AP
recalculation, multi-seed retraining or live edge validation was performed.
''')
    pilot_rows=[]
    for p in sorted((ART/'pilot_feature_tokens').rglob('history.json')):
        h=read(p)
        pilot_rows.append([str(p.parent.relative_to(PROJECT)).replace('\\','/'),len(h),
                           max(x['mean_validation_macro_f1'] for x in h),'Excluded'])
    add('22. Development chronology, failures and corrections', '''
This is a factual, coarse sequence reconstructed from the work session; exact
timestamps for every command were not retained.

1. Read the paper's mapping/mixed-domain sections and dataset READMEs; inspect
   local file types, sample schemas and label columns. Python's Windows Store
   alias failed under the default sandbox, so approved escalated local interpreter
   calls were used. PDF text needed UTF-8 stdout to avoid Windows encoding errors.
2. Explain that DataSense windows and CIC CSV statistics do not supply compatible
   individual flows. Preserve the dataset files; do not invent packet timestamps,
   ports or flow definitions. Prepare the requested native splits first.
3. Save mixed training membership without pretending it contains common feature
   values. Later, proceed with the user's explicit separate-encoder authorization.
4. Discover the global Torch installation could not import due to a missing DLL;
   matplotlib was also absent. Create workspace-local `.venv`, install numerical/
   plotting libraries, then replace its CPU Torch build with the official CUDA
   build from the cu128 index. Add ipykernel, nbclient and nbformat for execution
   and validation of the notebook. No raw dataset download was performed.
5. Stream numeric features. Initial string processing of all CIC numeric columns
   was slow. Stop only the owned feature job, optimize CIC float32 parsing and
   restart while reusing completed DataSense/Edge arrays. An attempted restart
   while an old memory map was still open failed; the specific worker was then
   identified and stopped before restarting successfully.
6. Prepare initial generic model definitions/configuration, then remove these
   unused files when the authorized adapted implementation became authoritative.
   The final architectures reside in `adapted_models.py`.
7. Start a full-feature-token prototype. Small physical microbatches caused high
   overhead. Enable autocast/GradScaler and larger microbatches while retaining
   effective batch 8192 and full record coverage. During restart handling, the
   Python launcher/worker process pair prevented an initial single-process match;
   transient duplicate development jobs were identified and stopped. Retained
   final results do not come from those concurrent jobs.
8. Preserve the full-feature-token pilot and switch to eight grouped tokens to
   make the shared CNN/recurrent comparison practical. This was a documented
   architecture change, not an unchanged-parameter replay of the pilot.
9. Detect ambiguous grouping substrings: `ack` can occur inside `packet`, and
   `iat` can occur inside `deviation`. Use boundary-aware flag matching and exact
   IAT matching, check representative expected groups, preserve the grouping
   draft separately and restart the final comparison with fresh checkpoints.
10. A Windows permission error interrupted atomic replacement of the progress
    JSON. Add bounded retries; progress-only write failure must not abort training.
    A checkpoint-resume attempt exposed CUDA RNG tensors mapped to GPU; convert
    them to CPU ByteTensors before restoring generator state. The corrected final
    comparison began fresh, so a full interrupted-epoch resume test is not claimed.
11. Run the five final grouped candidates to their epoch limits/early stopping.
    Save baseline selection before final test metrics. Evaluate all four held-out
    sets for every final candidate and save predictions/figures/metrics.
12. Some long-running/synthetic tool sessions reported nonzero exits after printing
    successful output; the captured output did not establish the cause. The result
    banner was not treated as sufficient evidence: artifact/loader audits passed,
    and the completed pipeline's idempotent rerun returned 0. This does not explain
    the earlier exit-status anomaly; it documents the observed resolution checks.
13. Benchmark warm inference, check outputs and execute the notebook. Initial
    notebook runs emitted Windows event-loop and local-kernel transport warnings;
    no cell errors occurred. Plot layout, per-class metric coverage and stale
    README/status descriptions were corrected.
14. In the present documentation pass, audit code/results, clarify exclusion
    descriptions, export exact feature/epoch catalogs, inventory artifacts and
    write this consolidated record. No training, threshold tuning or new test
    predictions are performed by the documentation generator.

Retained development histories:

'''+table(['Pilot directory','Recorded epochs','Best mean validation macro-F1','Final selection status'],pilot_rows)+'''

The original full-feature-token MLP pilot achieved validation macro-F1 0.944780,
slightly above the final grouped MLP's score. It was excluded to keep the final
five-model comparison on the corrected shared encoder design. Therefore the
chosen baseline is not described as the best score ever observed during
development. Pilot validation results had been viewed during architecture
development; validation is not an untouched estimate of model-selection error.
Final test scores were not used for those design decisions.

Pilots have retained configurations/checkpoints/history where saved, but complete
historical source versions and all interrupted attempts were not archived. Some
pilot checkpoints are incompatible with the current grouped model constructor.
They are provenance artifacts, not supported current-loader models, and do not
have final test evaluations in this comparison.
''')
    add('23. Environment, tools and files created',table(
        ['Item','Recorded value'],[['OS',environment['platform']],['Python',environment['python']],
        ['GPU',environment['gpu']],['Torch',environment['torch']],['Torch CUDA build',environment['cuda_build']],
        ['CPU',environment['cpu']],['Logical CPUs',environment['logical_cpus']],
        ['Training package versions',json.dumps(environment['training_package_versions'])],
        ['Original preprocessing versions',json.dumps(environment['preprocessing_package_versions'])]])+'''

The GPU was inspected with NVIDIA-SMI as an RTX 4060 with about 8 GiB total VRAM;
the WMI AdapterRAM value was not used as a reliable VRAM limit. Epoch training
used CUDA and four Torch CPU threads; timing separately exercised CPU and GPU.
This is a workstation, not an evaluated edge board. Runtime RAM capacity and a
complete historical machine-utilization trace were not captured in run artifacts.

Tools used included PowerShell filesystem/process inspection, Python CSV/tar/PDF
reading, pypdf for the supplied paper/READMEs, NumPy/Pandas/sklearn for data work,
Torch/CUDA for training, matplotlib for plots, and nbclient/nbformat for notebook
execution. The official PyTorch installation page was checked for CUDA setup;
no traffic records were uploaded to an external service and no external messaging
was part of this work.

Created workspace content includes project Python modules and Markdown docs,
the preview notebook, `.venv` and dependency lock files, native split sidecars,
mixed index, per-domain feature arrays/scalers, final checkpoints and optimizer
states, pilot artifacts, prediction arrays, scalar metrics, plot files, audit
records and documentation exports. Python/Jupyter also create ordinary caches.
Original datasets and reference documents are external inputs to this project.

[Environment record](artifacts/documentation/environment.json).
[Source code hashes](artifacts/documentation/source_code_sha256.json).
[Artifact inventory](artifacts/documentation/artifact_inventory.csv) lists retained
non-documentation artifacts by relative path, bytes and role; it excludes `.venv`,
Python caches and the documentation directory itself. This is not a complete
inventory of pip's external cache or temporary operating-system files.
''')
    add('24. Module map and data contracts',table(['File','Responsibility'],[
        ['prepare_splits.py','Read native sources/labels and save seed-42 split membership and provenance'],
        ['combine_training.py','Build/verify source-backed mixed training index'],
        ['prepare_features.py','Stream native numeric inputs and fit per-domain training-only scalers'],
        ['adapted_models.py','Feature-group encoder and five shared classifier backbones'],
        ['train_adapted.py','Full-data training, validation ranking, checkpoints, test inference, domain-prior reference'],
        ['evaluate.py','Binary metrics, probability/prediction persistence and initial confusion/ROC/PR plots'],
        ['report.py','Aggregate tables, normalized confusion plots, comparison figures and notebook skeleton'],
        ['baseline.py','Restore frozen baseline and predict from preprocessed known-domain inputs'],
        ['verify_results.py','Audit final predictions against native test membership and ranking'],
        ['benchmark_inference.py','Warm CPU/GPU model-only timing'],
        ['document_process.py','Build this consolidated documentation and exact catalogs without training']])+'''

| Artifact path | Contract |
|---|---|
| native_splits/<dataset>.npz | `labels:uint8` and `split:uint8`, native source order |
| native_splits/<dataset>.json | Ordered source/member offsets, counts, label/split hashes and label provenance |
| mixed_training/records.npy | Structured dataset_id/source_row/label rows; source-backed, shuffled index |
| mixed_training/manifest.json | Mixed counts, ratios, source manifest hashes and index hash |
| encoded_inputs/<dataset>/features.npy | Standardized float32 [native observations, selected features] |
| encoded_inputs/<dataset>/labels.npy, split.npy | Aligned native labels and split codes |
| encoded_inputs/<dataset>/preprocessing.json | Feature order, fitted means/scales, missing counts and provenance |
| adapted_experiment/configuration.json | Final model/training settings and every feature-group assignment |
| adapted_experiment/<model>/best.pt | Selected state_dict, model kind, input feature counts and best epoch |
| adapted_experiment/<model>/last.pt | Model/optimizer/GradScaler/RNG/history state for epoch-boundary resumption |
| adapted_experiment/<model>/history.json | Per-epoch train loss, validation scores, row counts and timer |
| adapted_experiment/<model>/selection.json | Completed-candidate selection scores, parameter count and epochs run |
| adapted_experiment/baseline.json | Frozen validation-only selection and candidate ranking values |
| adapted_experiment/<model>/test/<domain>/ | metrics.json, predictions.npz, source_rows.npy and plots |
| adapted_experiment/domain_prior_reference/ | Per-domain no-feature reference predictions/metrics/plots |
| adapted_experiment/verification.json | Final held-out membership/count/ranking audit |
| report/ | All-model metric CSV, timing CSV/scope and composition/comparison plots |
| pilot_feature_tokens/ | Excluded development artifacts; not final candidate evidence |

The domain order is DataSense, Edge-IIoTset, CICIoT2023, X-IIoTID (IDs 0-3).
Feature order is mandatory. A domain label routes preprocessing/encoder selection;
no automatic adaptation to an unknown schema is implemented. Raw flows cannot be
passed to the baseline helper as though they were preprocessed native features.
''')
    add('25. Reproduction and operational cautions', '''
Run from the workspace root. The original split/feature preparation used the
global Python environment described by `requirements.txt`; neural training and
notebook execution used `.venv` with `requirements-training-lock.txt`. For a fresh
environment, create isolated environments rather than replacing the existing
working environment. The training lock includes the official cu128 extra index.
The original paper/PDF reading also used global pypdf, which is not required by
the training pipeline or this documentation generator.

```powershell
# Example fresh preprocessing environment; these are instructions, not new runs performed here.
python -m venv Coding/edge_ids/.venv-preprocess
Coding/edge_ids/.venv-preprocess/Scripts/python.exe -m pip install -r Coding/edge_ids/requirements.txt
# For a new training environment at a separately chosen path:
# python -m venv <new-training-environment>
# <new-training-environment>/Scripts/python.exe -m pip install -r Coding/edge_ids/requirements-training-lock.txt

# Ordered pipeline (fresh generation is only fresh when output artifacts are absent).
Coding/edge_ids/.venv-preprocess/Scripts/python.exe Coding/edge_ids/prepare_splits.py
Coding/edge_ids/.venv-preprocess/Scripts/python.exe Coding/edge_ids/combine_training.py
Coding/edge_ids/.venv-preprocess/Scripts/python.exe Coding/edge_ids/prepare_features.py
Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/train_adapted.py
Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/verify_results.py
Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/benchmark_inference.py
Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/report.py
Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/document_process.py
```

To review existing results, open `preview.ipynb`, `EXPERIMENT_REPORT.md` and this
document. The verifier reads existing predictions; it does not retrain. The
benchmark does perform new timing repetitions if run again. `report.py` refreshes
metric summaries/figures and rewrites the notebook skeleton, clearing its executed
outputs; execute the notebook again afterwards if rendered outputs are needed.
This documentation generator preserves existing notebook code/output and adds
only a documentation link cell if absent.

Current cache/resume behavior matters:

- Feature preparation reuses a domain when its preprocessing JSON has matching
  split hash and row count. It does **not** fingerprint every source file, feature
  rule or scaler configuration. Changed code/data with the same split can leave a
  stale cache. Inspect provenance and use a separate fresh project/output area
  for a changed experiment; do not assume rerunning invalidates all caches.
- A candidate `selection.json` causes training reuse based on existence. `last.pt`
  supports epoch-boundary restoration of model, optimizer, scaler and RNG state;
  partial-epoch batches are not resumable. There is no full configuration-signature
  validation before reuse. Changing arguments over an old output directory can
  mix configurations, and configuration.json itself is rewritten on rerun.
- Existing test outputs are reused when metrics, ROC/PR plot and source_rows
  exist; this cache check does not test every prediction file or checkpoint hash.
  Run the independent verifier to audit the complete final artifact contract.
- The final outputs record one seed and one configured experiment. There is no
  multi-run experiment registry or complete historical code versioning. Source
  hashes generated now describe the current code, not the exact pre-fix pilot
  source. Preserve the complete current folder for review and start changes in a
  distinct fresh directory at the same workspace depth, with the original Datasets
  folder available. No existing datasets need deletion.
- `best.pt` uses weights-only loading in inference helpers; `last.pt` restores
  richer trusted local state. These are this project's local artifacts.

Baseline loading example from a notebook whose working directory is `Coding/edge_ids`:

```python
import numpy as np
from baseline import load_baseline, attack_probability
model, contract = load_baseline()
domain = 'DataSense'
source = np.load('artifacts/encoded_inputs/DataSense/features.npy', mmap_mode='r')
# Rows here are already transformed in the saved native feature order.
p_attack = attack_probability(model, np.asarray(source[:8]).copy(), domain)
prediction = (p_attack >= 0.5).astype('uint8')
```

The example is inference usage, not an additional evaluation protocol. It does not
parse packets, create flow features or refit scaling. For raw native CSV values,
apply precisely the saved domain's numeric transform, zero-imputation and
training mean/scale before calling this helper.
''')
    add('26. Limitations, unimplemented stages and defensible conclusions', '''
What this run establishes: all five final candidates trained on the recorded
mixed native memberships and were evaluated against the recorded four held-out
native test splits. Under the fixed seed, settings and validation criterion,
MLP was selected, and the artifact checks passed.

What it does not establish:

| Area | Actual status |
|---|---|
| Common physical flow representation | Not built; window/packet/connection units remain different |
| Semantic equality of byte, timing, protocol and flag definitions | Not proven by encoder grouping |
| Unseen-domain zero-shot transfer | Not evaluated; all four domains supply training data and own encoders |
| Chronological LSTM/GRU/BiLSTM modelling | Not evaluated; recurrence is over feature-group tokens |
| 2-D/3-D CNN comparison | Not run; no justified spatial axes |
| Duplicate/capture/session/device/time isolation | Not enforced by the requested random row splits |
| Attack-subtype generalization | Not established by binary class stratification |
| Reliable per-flow labels for CIC scenarios | Not established; folder labels are scenario-level provenance |
| Missing-value semantics | Fixed zero imputation can conflate missing and real zero; no missingness indicators |
| Statistical significance and stability | No multi-seed study, bootstrap CI, nested CV or retraining-stability tests |
| Threshold optimization and calibration | Threshold frozen at 0.5; no calibration training or guarantees |
| Adversarial attacks/defenses | FGSM, PGD, DeepFool, C&W and other attacks/defended models not run here |
| Explainability | SHAP, permutation importance, IG, counterfactuals, faithful CANDICE and fidelity/stability tests not implemented here |
| Other model families | Autoencoders, GNNs, transformers, classical/GAN models not part of this completed five-model stage |
| Edge deployment/real-time pipeline | No live extraction, streaming replay, edge-board benchmark or end-to-end timing |
| Scientific semantic feature audit | Current grouping rules are engineering heuristics, not a complete verified ontology |

Raw identities/labels were excluded, but ports, derived identity counts and
domain-specific distributions can still provide shortcuts. Random splits can
place duplicate or closely related observations across train and test. Known-
domain encoders reveal domain membership. High reported scores cannot therefore
be generalized to independent traffic, physical flows, unseen deployments or
adversarial robustness. The observed DataSense miss rate and CIC false-positive
rate are concrete remaining weaknesses, not merely theoretical caveats.

Future work would require newly defined, versioned experiments and appropriate
data/labels before those claims could be tested. The final test sets should not
be repeatedly used to tune the next model without treating them as development
data and establishing a fresh held-out evaluation.
''')
    appendix=[]
    for n in NAMES:
        subset=[r for r in features if r['dataset']==n]
        appendix.append('### '+n+'\n\n'+table(
            ['Index','Exact native feature','Encoder group','Training missing','Scaler mean','Scaler scale'],
            [[r[k] for k in ['index','feature','group','train_missing','scaler_mean','scaler_scale']] for r in subset]))
    add('Appendix A. Complete feature dictionary', '\n\n'.join(appendix))
    add('Appendix B. Every completed final epoch',table(
        ['Model','Epoch','Training rows','Train BCE','Mean validation macro-F1','Mean validation AP','Epoch timer s'],
        [[r[k] for k in ['model','epoch','training_rows','train_loss','mean_validation_macro_f1','mean_validation_ap','elapsed_seconds']] for r in epoch_rows])+'''

Exact per-domain validation macro-F1/AP for every epoch is in the
[training history CSV](artifacts/documentation/training_history.csv) and each
candidate's original history JSON. Timers are not complete project wall times.
''')
    add('Appendix C. Reading order and documentation exports', '''
The focused [training parameter reference](TRAINING_PARAMETERS.md) lists optimizer
defaults, learning rate, dropout placement, epoch/early-stopping rules and actual
checkpoint epochs. [Its JSON](artifacts/documentation/training_parameters.json)
provides the same information in machine-readable form.

1. This full process record: rationale, complete procedure, results, development
   history, limitations, operational behavior and reproducibility instructions.
2. [Executed preview](preview.ipynb): interactive tables and saved plots.
3. [Results report](EXPERIMENT_REPORT.md): concise model comparison and figures.
4. [Methods](EXPERIMENT_METHOD.md): shorter final-method description.
5. [Run status](RUN_STATUS.md): completed and pending components.
6. [Feature catalog](artifacts/documentation/feature_catalog.csv),
   [epoch catalog](artifacts/documentation/training_history.csv),
   [artifact inventory](artifacts/documentation/artifact_inventory.csv),
   [source hashes](artifacts/documentation/source_code_sha256.json), and
   [environment](artifacts/documentation/environment.json): machine-readable audit aids.

The full process record and these exports are generated from existing artifacts
by `document_process.py`; narrative development history is explicitly authored
from the work session. This documentation operation does not retrain, recalibrate,
change numerical preprocessing, modify labels/splits, or generate new predictions.
''')
    plot_lines=[]
    for candidate in baseline['candidates']:
        model=candidate['model']
        plot_lines.append(f"### {model} learning curves\n\n![{model} learning curves](artifacts/adapted_experiment/{model}/learning_curves.png)")
    plot_lines.append(table(['Model','Test dataset','Confusion counts/normalized','ROC/PR'],[
        [c['model'],n,
         f"[Confusion](artifacts/adapted_experiment/{c['model']}/test/{n}/confusion_matrix.png)",
         f"[ROC and PR](artifacts/adapted_experiment/{c['model']}/test/{n}/roc_pr.png)"]
        for c in baseline['candidates'] for n in NAMES]))
    plot_lines.append(table(['Domain-prior reference','Confusion','ROC/PR'],[
        [n,f'[Confusion](artifacts/adapted_experiment/domain_prior_reference/{n}/confusion_matrix.png)',
         f'[ROC and PR](artifacts/adapted_experiment/domain_prior_reference/{n}/roc_pr.png)'] for n in NAMES]))
    add('Appendix D. Complete visual result index','\n\n'.join(plot_lines))
    toc='\n'.join(f'- [{s.splitlines()[0][3:]}](#{s.splitlines()[0][3:].lower().replace(".","").replace(",","").replace(":","").replace(" ","-")})' for s in sections)
    document='# Complete process documentation\n\nAudited: 2026-10-31 (Asia/Seoul client date).\n\n'+toc+'\n\n'+'\n'.join(sections)
    target=PROJECT/'FULL_PROCESS_DOCUMENTATION.md'
    target.write_text(document,encoding='utf-8')
    notebook=PROJECT/'preview.ipynb'
    if notebook.exists():
        nb=read(notebook)
        marker='full-process-documentation-link'
        cell=next((c for c in nb['cells'] if c.get('id')==marker),None)
        if cell is None:
            cell=dict(cell_type='markdown',id=marker,metadata={})
            nb['cells'].insert(1,cell)
        cell['source']=[
            'Full procedure, feature dictionary, development history, all results and limitations: ',
            '[Complete process documentation](FULL_PROCESS_DOCUMENTATION.md).\n\n',
            'Optimizer, learning rate, dropout, epoch limits, validation and stopping rules: ',
            '[Training parameter reference](TRAINING_PARAMETERS.md).\n']
        notebook.write_text(json.dumps(nb,indent=2),encoding='utf-8')
    inventory=[]
    for p in sorted(ART.rglob('*')):
        if not p.is_file() or DOC in p.parents:
            continue
        relative=p.relative_to(ART)
        first=relative.parts[0]
        role={'native_splits':'Native labels, membership or provenance',
              'mixed_training':'Mixed training membership/provenance',
              'encoded_inputs':'Native numeric arrays or preprocessing statistics',
              'adapted_experiment':'Final checkpoint, prediction, metric, plot or audit',
              'pilot_feature_tokens':'Excluded development trial',
              'report':'Aggregate result/timing table or figure'}.get(first,'Preflight or auxiliary artifact')
        inventory.append(dict(path=str(p.relative_to(PROJECT)).replace('\\','/'),bytes=p.stat().st_size,role=role))
    csv_file(DOC/'artifact_inventory.csv',inventory)
    summary=dict(documentation_date='2026-10-31',scope='Existing current four-dataset experiment',
                 full_document=target.name,feature_rows=len(features),epoch_rows=len(epoch_rows),
                 final_model_test_evaluations=len(test),timing_configurations=len(timings),
                 inventory_files=len(inventory),source_modules_hashed=len(hashes),
                 model_training_or_test_inference_performed=False,
                 metadata_only_corrections='metadata_description_corrections.json')
    (DOC/'documentation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
