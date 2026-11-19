# Submission declarations (*Array* / Elsevier)

Fill every item before journal upload. Incomplete declarations are a desk-rejection risk.

## Manuscript scope (required wording)
- [ ] Title names **five-class joke-type / structured-play video classification** (not “responsive caregiving assessment”)
- [ ] Abstract ≤ **250 words**; no caregiving-measurement claims beyond the joke-type label
- [ ] Discussion/conclusion do **not** claim reciprocity, sensitivity, bidirectional feedback, developmental monitoring, or intervention readiness unless separately measured
- [ ] Explicit “not evaluated” paragraph: responsiveness, deployment, cultural transfer, clinical use

## Corresponding author
- [ ] **One** corresponding author: __________________
- [ ] Email: __________________
- [ ] Equal-contribution statement clarified (if used)

## CRediT authorship
- [ ] Conceptualization:
- [ ] Methodology / software:
- [ ] Validation / formal analysis:
- [ ] Investigation / data curation:
- [ ] Writing — original draft / review & editing:
- [ ] Supervision / funding acquisition:

## Competing interests
- [ ] No known competing interests, **or** disclose: __________________

## Funding
- [ ] No specific grant, **or** list grants: __________________

## Ethics / IRB
- [ ] Approval body: Goldsmiths, University of London
- [ ] Approval reference: 1408
- [ ] **Approval date:** ____ / ____ / ________
- [ ] Approval documentation available to editors on request

## Consent
- [ ] Informed consent from caregivers / legal guardians
- [ ] Consent for publication of any identifiable stills
- [ ] Data-sharing permissions stated

## Generative AI use
- [ ] None used, **or** disclose tools/roles: __________________

## Permissions
- [ ] Rights cleared for all figures / reused material

## Data / code availability
- Code: this repository (`scripts/`, `models/`, `src/`, `config/shared_fold_indices.json`)
- Reference fold artifacts: `analysis/reference_results/` (predictions, performance CSV, inner-split JSONs, manifests)
- Restricted videos: under the study data-sharing agreement (not in git)
- Reproduction limit: end-to-end video retrain requires local `data/videos/` + workbook

## Citation / presentation audit
- [ ] No `[??]`, `???`, duplicate references, or editorial notes in the PDF
- [ ] No visible revision colouring / blank pages
- [ ] Table 3 units match values (percent vs proportion)
- [ ] Metric aggregation defined (fold-macro vs pooled-macro vs weighted)
- [ ] Methods text matches **implemented** pipeline: 20 fps timeline; T=20 @ 224²; uniform∪high-motion sampling; augmentation **on** (rotation/brightness/flip/jitter/drop); ES patience **25**; nested selection then outer-train refit

## Scope note
This repository evaluates **joke-type classification** under infant-held-out nested CV.
Do **not** claim automated measurement of responsive caregiving unless that construct is separately validated.
