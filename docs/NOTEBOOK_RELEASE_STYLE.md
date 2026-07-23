# CPS notebook release standard

The Colab notebooks are public research instruments. They must be easy to enter, difficult to misread, and complete enough to execute from a fresh runtime.

## Required structure

Every release notebook contains, in this order:

1. a numbered title and one-sentence purpose;
2. a release contract stating the scientific question, default path, evidence boundary, and outputs;
3. an interpretation checklist;
4. a repository/bootstrap cell;
5. a visible runtime inventory;
6. numbered stages with declared objectives and evidence products;
7. a single export step producing `/content/cps-export.zip`.

## Language doctrine

- State the scientific question before implementation details.
- Distinguish measurement, hypothesis, intervention, and evidence.
- Use prospective language for planner outputs.
- Name limitations at the point where a result could be overread.
- Prefer concrete nouns and verbs over promotional language.
- Report negative or partial outcomes without changing the acceptance criterion.

## Visual doctrine

The visual system is restrained and functional. Stage cards use one accent, one neutral surface, strong typographic hierarchy, and no decorative illustration. Tables and figures carry the scientific content; styling only clarifies navigation and evidence boundaries. The theme supports light and dark system preferences.

## Self-containment contract

A notebook must execute from a fresh Colab runtime using the repository revision and declared public data sources. A previous notebook may provide an optional cache or evidence packet, but it may not be required unless the notebook explicitly declares itself as a campaign continuation artifact.

## Export contract

Every notebook writes its products beneath `/content/cps-artifacts` and ends by creating `/content/cps-export.zip`. The archive must include enough metadata to determine the model revision, evidence class, active backends, configuration, and generated result files.

## Acceptance checklist

A notebook is release-ready only when:

- all code cells compile;
- the notebook validator passes;
- self-containment tests pass;
- stage numbering and visible runtime banners agree;
- evidence boundaries are explicit;
- the export path is consistent;
- no output claims exceed the executed evidence.
