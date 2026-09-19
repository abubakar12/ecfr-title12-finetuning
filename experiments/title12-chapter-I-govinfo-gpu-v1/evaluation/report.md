# Base Instruct versus continued pretraining

Seen regulations, unseen questions; no retrieval. Claim support was reviewed blind to checkpoint identity.

```json
{
  "base": {
    "citation_accuracy": 0.0,
    "citation_precision": 0.008417508417508417,
    "citation_coverage": 0.19888888888888887,
    "substantive_accuracy": 0.1,
    "section_accuracy": 0.022222222222222223,
    "paragraph_accuracy": null,
    "nonexistent_citation_count": 0.4,
    "unresolved_section_citation_count": 2.7666666666666666,
    "outside_scope_accuracy": 0.1
  },
  "cpt": {
    "citation_accuracy": 0.0,
    "citation_precision": 0.009387526054192722,
    "citation_coverage": 0.2148148148148148,
    "substantive_accuracy": 0.1111111111111111,
    "section_accuracy": 0.022222222222222223,
    "paragraph_accuracy": null,
    "nonexistent_citation_count": 0.4777777777777778,
    "unresolved_section_citation_count": 1.6666666666666667,
    "outside_scope_accuracy": 0.0
  },
  "paired_primary_95_percent_ci": {
    "delta": 0.0,
    "lower": 0.0,
    "upper": 0.0
  },
  "review_sha256": "8611b54477256ed02c28a17291456b021dbd213a54a88144f501f8445fbc2f48"
}
```
