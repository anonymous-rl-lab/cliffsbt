# Security and safety scope

`sbt-monitor` is research software for measurement and monitoring. It is not a safety-rated control system and must not be used as the sole basis for medical, industrial, transportation, financial, or other high-consequence actions.

Please report software vulnerabilities privately to the project maintainers through the security contact configured in the public repository before release. Do not include sensitive deployment data in a public issue.

The package deliberately does not:

- choose an operational or safety threshold;
- ship a pretrained warning readout;
- trigger model updates, shutdowns, or physical actions;
- suppress pairing, missingness, calibration, or model-fingerprint warnings.
