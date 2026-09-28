"""Turku (OSF pehj9) locked-site handling: download | seal | evaluate.

download: page the OSF API, fetch the 40 raw eyes-open .set files plus
InfoAboutParticipants.xlsx and the preprocessing .m file into
/Volumes/Extreme Pro/EEG data/locked-sites/osf-pehj9/raw/, with a
manifest.json of sha256/bytes/OSF version.

seal: read InfoAboutParticipants.xlsx once, write labels to
locked-sites/osf-pehj9/sealed/labels.json, and record its sha256 in
data/pd-runs/v4/locked-site-seal.json. load_labels() must raise unless
both outer-models-lock.json and final-models-lock.json exist.

evaluate: the one-shot Turku evaluation. Refuses a second run without
--i-understand-this-is-a-repeat, which also stamps results as
non-confirmatory.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
See docs/PD_PROTOCOL_V4.md (frozen before any supervised fit) for the exact
rules this module must implement.
"""
