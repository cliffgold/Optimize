Inbox.csv is the way Overlord.xlsm commuicates with Optimize.py.  Row 3 contains per-year multipliers, where that makes sense.

Knobs_Debug.csv is a sample set of knobs over a 12-year period.  If you want to control the knobs and just run one case (really fast), copy this into directory Optimize/Analyse/Knobs_Debug.csv, edit as required, and set debug_one_case in Optimize.py.