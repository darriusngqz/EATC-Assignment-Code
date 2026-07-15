"""
constants.py defines the shared variables the v2 pipeline depends on: the
20 feature columns (in order), the label definition, and the standard
lists used to derive suspicious-import and suspicious-section signals.
This file ensures the columns used to train the model are the exact same
columns, in the exact same order, that the live app uses to make a
prediction. See src/extract_features.py for how each one is computed from
a real uploaded file, and data_pipeline/build_dataset.py for how each one
is computed from EMBER's raw JSON at training-data-build time (same logic,
different input shape).

Why these 20 and not the old 10: the v1 model (legacy_v1/) used only 10
section-derived features. Real-world testing (legacy_v1/notebooks/07)
found this model flagged ~90% of genuine Windows System32 files as
malicious. Two contributing causes were identified: the v1 training data
(a fixed ~19,600-row academic CSV) did not represent typical real-world
software well, and 10 features gave the model little to work with beyond
section-entropy shortcuts. v2 addresses both: EMBER2018 (VirusTotal-sourced,
real-world software) replaces the training data, and the feature set is
widened to include import richness and string-based signals, both
well-established, non-file-type-identifying malware signals. Deliberately
still excludes raw header fields (Machine, Characteristics, Subsystem, ...)
that structurally separate DLLs from EXEs rather than malicious from
benign, that was the original shortcut-learning trap legacy_v1/notebooks/
03_eda.ipynb caught and this project does not want to reintroduce it.
"""

# The 20 behavioural feature columns, in the exact order the model expects.
ORDER_OF_FEATURES = [
    # Section table stats (packing/entropy signal, same spirit as v1's
    # CORE_TRAITS, kept because they were validated to carry genuine signal,
    # not just a DLL/EXE shortcut, in legacy_v1/notebooks/03_eda.ipynb)
    "SuspiciousImportFunctions", "SuspiciousNameSection", "SectionsLength",
    "SectionMinEntropy", "SectionMaxEntropy", "SectionMinRawsize", "SectionMaxRawsize",
    "SectionMinVirtualsize", "SectionMaxVirtualsize",
    # Import richness (new: v1 only counted suspicious-function hits, not
    # overall import volume, so a file with 0 suspicious hits but almost no
    # imports at all, common in packed/obfuscated malware, was invisible)
    "DirectoryEntryImport", "NumberOfImportedFunctions",
    # String-based signals (new category entirely; EMBER's own feature
    # extraction defines a "string" as 5+ consecutive printable characters,
    # matched here so training and inference compute this identically)
    "NumStrings", "StringEntropy", "AvgStringLength",
    "NumURLs", "NumRegistryKeys", "NumPaths", "NumMZStrings",
    # Overall size (packers/droppers often show a large raw-vs-virtual-size
    # mismatch; this is file-content-derived, not a file-type flag)
    "FileSize", "VirtualSize",
]

# Defines normal section names for a legitimate Windows PE file. Any
# section name not listed here adds one signal of possible malware.
# Unchanged from legacy_v1, this list was already correct.
STANDARD_SECTION_NAMES = {
    ".text", ".data", ".rdata", ".idata", ".edata", ".pdata", ".rsrc",
    ".reloc", ".bss", ".crt", ".tls", ".debug", ".didat", ".apiset", "fothk",
}

# WinAPI functions frequently seen in malicious PE imports (process
# injection, anti-debugging, persistence, C2). Unchanged from legacy_v1.
SUSPICIOUS_IMPORT_FUNCTIONS = {
    "virtualalloc", "virtualallocex", "virtualprotect", "virtualprotectex",
    "writeprocessmemory", "readprocessmemory", "createremotethread",
    "createremotethreadex", "ntunmapviewofsection", "zwunmapviewofsection",
    "queueuserapc", "setwindowshookexa", "setwindowshookexw",
    "isdebuggerpresent", "checkremotedebuggerpresent", "ntqueryinformationprocess",
    "outputdebugstringa", "getprocaddress", "loadlibrarya", "loadlibraryw",
    "loadlibraryexa", "loadlibraryexw", "winexec", "shellexecutea", "shellexecutew",
    "urldownloadtofilea", "urldownloadtofilew", "internetopena", "internetopenurla",
    "internetreadfile", "httpsendrequesta", "httpsendrequestw", "createprocessa",
    "createprocessw", "getasynckeystate", "setfileattributesa", "setfileattributesw",
    "cryptencrypt", "cryptdecrypt", "adjusttokenprivileges", "openprocesstoken",
    "resumethread", "suspendthread", "regsetvalueexa", "regsetvalueexw",
    "regcreatekeyexa", "regcreatekeyexw", "findfirstfilea", "findnextfilea",
    "deletefilea", "deletefilew",
}

# A "string" is 5+ consecutive printable ASCII characters, matching
# EMBER's own definition so training (from EMBER JSON) and inference
# (from a live uploaded file) count strings identically.
MIN_STRING_LENGTH = 5

# Identifier and label columns.
ID_COLUMNS = ["Name"]  # holds each sample's SHA-256 hash (from EMBER) or a filename
LABEL_COLUMN = "Malware"

# 1 = malicious, 0 = benign. Matches EMBER's own "label" convention
# (confirmed directly against the raw EMBER JSON) and legacy_v1's.
MALICIOUS_LABEL = 1
BENIGN_LABEL = 0

# Fixed seed so every rerun of the code produces the same output and splits.
RANDOM_STATE = 42
