"""
constants.py defines the shared variables the data pipeline depends on: the
20 feature columns (in order), the label definition, and the standard
lists used to derive suspicious-import and suspicious-section signals.
This file ensures the columns used to train the model are the exact same
columns, in the exact same order, that the deployed app uses to make a
prediction. See src/extract_features.py for how each one is computed from
a real uploaded file, and data_pipeline/build_dataset.py for how each one
is computed from EMBER's raw JSON at training-data-build time (same logic,
different input shape).

The 20-feature schema combines section-table statistics, import-richness
signals, and string-based signals, deliberately excluding raw header
fields (Machine, Characteristics, Subsystem) that structurally separate
DLLs from EXEs rather than distinguishing malicious from benign behaviour.
This anti-shortcut design is validated directly in notebooks/03_eda.ipynb,
which confirms no feature in this set acts as a near-perfect proxy for
the label.
"""

# The 20 behavioural feature columns, in the exact order the model expects.
ORDER_OF_FEATURES = [
    # Section table stats: packing/entropy signal, validated as genuine
    # signal (not a DLL/EXE shortcut) in notebooks/03_eda.ipynb
    "SuspiciousImportFunctions", "SuspiciousNameSection", "SectionsLength",
    "SectionMinEntropy", "SectionMaxEntropy", "SectionMinRawsize", "SectionMaxRawsize",
    "SectionMinVirtualsize", "SectionMaxVirtualsize",
    # Import richness: counts both suspicious-function hits and overall
    # import volume, since a file with almost no imports at all (common in
    # packed/obfuscated malware) is itself a signal worth capturing
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
STANDARD_SECTION_NAMES = {
    ".text", ".data", ".rdata", ".idata", ".edata", ".pdata", ".rsrc",
    ".reloc", ".bss", ".crt", ".tls", ".debug", ".didat", ".apiset", "fothk",
}

# WinAPI functions frequently seen in malicious PE imports (process
# injection, anti-debugging, persistence, command-and-control).
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
# (confirmed directly against the raw EMBER JSON).
MALICIOUS_LABEL = 1
BENIGN_LABEL = 0

# Fixed seed so every rerun of the code produces the same output and splits.
RANDOM_STATE = 42
