"""
constants.py defines the shared variables this project depends on: the
77 feature (numeric) columns and their order, the label definition, and
the different feature-set groupings built from them. The full dataset has
79 columns in total (77 features, 1 Name, and 1 Malware binary column). This
file ensures the columns used to train the model are the exact same
columns, in the exact same order, that the live app and model use to
make a prediction.
"""

# The 77 numeric feature columns, in the exact order the model expects.
ORDER_OF_FEATURES = [
    # IMAGE_DOS_HEADER (MS-DOS stub header, 17 raw fields)
    "e_magic", "e_cblp", "e_cp", "e_crlc", "e_cparhdr", "e_minalloc", "e_maxalloc",
    "e_ss", "e_sp", "e_csum", "e_ip", "e_cs", "e_lfarlc", "e_ovno", "e_oemid",
    "e_oeminfo", "e_lfanew",
    # IMAGE_FILE_HEADER (COFF header, 7 raw fields)
    "Machine", "NumberOfSections", "TimeDateStamp", "PointerToSymbolTable",
    "NumberOfSymbols", "SizeOfOptionalHeader", "Characteristics",
    # IMAGE_OPTIONAL_HEADER (28 raw fields)
    "Magic", "MajorLinkerVersion", "MinorLinkerVersion", "SizeOfCode",
    "SizeOfInitializedData", "SizeOfUninitializedData", "AddressOfEntryPoint",
    "BaseOfCode", "ImageBase", "SectionAlignment", "FileAlignment",
    "MajorOperatingSystemVersion", "MinorOperatingSystemVersion", "MajorImageVersion",
    "MinorImageVersion", "MajorSubsystemVersion", "MinorSubsystemVersion",
    "SizeOfHeaders", "CheckSum", "SizeOfImage", "Subsystem", "DllCharacteristics",
    "SizeOfStackReserve", "SizeOfStackCommit", "SizeOfHeapReserve", "SizeOfHeapCommit",
    "LoaderFlags", "NumberOfRvaAndSizes",
    # Derived / engineered features (25 fields)
    "SuspiciousImportFunctions", "SuspiciousNameSection", "SectionsLength",
    "SectionMinEntropy", "SectionMaxEntropy", "SectionMinRawsize", "SectionMaxRawsize",
    "SectionMinVirtualsize", "SectionMaxVirtualsize", "SectionMaxPhysical",
    "SectionMinPhysical", "SectionMaxVirtual", "SectionMinVirtual",
    "SectionMaxPointerData", "SectionMinPointerData", "SectionMaxChar", "SectionMainChar",
    "DirectoryEntryImport", "DirectoryEntryImportSize", "DirectoryEntryExport",
    "ImageDirectoryEntryExport", "ImageDirectoryEntryImport", "ImageDirectoryEntryResource",
    "ImageDirectoryEntryException", "ImageDirectoryEntrySecurity",
]

# Defines the name of the malware.
ID_COLUMNS = ["Name"]

# Binary column that defines whether the sample is malicious or benign.
LABEL_COLUMN = "Malware"

# 1 = malicious, 0 = benign. Confirmed against real rows: VirusShare-hash
# rows are all 1, known-legitimate files (winhttp.dll, ldifde.exe) are all 0.
MALICIOUS_LABEL = 1
BENIGN_LABEL = 0

# Fixed seed so that every rerun of the code produces the same output and splits.
RANDOM_STATE = 42

# Data preprocessing process (calculating variance) revealed that 7 columns have zero variance
# across all rows. These columns are likely to be a dataset error and are dropped to ensure that
# a real file's genuine non-zero value is not mistaken for an anomaly.
DROPPED_FEATURES = [
    "SectionMaxEntropy", "SectionMaxRawsize", "SectionMaxVirtualsize",
    "SectionMinPhysical", "SectionMinVirtual", "SectionMinPointerData", "SectionMainChar",
]

# Behavioural features excludes the raw header fields and characteristics of DLLs and EXE files, as
# they reflect the type of file instead of what the file actually does. This allows the model to
# judge the malware behaviour without any influence from the file type, avoiding shortcuts or
# inaccurate signals.
BEHAVIORAL_FEATURES = [f for f in ORDER_OF_FEATURES[52:] if f not in DROPPED_FEATURES]

# To streamline behavioural features even more, CORE_TRAITS drops the
# directory fields since they correlate with DLL/EXE structure, not actual
# malware behaviour. Keeps only traits relevant to malicious behaviour including:
# suspicious imports, odd section naming, and section entropy/size.
CORE_TRAITS = [f for f in [
    "SuspiciousImportFunctions", "SuspiciousNameSection", "SectionsLength",
    "SectionMinEntropy", "SectionMaxEntropy", "SectionMinRawsize", "SectionMaxRawsize",
    "SectionMinVirtualsize", "SectionMaxVirtualsize", "SectionMaxPhysical",
    "SectionMinPhysical", "SectionMaxVirtual", "SectionMinVirtual",
    "SectionMaxPointerData", "SectionMinPointerData", "SectionMaxChar", "SectionMainChar",
] if f not in DROPPED_FEATURES]

# Raw header fields only, no derived features at all. Allows us to check if a
# problem (e.g. false positives) comes from the derived features, or would
# happen anyway even without them.
# e_magic is a PE file header and signature of Windows executable files as a number ("MZ").
# Dropped as it does not tell us any meaningful information about whether a file is malicious.
RAW_HEADER_FEATURES = [f for f in ORDER_OF_FEATURES[:52] if f != "e_magic"]
