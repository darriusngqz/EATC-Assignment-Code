"""
extract_features.py turns a raw .exe/.dll file's bytes into the 20 numbers
listed in ORDER_OF_FEATURES. Used by both training (data_pipeline/ derives
the same 20 values from EMBER's raw JSON instead of a live file) and the
deployed app, so both always compute features the exact same way. If
separate code paths existed, they could slowly drift apart and the app
would end up feeding the model numbers computed differently from what it
was trained on.

Security boundary: only parses the file's static structure via pefile and
a plain byte-level string scan. Never executes, runs, or unpacks the file
beyond reading its headers/sections and looking for printable-character
runs.
"""
import re

from constants import (
    ORDER_OF_FEATURES,
    STANDARD_SECTION_NAMES,
    SUSPICIOUS_IMPORT_FUNCTIONS,
    MIN_STRING_LENGTH,
)

# Matches runs of 5+ consecutive printable ASCII characters (0x20-0x7e),
# the same definition EMBER uses for its "strings" features.
_PRINTABLE_RUN = re.compile(rb"[\x20-\x7e]{%d,}" % MIN_STRING_LENGTH)


# Returns (min, max) of a list, or (0, 0) if empty (e.g. a file with no
# sections, or no entropy could be computed for any section).
def _safe_min_max(values):
    if not values:
        return 0, 0
    return min(values), max(values)


# pefile stores section names as a fixed 8-byte chunk padded with null
# bytes; cut at the first null byte to get the readable name.
def _section_name(section):
    raw = getattr(section, "Name", b"")
    if isinstance(raw, bytes):
        return raw.split(b"\x00")[0].decode("latin-1", errors="replace")
    return str(raw).split("\x00")[0]


# Walks every section and computes entropy/size stats plus a count of
# unrecognised section names. Packed or obfuscated code tends to show
# unusually high section entropy.
def _section_derived_fields(pe):
    sections = list(getattr(pe, "sections", []))
    entropies, raw_sizes, virt_sizes = [], [], []
    suspicious_names = 0

    for section in sections:
        name = _section_name(section).lower()
        if name not in STANDARD_SECTION_NAMES:
            suspicious_names += 1
        try:
            entropies.append(section.get_entropy())
        except Exception:
            pass
        raw_sizes.append(getattr(section, "SizeOfRawData", 0))
        virt_sizes.append(getattr(section, "Misc_VirtualSize", 0))

    min_ent, max_ent = _safe_min_max(entropies)
    min_raw, max_raw = _safe_min_max(raw_sizes)
    min_virt, max_virt = _safe_min_max(virt_sizes)

    return {
        "SuspiciousNameSection": suspicious_names,
        "SectionsLength": len(sections),
        "SectionMinEntropy": min_ent, "SectionMaxEntropy": max_ent,
        "SectionMinRawsize": min_raw, "SectionMaxRawsize": max_raw,
        "SectionMinVirtualsize": min_virt, "SectionMaxVirtualsize": max_virt,
    }


# Counts imported DLLs, total imported functions, and how many imported
# function names match SUSPICIOUS_IMPORT_FUNCTIONS. Also reports the raw
# import volume (NumberOfImportedFunctions): a file with almost no imports
# at all is itself a signal, common in packed malware that resolves
# everything dynamically at runtime instead.
def _import_derived_fields(pe):
    if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        return {
            "SuspiciousImportFunctions": 0,
            "DirectoryEntryImport": 0,
            "NumberOfImportedFunctions": 0,
        }

    suspicious = 0
    total_funcs = 0
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        for imp in entry.imports:
            total_funcs += 1
            name = imp.name
            if name is None:
                continue
            if isinstance(name, bytes):
                name = name.decode("latin-1", errors="replace")
            if name.lower() in SUSPICIOUS_IMPORT_FUNCTIONS:
                suspicious += 1

    return {
        "SuspiciousImportFunctions": suspicious,
        "DirectoryEntryImport": len(pe.DIRECTORY_ENTRY_IMPORT),
        "NumberOfImportedFunctions": total_funcs,
    }


# Scans the raw file bytes for printable-character runs of 5+ (matching
# EMBER's own "string" definition) and summarises them: how many, average
# length, the entropy of the combined string bytes, and simple counts of
# patterns that tend to appear in dropped paths, C2 URLs, and registry
# persistence (paths, URLs, registry keys, and embedded "MZ" headers that
# can indicate a bundled/dropped second executable).
def _string_derived_fields(file_bytes):
    runs = _PRINTABLE_RUN.findall(file_bytes)
    num_strings = len(runs)
    if num_strings == 0:
        return {
            "NumStrings": 0, "StringEntropy": 0.0, "AvgStringLength": 0.0,
            "NumURLs": 0, "NumRegistryKeys": 0, "NumPaths": 0, "NumMZStrings": 0,
        }

    total_len = sum(len(r) for r in runs)
    avg_len = total_len / num_strings

    combined = b"".join(runs)
    counts = {}
    for b in combined:
        counts[b] = counts.get(b, 0) + 1
    entropy = 0.0
    for c in counts.values():
        p = c / len(combined)
        entropy -= p * (p and __import__("math").log2(p))

    num_urls = sum(r.count(b"http://") + r.count(b"https://") for r in runs)
    num_registry = sum(r.count(b"HKEY_") for r in runs)
    num_paths = sum(r.count(b"C:\\") for r in runs)
    num_mz = sum(r.count(b"MZ") for r in runs)

    return {
        "NumStrings": num_strings,
        "StringEntropy": round(entropy, 4),
        "AvgStringLength": round(avg_len, 4),
        "NumURLs": num_urls,
        "NumRegistryKeys": num_registry,
        "NumPaths": num_paths,
        "NumMZStrings": num_mz,
    }


# File size (on disk) and virtual size (in memory once loaded), both from
# content-derived structure, not a file-type flag. A large gap between the
# two is a common packer/dropper signal.
def _size_fields(pe, file_bytes):
    return {
        "FileSize": len(file_bytes),
        "VirtualSize": getattr(pe.OPTIONAL_HEADER, "SizeOfImage", 0),
    }


# Computes every feature in feature_order from an already-parsed PE object
# plus the original raw bytes (needed separately for the string scan).
# Kept apart from extract_pe_features so tests can use a fake pe object.
def _features_from_pe(pe, file_bytes, feature_order=ORDER_OF_FEATURES):
    values = {}
    values.update(_section_derived_fields(pe))
    values.update(_import_derived_fields(pe))
    values.update(_string_derived_fields(file_bytes))
    values.update(_size_fields(pe, file_bytes))
    return [values[name] for name in feature_order]


# Parses raw file bytes as a PE file, returns a feature row. Never executes
# the file, only parses its structure. Raises an exception if file_bytes
# isn't a valid PE file, callers must catch this (see app.py).
def extract_pe_features(file_bytes, feature_order=ORDER_OF_FEATURES):
    import pefile

    # fast_load skips auto-parsing every directory up front (faster, avoids
    # crashing on malformed directories this project doesn't need).
    pe = pefile.PE(data=file_bytes, fast_load=True)
    try:
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
        ])
    except Exception:
        pass
    return _features_from_pe(pe, file_bytes, feature_order)
