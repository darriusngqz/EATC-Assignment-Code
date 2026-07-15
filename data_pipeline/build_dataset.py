"""
Streams every labeled (label 0 or 1) record out of the raw EMBER2018 tar
and derives this project's own 20-feature schema (src/constants.py's
ORDER_OF_FEATURES) from each record's raw JSON, writing the full labeled
set (800,000 rows: 400,000 malicious + 400,000 benign) straight to CSV.

No sampling here on purpose: the dataset-size decision (keep the full
800,000 rows vs. downsample) is made and justified in
notebooks/02_data_preparation.ipynb, not baked silently into this script.
See that notebook for the reasoning.

Feature derivation logic is intentionally identical to
src/extract_features.py's (which computes the same 20 values from a live
pefile-parsed upload), just against EMBER's JSON shape instead of a parsed
PE object, so training and inference never drift apart.

Usage:
    python data_pipeline/build_dataset.py path/to/ember_dataset_2018_2.tar

Expects the standard EMBER2018 tar layout: ember2018/train_features_0.jsonl
through _5.jsonl, and ember2018/test_features.jsonl. Download EMBER2018
from https://github.com/elastic/ember if you don't already have it.

Takes a while (all ~1,000,000 records, ~800,000 of which are labeled and
kept): expect roughly 1-2 minutes per ~130,000-line member file on
ordinary hardware, a few minutes total.
"""
import csv
import json
import sys
import tarfile

MEMBERS = [
    "ember2018/train_features_0.jsonl",
    "ember2018/train_features_1.jsonl",
    "ember2018/train_features_2.jsonl",
    "ember2018/train_features_3.jsonl",
    "ember2018/train_features_4.jsonl",
    "ember2018/train_features_5.jsonl",
    "ember2018/test_features.jsonl",
]

STANDARD_SECTION_NAMES = {
    ".text", ".data", ".rdata", ".idata", ".edata", ".pdata", ".rsrc",
    ".reloc", ".bss", ".crt", ".tls", ".debug", ".didat", ".apiset", "fothk",
}

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

# Must match src/constants.py's ORDER_OF_FEATURES exactly.
FEATURE_ORDER = [
    "SuspiciousImportFunctions", "SuspiciousNameSection", "SectionsLength",
    "SectionMinEntropy", "SectionMaxEntropy", "SectionMinRawsize", "SectionMaxRawsize",
    "SectionMinVirtualsize", "SectionMaxVirtualsize",
    "DirectoryEntryImport", "NumberOfImportedFunctions",
    "NumStrings", "StringEntropy", "AvgStringLength",
    "NumURLs", "NumRegistryKeys", "NumPaths", "NumMZStrings",
    "FileSize", "VirtualSize",
]

COLUMNS = ["Name"] + FEATURE_ORDER + ["Malware", "OriginalSplit"]


def derive_features(rec):
    sections = rec.get("section", {}).get("sections", []) or []
    entropies, raw_sizes, virt_sizes = [], [], []
    suspicious_names = 0
    for s in sections:
        name = (s.get("name") or "").lower()
        if name not in STANDARD_SECTION_NAMES:
            suspicious_names += 1
        ent = s.get("entropy")
        if ent is not None:
            entropies.append(ent)
        raw_sizes.append(s.get("size", 0) or 0)
        virt_sizes.append(s.get("vsize", 0) or 0)

    def mm(vals):
        return (0, 0) if not vals else (min(vals), max(vals))

    min_ent, max_ent = mm(entropies)
    min_raw, max_raw = mm(raw_sizes)
    min_virt, max_virt = mm(virt_sizes)

    imports = rec.get("imports", {}) or {}
    num_dlls = len(imports)
    total_funcs = 0
    suspicious_funcs = 0
    for dll, funcs in imports.items():
        for fn in funcs:
            total_funcs += 1
            if isinstance(fn, str) and fn.lower() in SUSPICIOUS_IMPORT_FUNCTIONS:
                suspicious_funcs += 1

    strings = rec.get("strings", {}) or {}
    general = rec.get("general", {}) or {}

    return {
        "SuspiciousImportFunctions": suspicious_funcs,
        "SuspiciousNameSection": suspicious_names,
        "SectionsLength": len(sections),
        "SectionMinEntropy": round(min_ent, 4),
        "SectionMaxEntropy": round(max_ent, 4),
        "SectionMinRawsize": min_raw,
        "SectionMaxRawsize": max_raw,
        "SectionMinVirtualsize": min_virt,
        "SectionMaxVirtualsize": max_virt,
        "DirectoryEntryImport": num_dlls,
        "NumberOfImportedFunctions": total_funcs,
        "NumStrings": strings.get("numstrings", 0),
        "StringEntropy": round(strings.get("entropy", 0) or 0, 4),
        "AvgStringLength": round(strings.get("avlength", 0) or 0, 4),
        "NumURLs": strings.get("urls", 0),
        "NumRegistryKeys": strings.get("registry", 0),
        "NumPaths": strings.get("paths", 0),
        "NumMZStrings": strings.get("MZ", 0),
        "FileSize": general.get("size", 0),
        "VirtualSize": general.get("vsize", 0),
    }


def main():
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} path/to/ember_dataset_2018_2.tar")
        sys.exit(1)
    tar_path = sys.argv[1]
    out_path = "data/dataset_pe_v2_full.csv"

    total_written = 0
    with open(out_path, "w", newline="") as out, tarfile.open(tar_path, "r") as tar:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()

        for member_name in MEMBERS:
            split = "test" if "test" in member_name else "train"
            member_written = 0
            f = tar.extractfile(member_name)
            for raw_line in f:
                try:
                    rec = json.loads(raw_line)
                except Exception:
                    continue
                label = rec.get("label", -1)
                if label not in (0, 1):
                    continue  # skip unlabeled rows
                row = derive_features(rec)
                row["Name"] = rec.get("sha256", "")
                row["Malware"] = label
                row["OriginalSplit"] = split
                writer.writerow(row)
                member_written += 1
            total_written += member_written
            print(f"{member_name}: wrote {member_written} labeled rows "
                  f"(total so far: {total_written})")

    print(f"done. wrote {total_written} rows to {out_path}")


if __name__ == "__main__":
    main()
