"""
extract_features.py is responsible for extracting the data from the PE file (.exe/.dll).
turning a raw .exe/.dll file into the 77 numbers listed in ORDER_OF_FEATURES.
Features extracted are used by both training and the deployed app. Data training
and the deployed app both use the same method of feature extraction to ensure that
they both always calculate features in the same exact way. If separate codes are used,
they could slowly become inconsistent and the app would end up feediung the model
numbers computed differntly from training data.

To ensure adequate security, extract_features.py only parse the bytes and interprets the
file's header structure via pefile. It does not execute, run  or unpack the file beyond
what is required to read the headers.
"""

from constants import ORDER_OF_FEATURES

# Defines normal section names for a legitimate Windows PE file. Any section names that are not listed
# adds one signal of possible malware.
STANDARD_SECTION_NAMES = {
    ".text", ".data", ".rdata", ".idata", ".edata", ".pdata", ".rsrc",
    ".reloc", ".bss", ".crt", ".tls", ".debug", ".didat", ".apiset", "fothk",
}

# Defines WinAPI functions frequently seen in malicious PE imports such a process
# injection, anti-debugging, persistence, and command-and-control (C2) servers.
# Imported function names not listed here adds one signal of possible malwares
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

# pefile OPTIONAL_HEADER.DATA_DIRECTORY entries, named consistently across
# the standard IMAGE_DIRECTORY_ENTRY_* constants.
DIR_EXPORT = "IMAGE_DIRECTORY_ENTRY_EXPORT"
DIR_IMPORT = "IMAGE_DIRECTORY_ENTRY_IMPORT"
DIR_RESOURCE = "IMAGE_DIRECTORY_ENTRY_RESOURCE"
DIR_EXCEPTION = "IMAGE_DIRECTORY_ENTRY_EXCEPTION"
DIR_SECURITY = "IMAGE_DIRECTORY_ENTRY_SECURITY"


# Returns (min, max) of a list, or (0, 0) if empty (a file with no sections).
def _safe_min_max(values): 
    if not values:                                                          # true only when the list is empty
        return 0, 0                                                         # no sections -> return a safe default
    return min(values), max(values)                                         # otherwise return the smallest and largest value


# Reads a section name
# Since pefile stores names as a fixed 8-byte chunk padded with null bytes at the end,
# the functioin cuts the string off at the first null byte turning the remaining bytes 
# into readable text
def _section_name(section):
    raw = getattr(section, "Name", b"")                                     # read raw file name, default b"" if missing
    if isinstance(raw, bytes):
        return raw.split(b"\x00")[0].decode("latin-1", errors="replace")
    return str(raw).split("\x00")[0]


# Finds one DATA_DIRECTORY entry by name (e.g. import table), or None if
# this PE file has no such entry.
def _data_directory(pe, entry_name):
    for entry in getattr(pe.OPTIONAL_HEADER, "DATA_DIRECTORY", []):         # loop through every directory entry in the Optional Header's table
        if getattr(entry, "name", None) == entry_name:                      # getattr -> if DATA_DIRECTORY does not exist, loop over an empty list instead of crashing
            return entry
    return None


# Memory address (RVA) of a DATA_DIRECTORY entry, or 0 if absent. Raw
# address, not a 0/1 presence flag, values run into the billions.
def _directory_rva(pe, entry_name):
    entry = _data_directory(pe, entry_name)                                 # finds the specific directory entry (1 if real entry, None if pefile dosent have one)
    return int(getattr(entry, "VirtualAddress", 0) or 0)                    # reads the VirtualAddress value from entry. If entry = None, return 0 instead of crashing


# Reads the 17 DOS header fields directly, raw values, no calculation.
def _dos_header_fields(pe):
    dh = pe.DOS_HEADER                                                      # pefile's object holding the 17 DOS header fields
    return {                                                                # copies the existing field's value into the dictionary
        "e_magic": dh.e_magic, "e_cblp": dh.e_cblp, "e_cp": dh.e_cp,
        "e_crlc": dh.e_crlc, "e_cparhdr": dh.e_cparhdr, "e_minalloc": dh.e_minalloc,
        "e_maxalloc": dh.e_maxalloc, "e_ss": dh.e_ss, "e_sp": dh.e_sp,
        "e_csum": dh.e_csum, "e_ip": dh.e_ip, "e_cs": dh.e_cs,
        "e_lfarlc": dh.e_lfarlc, "e_ovno": dh.e_ovno, "e_oemid": dh.e_oemid,
        "e_oeminfo": dh.e_oeminfo, "e_lfanew": dh.e_lfanew,
    }


# Reads the 7 COFF/File header fields directly, raw values, no calculation.
def _file_header_fields(pe):
    fh = pe.FILE_HEADER                                                     # pefile's object holding the 7 COFF/File header fields
    return {                                                                # copies the existing field's value into the directory
        "Machine": fh.Machine, "NumberOfSections": fh.NumberOfSections,
        "TimeDateStamp": fh.TimeDateStamp, "PointerToSymbolTable": fh.PointerToSymbolTable,
        "NumberOfSymbols": fh.NumberOfSymbols, "SizeOfOptionalHeader": fh.SizeOfOptionalHeader,
        "Characteristics": fh.Characteristics,
    }


# Reads the 28 Optional header fields directly, raw values, no calculation.
def _optional_header_fields(pe):
    oh = pe.OPTIONAL_HEADER                                                 # pefile's object holding the 28 Optional header fields
    return {                                                                # copies the existing field's value into the directory
        "Magic": oh.Magic,
        "MajorLinkerVersion": oh.MajorLinkerVersion, "MinorLinkerVersion": oh.MinorLinkerVersion,
        "SizeOfCode": oh.SizeOfCode, "SizeOfInitializedData": oh.SizeOfInitializedData,
        "SizeOfUninitializedData": oh.SizeOfUninitializedData,
        "AddressOfEntryPoint": oh.AddressOfEntryPoint, "BaseOfCode": oh.BaseOfCode,
        "ImageBase": oh.ImageBase, "SectionAlignment": oh.SectionAlignment,
        "FileAlignment": oh.FileAlignment,
        "MajorOperatingSystemVersion": oh.MajorOperatingSystemVersion,
        "MinorOperatingSystemVersion": oh.MinorOperatingSystemVersion,
        "MajorImageVersion": oh.MajorImageVersion, "MinorImageVersion": oh.MinorImageVersion,
        "MajorSubsystemVersion": oh.MajorSubsystemVersion,
        "MinorSubsystemVersion": oh.MinorSubsystemVersion,
        "SizeOfHeaders": oh.SizeOfHeaders, "CheckSum": oh.CheckSum,
        "SizeOfImage": oh.SizeOfImage, "Subsystem": oh.Subsystem,
        "DllCharacteristics": oh.DllCharacteristics,
        "SizeOfStackReserve": oh.SizeOfStackReserve, "SizeOfStackCommit": oh.SizeOfStackCommit,
        "SizeOfHeapReserve": oh.SizeOfHeapReserve, "SizeOfHeapCommit": oh.SizeOfHeapCommit,
        "LoaderFlags": oh.LoaderFlags, "NumberOfRvaAndSizes": oh.NumberOfRvaAndSizes,
    }


# Walks every section (.text, .data, etc.) and computes stats across them:
# entropy (packing/encryption sign), size, naming, layout. Main source of
# genuine malware signal, packed malware tends to have very high entropy.
def _section_derived_fields(pe):
    sections = list(getattr(pe, "sections", []))                            # every section in the file (.text, .data, etc.)

    entropies, raw_sizes, virtual_sizes = [], [], []                        # will hold one value per section
    physical, virtual_addr, pointer_data, characteristics = [], [], [], []
    suspicious_names = 0                                                    # counts sections with an unrecognised name
    main_char = 0                                                           # will hold the "main" section's Characteristics flags
    max_raw_seen = -1                                                       # tracks the biggest raw size seen so far

    for section in sections:                                                # loop once per section
        name = _section_name(section)                                       # extract section name value
        if name.lower() not in STANDARD_SECTION_NAMES:
            suspicious_names += 1                                           # if the section name is unusual name -> count it as a signal of malicious PE

        try:
            entropies.append(section.get_entropy())                         # 0 = uniform/predictable, 8 = fully random (packing sign)
        except Exception:
            pass                                                            # a damaged section can fail calculation so we use try/except

        raw = getattr(section, "SizeOfRawData", 0)
        raw_sizes.append(raw)                                               # size on disk
        virtual_sizes.append(getattr(section, "Misc_VirtualSize", 0))       # size in memory
        physical.append(getattr(section, "Misc_PhysicalAddress", getattr(section, "Misc_VirtualSize", 0)))
        virtual_addr.append(getattr(section, "VirtualAddress", 0))          # where it loads in memory
        pointer_data.append(getattr(section, "PointerToRawData", 0))        # where it sits in the file
        char = getattr(section, "Characteristics", 0)
        characteristics.append(char)                                        # this section's flags

        # "Main" section = largest raw size, usually holds the bulk of the code (.text). An undocumented but reasonable heuristic.
        if raw > max_raw_seen:
            max_raw_seen = raw
            main_char = char                                                # main section's its flags

    min_entropy, max_entropy = _safe_min_max(entropies)                     # collapse each per-section list down into just (smallest, largest)
    min_raw, max_raw = _safe_min_max(raw_sizes)
    min_virt, max_virt = _safe_min_max(virtual_sizes)
    min_phys, max_phys = _safe_min_max(physical)
    min_vaddr, max_vaddr = _safe_min_max(virtual_addr)
    min_ptr, max_ptr = _safe_min_max(pointer_data)
    max_char = max(characteristics) if characteristics else 0               # largest flag value

    return {
        "SuspiciousNameSection": suspicious_names,
        "SectionsLength": len(sections),                                    # length of sections -> number of sections the file has
        "SectionMinEntropy": min_entropy, "SectionMaxEntropy": max_entropy,
        "SectionMinRawsize": min_raw, "SectionMaxRawsize": max_raw,
        "SectionMinVirtualsize": min_virt, "SectionMaxVirtualsize": max_virt,
        "SectionMaxPhysical": max_phys, "SectionMinPhysical": min_phys,
        "SectionMaxVirtual": max_vaddr, "SectionMinVirtual": min_vaddr,
        "SectionMaxPointerData": max_ptr, "SectionMinPointerData": min_ptr,
        "SectionMaxChar": max_char, "SectionMainChar": main_char,
    }


# Counts DLLs imported from, and how many imported function names match
# SUSPICIOUS_IMPORT_FUNCTIONS.
def _import_derived_fields(pe):
    if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):                           # if the pefile has no import table at all
        suspicious = 0
        nb_dll = 0
    else:
        suspicious = 0
        nb_dll = len(pe.DIRECTORY_ENTRY_IMPORT)                             # one entry per imported DLL
        for entry in pe.DIRECTORY_ENTRY_IMPORT:                             # loop through each imported DLL
            for imp in entry.imports:                                       # each function imported from that DLL
                name = imp.name
                if name is None:
                    continue                                                # if imports have no name, skip
                if isinstance(name, bytes):                                 
                    name = name.decode("latin-1", errors="replace")         # decode the bytes into text
                if name.lower() in SUSPICIOUS_IMPORT_FUNCTIONS:
                    suspicious += 1                                         # if it matches the suspicious import function list, count it as a signal

    import_size = getattr(_data_directory(pe, DIR_IMPORT), "Size", 0) or 0  # size in bytes of the whole import table, 0 if missing

    return {
        "SuspiciousImportFunctions": suspicious,
        "DirectoryEntryImport": nb_dll,
        "DirectoryEntryImportSize": import_size,
    }


# Counts exported functions. DLLs commonly export many; EXEs usually none.
def _export_derived_fields(pe):
    if not hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        return {"DirectoryEntryExport": 0}                                  # no export table, 0 exported functions
    return {"DirectoryEntryExport": len(pe.DIRECTORY_ENTRY_EXPORT.symbols)} # counts how many functions this file exports


# Collects the RVA of each of the 5 tracked data directories (see
# _directory_rva above for why these are addresses, not flags).
def _directory_rva_fields(pe):
    return {                                                                # each line looks up one directory's memory address (RVA)
        "ImageDirectoryEntryExport": _directory_rva(pe, DIR_EXPORT),
        "ImageDirectoryEntryImport": _directory_rva(pe, DIR_IMPORT),
        "ImageDirectoryEntryResource": _directory_rva(pe, DIR_RESOURCE),
        "ImageDirectoryEntryException": _directory_rva(pe, DIR_EXCEPTION),
        "ImageDirectoryEntrySecurity": _directory_rva(pe, DIR_SECURITY),
    }


# Computes every feature in feature_order from an already-parsed PE object.
# Kept separate from extract_pe_features so tests can use a fake object,
# no real PE file or pefile needed, see tests/test_extract_features.py.
def _features_from_pe(pe, feature_order=ORDER_OF_FEATURES):
    values = {}
    values.update(_dos_header_fields(pe))                                   # add the 17 DOS fields
    values.update(_file_header_fields(pe))                                  # add the 7 File header fields
    values.update(_optional_header_fields(pe))                              # add the 28 Optional header fields
    values.update(_section_derived_fields(pe))                              # add the 15 section-based features
    values.update(_import_derived_fields(pe))                               # add the 3 import-based features
    values.update(_export_derived_fields(pe))                               # add the 1 export-based feature
    values.update(_directory_rva_fields(pe))                                # add the 5 directory RVA features

    return [values[name] for name in feature_order]                         # reorder into a plain list matching feature_order, the order training used


# Parses raw file bytes as a PE file, returns a feature row. Never executes
# the file, only parses its structure. Raises an exception if file_bytes
# isn't a valid PE file, callers must catch this (see app.py).
def extract_pe_features(file_bytes, feature_order=ORDER_OF_FEATURES):
    
    import pefile                                                           # import pefile here to ensure that the module works without pefile, installed unless this specific function is called

    # fast_load skips auto-parsing every directory up front (faster, avoids
    # crashing on malformed directories this project doesn't need).
    pe = pefile.PE(data=file_bytes, fast_load=True)                         # parse the raw bytes into a PE object

    try:
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"],
        ])

    except Exception:
        pass
    return _features_from_pe(pe, feature_order)
