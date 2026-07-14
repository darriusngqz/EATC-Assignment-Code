"""Unit tests for src/extract_features.py.

These tests deliberately avoid needing the pefile library or a real PE
file: _features_from_pe() is tested directly against small hand-built fake
objects that expose only the attributes it reads. This keeps the tests
fast, self-contained, and runnable in any environment, including CI
without pefile installed.
"""
from types import SimpleNamespace

import pytest

from constants import ORDER_OF_FEATURES
from extract_features import (
    _directory_rva_fields,
    _export_derived_fields,
    _features_from_pe,
    _import_derived_fields,
    _section_derived_fields,
)


def make_data_directory(entries, rvas=None):
    """entries: dict of {IMAGE_DIRECTORY_ENTRY_NAME: size}.
    rvas: optional dict of {IMAGE_DIRECTORY_ENTRY_NAME: VirtualAddress}."""
    rvas = rvas or {}
    return [
        SimpleNamespace(name=name, Size=size, VirtualAddress=rvas.get(name, 0))
        for name, size in entries.items()
    ]


def make_fake_pe(sections=None, data_directory=None):
    dh = SimpleNamespace(
        e_magic=23117, e_cblp=144, e_cp=3, e_crlc=0, e_cparhdr=4, e_minalloc=0,
        e_maxalloc=65535, e_ss=0, e_sp=184, e_csum=0, e_ip=0, e_cs=0,
        e_lfarlc=64, e_ovno=0, e_oemid=0, e_oeminfo=0, e_lfanew=128,
    )
    fh = SimpleNamespace(
        Machine=0x8664, NumberOfSections=len(sections or []), TimeDateStamp=1600000000,
        PointerToSymbolTable=0, NumberOfSymbols=0, SizeOfOptionalHeader=240, Characteristics=0x22,
    )
    oh = SimpleNamespace(
        Magic=0x20B, MajorLinkerVersion=14, MinorLinkerVersion=0, SizeOfCode=4096,
        SizeOfInitializedData=8192, SizeOfUninitializedData=0, AddressOfEntryPoint=0x1000,
        BaseOfCode=0x1000, ImageBase=0x140000000, SectionAlignment=4096, FileAlignment=512,
        MajorOperatingSystemVersion=6, MinorOperatingSystemVersion=0, MajorImageVersion=0,
        MinorImageVersion=0, MajorSubsystemVersion=6, MinorSubsystemVersion=0,
        SizeOfHeaders=1024, CheckSum=0, SizeOfImage=20480, Subsystem=3,
        DllCharacteristics=0x8160, SizeOfStackReserve=1048576, SizeOfStackCommit=4096,
        SizeOfHeapReserve=1048576, SizeOfHeapCommit=4096, LoaderFlags=0, NumberOfRvaAndSizes=16,
        DATA_DIRECTORY=data_directory or [],
    )
    return SimpleNamespace(DOS_HEADER=dh, FILE_HEADER=fh, OPTIONAL_HEADER=oh, sections=sections or [])


class FakeSection:
    def __init__(self, name, entropy, raw, virt, phys=None, vaddr=0, ptr=0, char=0):
        self.Name = name.encode() + b"\x00" * (8 - len(name))
        self._entropy = entropy
        self.SizeOfRawData = raw
        self.Misc_VirtualSize = virt
        self.Misc_PhysicalAddress = phys if phys is not None else virt
        self.VirtualAddress = vaddr
        self.PointerToRawData = ptr
        self.Characteristics = char

    def get_entropy(self):
        return self._entropy


def test_features_from_pe_returns_correct_length_and_order():
    pe = make_fake_pe(sections=[FakeSection(".text", 7.8, 4096, 4096)])
    row = _features_from_pe(pe, ORDER_OF_FEATURES)
    assert len(row) == len(ORDER_OF_FEATURES) == 77


def test_section_derived_fields_computes_min_max_stats():
    sections = [
        FakeSection(".text", 7.8, 100, 200, phys=200, vaddr=0x1000, ptr=0x400, char=0x60),
        FakeSection(".data", 2.2, 300, 400, phys=400, vaddr=0x2000, ptr=0x800, char=0xC0),
    ]
    stats = _section_derived_fields(make_fake_pe(sections=sections))
    assert stats["SectionsLength"] == 2
    assert stats["SectionMaxEntropy"] == 7.8
    assert stats["SectionMinEntropy"] == 2.2
    assert stats["SectionMinRawsize"] == 100
    assert stats["SectionMaxRawsize"] == 300
    assert stats["SuspiciousNameSection"] == 0  # both are standard names
    # "main" section = largest raw size = .data (300), so SectionMainChar
    # should be .data's Characteristics (0xC0), not .text's.
    assert stats["SectionMainChar"] == 0xC0
    assert stats["SectionMaxChar"] == 0xC0


def test_section_derived_fields_flags_nonstandard_section_names():
    sections = [FakeSection(".text", 5.0, 100, 100), FakeSection("UPX0", 7.9, 50, 50)]
    stats = _section_derived_fields(make_fake_pe(sections=sections))
    assert stats["SuspiciousNameSection"] == 1


def test_section_derived_fields_handles_no_sections():
    stats = _section_derived_fields(make_fake_pe(sections=[]))
    assert stats["SectionsLength"] == 0
    assert stats["SectionMinEntropy"] == 0
    assert stats["SectionMaxEntropy"] == 0


def test_import_derived_fields_counts_dlls_and_suspicious_functions():
    class FakeImport:
        def __init__(self, name):
            self.name = name

    class FakeEntry:
        def __init__(self, imports):
            self.imports = imports

    pe = make_fake_pe(data_directory=make_data_directory({"IMAGE_DIRECTORY_ENTRY_IMPORT": 320}))
    pe.DIRECTORY_ENTRY_IMPORT = [
        FakeEntry([FakeImport(b"CreateFileW"), FakeImport(b"VirtualAlloc"), FakeImport(None)]),
        FakeEntry([FakeImport(b"WriteProcessMemory")]),
    ]
    stats = _import_derived_fields(pe)
    assert stats["DirectoryEntryImport"] == 2
    assert stats["DirectoryEntryImportSize"] == 320
    # VirtualAlloc and WriteProcessMemory are on the suspicious list, CreateFileW is not.
    assert stats["SuspiciousImportFunctions"] == 2


def test_import_derived_fields_handles_no_import_directory():
    stats = _import_derived_fields(make_fake_pe())
    assert stats == {"SuspiciousImportFunctions": 0, "DirectoryEntryImport": 0, "DirectoryEntryImportSize": 0}


def test_export_derived_fields_counts_symbols():
    pe = make_fake_pe()
    pe.DIRECTORY_ENTRY_EXPORT = SimpleNamespace(symbols=[1, 2, 3, 4])
    assert _export_derived_fields(pe) == {"DirectoryEntryExport": 4}


def test_export_derived_fields_handles_no_export_directory():
    assert _export_derived_fields(make_fake_pe()) == {"DirectoryEntryExport": 0}


def test_directory_rva_fields_reads_virtual_address():
    # Regression test: data/dataset_malwares.csv showed these columns are
    # NOT 0/1 presence flags (an earlier, wrong assumption) but the raw
    # VirtualAddress (RVA), with values running into the hundreds of
    # millions for real files. See extract_features.py _directory_rva().
    dd = make_data_directory(
        {"IMAGE_DIRECTORY_ENTRY_EXPORT": 0, "IMAGE_DIRECTORY_ENTRY_IMPORT": 320,
         "IMAGE_DIRECTORY_ENTRY_RESOURCE": 0, "IMAGE_DIRECTORY_ENTRY_EXCEPTION": 0,
         "IMAGE_DIRECTORY_ENTRY_SECURITY": 1024},
        rvas={"IMAGE_DIRECTORY_ENTRY_IMPORT": 8364, "IMAGE_DIRECTORY_ENTRY_SECURITY": 226816},
    )
    pe = make_fake_pe(data_directory=dd)
    flags = _directory_rva_fields(pe)
    assert flags == {
        "ImageDirectoryEntryExport": 0,
        "ImageDirectoryEntryImport": 8364,
        "ImageDirectoryEntryResource": 0,
        "ImageDirectoryEntryException": 0,
        "ImageDirectoryEntrySecurity": 226816,
    }


def test_directory_rva_fields_handles_missing_entries():
    # Optional header exposes no DATA_DIRECTORY entries at all, a
    # legitimate case for a minimal/stripped executable.
    flags = _directory_rva_fields(make_fake_pe(data_directory=[]))
    assert all(v == 0 for v in flags.values())


def test_extract_pe_features_rejects_non_pe_bytes():
    pefile = pytest.importorskip("pefile", reason="pefile not installed in this environment")
    from extract_features import extract_pe_features

    with pytest.raises(pefile.PEFormatError):
        extract_pe_features(b"not a real PE file at all")
