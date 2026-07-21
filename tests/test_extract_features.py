"""Unit tests for src/extract_features.py (20-feature schema).

Deliberately avoid needing the pefile library or a real PE file for most
tests: the private helper functions are tested directly against small
hand-built fake objects that expose only the attributes each one reads.
Keeps these fast, self-contained, and runnable in any environment,
including CI without pefile installed.
"""
import pytest

from constants import ORDER_OF_FEATURES
from extract_features import (
    _features_from_pe,
    _import_derived_fields,
    _safe_min_max,
    _section_derived_fields,
    _section_name,
    _size_fields,
    _string_derived_fields,
)


class FakeSection:
    def __init__(self, name, entropy, raw, virt):
        self.Name = name.encode() + b"\x00" * (8 - len(name))
        self._entropy = entropy
        self.SizeOfRawData = raw
        self.Misc_VirtualSize = virt

    def get_entropy(self):
        return self._entropy


class FakeOptionalHeader:
    def __init__(self, size_of_image):
        self.SizeOfImage = size_of_image


class FakePE:
    def __init__(self, sections=None, size_of_image=20480):
        self.sections = sections or []
        self.OPTIONAL_HEADER = FakeOptionalHeader(size_of_image)


class FakeImport:
    def __init__(self, name):
        self.name = name


class FakeImportEntry:
    def __init__(self, imports):
        self.imports = imports


def test_safe_min_max_handles_empty_list():
    assert _safe_min_max([]) == (0, 0)


def test_safe_min_max_returns_min_and_max():
    assert _safe_min_max([3, 1, 2]) == (1, 3)


def test_section_name_strips_null_padding():
    section = FakeSection(".text", 7.8, 4096, 4096)
    assert _section_name(section) == ".text"


def test_section_derived_fields_computes_min_max_stats():
    sections = [FakeSection(".text", 7.8, 100, 200), FakeSection("UPX0", 6.9, 300, 400)]
    stats = _section_derived_fields(FakePE(sections=sections))
    assert stats["SectionsLength"] == 2
    assert stats["SectionMinEntropy"] == 6.9
    assert stats["SectionMaxEntropy"] == 7.8
    assert stats["SectionMinRawsize"] == 100
    assert stats["SectionMaxRawsize"] == 300
    assert stats["SectionMinVirtualsize"] == 200
    assert stats["SectionMaxVirtualsize"] == 400
    # "UPX0" (a common packer section name) is not in STANDARD_SECTION_NAMES.
    assert stats["SuspiciousNameSection"] == 1


def test_section_derived_fields_handles_no_sections():
    stats = _section_derived_fields(FakePE(sections=[]))
    assert stats["SectionsLength"] == 0
    assert stats["SectionMinEntropy"] == 0
    assert stats["SectionMaxEntropy"] == 0


def test_import_derived_fields_handles_no_import_directory():
    stats = _import_derived_fields(FakePE())
    assert stats == {
        "SuspiciousImportFunctions": 0,
        "DirectoryEntryImport": 0,
        "NumberOfImportedFunctions": 0,
    }


def test_import_derived_fields_counts_dlls_functions_and_suspicious_hits():
    pe = FakePE()
    pe.DIRECTORY_ENTRY_IMPORT = [
        FakeImportEntry([FakeImport(b"CreateFileW"), FakeImport(b"VirtualAlloc"), FakeImport(None)]),
        FakeImportEntry([FakeImport(b"WriteProcessMemory")]),
    ]
    stats = _import_derived_fields(pe)
    assert stats["DirectoryEntryImport"] == 2  # two DLLs
    assert stats["NumberOfImportedFunctions"] == 4  # 3 + 1, including the None entry
    # VirtualAlloc and WriteProcessMemory are on the suspicious list, CreateFileW is not.
    assert stats["SuspiciousImportFunctions"] == 2


def test_string_derived_fields_detects_urls_paths_registry_and_mz():
    data = (
        b"hello world this is a test http://evil.com C:\\Windows\\System32 "
        b"HKEY_LOCAL_MACHINE MZ MZ junk"
    )
    stats = _string_derived_fields(data)
    assert stats["NumStrings"] >= 1
    assert stats["NumURLs"] == 1
    assert stats["NumPaths"] == 1
    assert stats["NumRegistryKeys"] == 1
    assert stats["NumMZStrings"] == 2


def test_string_derived_fields_handles_no_printable_runs():
    stats = _string_derived_fields(b"\x00\x01\x02")
    assert stats == {
        "NumStrings": 0, "StringEntropy": 0.0, "AvgStringLength": 0.0,
        "NumURLs": 0, "NumRegistryKeys": 0, "NumPaths": 0, "NumMZStrings": 0,
    }


def test_size_fields_reads_file_and_virtual_size():
    stats = _size_fields(FakePE(size_of_image=20480), b"x" * 5000)
    assert stats["FileSize"] == 5000
    assert stats["VirtualSize"] == 20480


def test_features_from_pe_returns_correct_length_and_order():
    pe = FakePE(sections=[FakeSection(".text", 7.8, 4096, 4096)])
    row = _features_from_pe(pe, b"x" * 5000, ORDER_OF_FEATURES)
    assert len(row) == len(ORDER_OF_FEATURES) == 20


def test_extract_pe_features_rejects_non_pe_bytes():
    pefile = pytest.importorskip("pefile", reason="pefile not installed in this environment")
    from extract_features import extract_pe_features

    with pytest.raises(pefile.PEFormatError):
        extract_pe_features(b"not a real PE file at all")
