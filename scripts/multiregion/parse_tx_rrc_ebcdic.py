#!/usr/bin/env python3
"""Parse Texas RRC PDF101 EBCDIC production data into canonical CSV.

The PDF101 file is a mainframe hierarchical database dump with:
- Fixed-width records
- EBCDIC character encoding (cp037)
- Packed decimal (COMP-3) numeric fields
- Record types identified by first 2 bytes

Key record types for gas tape:
  01 = PDROOT      (Gas Root: district, lease/well ID)
  14 = PDGRPTCY    (Reporting Cycle: year-month cycle key)
  15 = PDGPROD     (Production: gas MCF, condensate barrels)
"""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path


def decode_ebcdic(data: bytes) -> str:
    """Decode EBCDIC bytes to ASCII string."""
    return data.decode("cp037", errors="replace").strip()


def unpack_comp3(data: bytes) -> int:
    """Unpack a COMP-3 (packed BCD) field to integer.

    Each byte holds two BCD digits (high nibble, low nibble), except
    the last byte where the low nibble is the sign (0xC=positive, 0xD=negative, 0xF=unsigned).
    """
    if not data:
        return 0
    digits = []
    for i, byte in enumerate(data):
        high = (byte >> 4) & 0x0F
        low = byte & 0x0F
        if i < len(data) - 1:
            digits.append(high)
            digits.append(low)
        else:
            # Last byte: high nibble is digit, low nibble is sign
            digits.append(high)
            sign = low

    value = 0
    for d in digits:
        if d > 9:
            d = 0  # Invalid BCD digit
        value = value * 10 + d

    if len(data) > 0 and (data[-1] & 0x0F) == 0x0D:
        value = -value

    return value


def parse_gas_root(record: bytes) -> dict:
    """Parse Gas Root Segment (key=01, 50 bytes).

    Pos 1-2: Record ID "01"
    Pos 3:   PD-GAS-CODE      PIC X(1)  = 'G'
    Pos 4-5: PD-GAS-DISTRICT  PIC 9(2)
    Pos 6-11: PD-GAS-RRC-ID   PIC 9(6)
    """
    gas_code = decode_ebcdic(record[2:3])
    district = decode_ebcdic(record[3:5])
    rrc_id = decode_ebcdic(record[5:11])
    return {
        "gas_code": gas_code,
        "district": district,
        "rrc_id": rrc_id,
    }


def parse_gas_cycle(record: bytes) -> dict:
    """Parse Gas Reporting Cycle Segment (key=14, 90 bytes).

    Pos 1-2: Record ID "14"
    Pos 3-6: PD-GAS-REPORT-CYCLE-KEY  PIC 9(4) = YYMM
    """
    cycle_key = decode_ebcdic(record[2:6])
    return {"cycle_key": cycle_key}


def parse_gas_production(record: bytes) -> dict:
    """Parse Gas Production Report Segment (key=15, 50 bytes).

    Pos 1-2:  Record ID "15"
    Pos 3-7:  PD-GAS-PROD               PIC S9(9) COMP-3 (5 bytes) - gas MCF
    Pos 8-12: PD-GAS-LIFT-GAS-INJECTED  PIC S9(9) COMP-3 (5 bytes)
    Pos 13-17: PD-COND-PROD             PIC S9(9) COMP-3 (5 bytes) - condensate barrels
    """
    gas_prod = unpack_comp3(record[2:7])
    gas_lift = unpack_comp3(record[7:12])
    cond_prod = unpack_comp3(record[12:17])
    return {
        "gas_mcf": gas_prod,
        "gas_lift_mcf": gas_lift,
        "condensate_bbl": cond_prod,
    }


def parse_pdf101(input_path: Path, output_path: Path, max_records: int = 0) -> dict:
    """Parse the PDF101 EBCDIC file and write production CSV."""
    print(f"Reading {input_path} ({input_path.stat().st_size / 1024 / 1024:.0f} MB)...")

    RECORD_LENGTH = 102  # Fixed record length verified from hex analysis

    key_patterns = {
        b'\xf0\xf1': '01',  # Root
        b'\xf1\xf4': '14',  # Gas Reporting Cycle
        b'\xf1\xf5': '15',  # Gas Production
        b'\xf1\xf6': '16',  # Gas Disposition
        b'\xf1\xf7': '17',  # Condensate Disposition
        b'\xf1\xf8': '18',  # Commingle
        b'\xf1\xf9': '19',  # Balancing
        b'\xf2\xf0': '20',  # Reinstated
        b'\xf2\xf1': '21',  # Previous Prod
        b'\xf2\xf2': '22',  # Remarks
        b'\xf2\xf5': '25',  # Gas Disposition Remarks
        b'\xf2\xf6': '26',  # Condensate Disp Remarks
        b'\xf2\xf7': '27',  # Gas Separation/Extraction Loss
    }
    record_length = RECORD_LENGTH
    print(f"Record length: {record_length} bytes")

    # Now parse the full file
    file_size = input_path.stat().st_size
    total_records = file_size // record_length
    print(f"Total records: {total_records:,}")

    current_root = None
    current_cycle = None
    rows = []
    record_counts = {}

    with open(input_path, "rb") as f:
        for rec_num in range(total_records):
            record = f.read(record_length)
            if len(record) < record_length:
                break

            key = record[:2]
            key_str = key_patterns.get(key, decode_ebcdic(key))
            record_counts[key_str] = record_counts.get(key_str, 0) + 1

            if key == b'\xf0\xf1':  # Root segment
                current_root = parse_gas_root(record)
                current_cycle = None

            elif key == b'\xf1\xf4':  # Reporting cycle
                current_cycle = parse_gas_cycle(record)

            elif key == b'\xf1\xf5':  # Production
                if current_root and current_cycle:
                    prod = parse_gas_production(record)
                    cycle = current_cycle["cycle_key"]
                    # Cycle key is YYMM or CCYYMM
                    if len(cycle) == 4:
                        year = 2000 + int(cycle[:2]) if int(cycle[:2]) < 50 else 1900 + int(cycle[:2])
                        month = int(cycle[2:4])
                    elif len(cycle) == 6:
                        year = int(cycle[:4])
                        month = int(cycle[4:6])
                    else:
                        continue

                    if month < 1 or month > 12:
                        continue

                    rows.append({
                        "district": current_root["district"],
                        "rrc_id": current_root["rrc_id"],
                        "cycle_key": cycle,
                        "production_year": year,
                        "production_month": month,
                        "gas_mcf": prod["gas_mcf"],
                        "gas_lift_mcf": prod["gas_lift_mcf"],
                        "condensate_bbl": prod["condensate_bbl"],
                    })

            if max_records and rec_num >= max_records:
                break

            if rec_num > 0 and rec_num % 1_000_000 == 0:
                print(f"  Processed {rec_num:,} records, {len(rows):,} production rows...")

    print(f"\nRecord type counts:")
    for key, count in sorted(record_counts.items()):
        print(f"  {key}: {count:,}")

    print(f"\nProduction rows: {len(rows):,}")

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "district", "rrc_id", "cycle_key", "production_year", "production_month",
            "gas_mcf", "gas_lift_mcf", "condensate_bbl",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {output_path}")

    return {
        "total_records": total_records,
        "production_rows": len(rows),
        "record_counts": record_counts,
        "output_path": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse TX RRC PDF101 EBCDIC production file.")
    parser.add_argument("--input", default="/home/shin/Mineral_Gas_Locator/data/raw/tx_rrc/PDF101.ebc")
    parser.add_argument("--output", default="/home/shin/Mineral_Gas_Locator/data/canonical/tx_mvp/production.csv")
    parser.add_argument("--max-records", type=int, default=0, help="Stop after N records (0=all)")
    args = parser.parse_args()

    result = parse_pdf101(Path(args.input), Path(args.output), args.max_records)

    import json
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
