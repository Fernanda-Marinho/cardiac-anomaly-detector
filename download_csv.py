#!/usr/bin/env python3
"""Download one public Google Drive CSV file by number.

Usage examples:
  python3 download_csv.py 100 --file-id 1aBcDeFgHiJkLmNoPqRsTuVwXyZ
  python3 download_csv.py 100 --id-map drive_ids.json

The script does not require Google authentication if the file is shared
publicly (Anyone with the link).
"""

import argparse
import json
import os
import re
import sys

import requests

DRIVE_DOWNLOAD_URL = "https://docs.google.com/uc?export=download"


def parse_google_drive_id(url_or_id):
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", url_or_id):
        return url_or_id

    patterns = [
        r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"https?://drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)",
        r"https?://drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot parse Google Drive ID from '{url_or_id}'")


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


def download_file_from_google_drive(file_id, destination):
    session = requests.Session()

    response = session.get(DRIVE_DOWNLOAD_URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        response = session.get(
            DRIVE_DOWNLOAD_URL,
            params={"id": file_id, "confirm": token},
            stream=True,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Download failed: status {response.status_code} for file id {file_id}"
        )

    save_response_content(response, destination)


def load_id_map(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_filename(number, prefix, suffix):
    return f"{prefix}{number}{suffix}"


def main():
    parser = argparse.ArgumentParser(
        description="Download a public Google Drive CSV file by number or download all mapped files."
    )
    parser.add_argument("number", help="File number (e.g. 100) or 'all' to download all mapped files")
    parser.add_argument(
        "--file-id",
        help="Google Drive file ID or full share link for the target CSV",
    )
    parser.add_argument(
        "--id-map",
        help="JSON file mapping numbers to Google Drive file IDs (default: drive_ids.json)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix for the local filename, default is ''",
    )
    parser.add_argument(
        "--suffix",
        default="_signals.csv",
        help="Suffix for the local filename, default is '_signals.csv'",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the file(s) will be saved",
    )
    args = parser.parse_args()

    # Determine id_map path (default to ./drive_ids.json when not provided)
    id_map_path = args.id_map or "drive_ids.json"
    id_map = {}
    if os.path.exists(id_map_path):
        try:
            id_map = load_id_map(id_map_path)
        except Exception:
            sys.exit(f"Failed to load id map: {id_map_path}")
    elif not args.file_id and args.number != "all":
        # if neither id_map nor file-id provided and number isn't 'all', error
        sys.exit(f"ID map not found: {id_map_path}. Provide --file-id or create {id_map_path}.")

    # Handle 'all' request
    if args.number == "all":
        if not id_map:
            sys.exit("No id map available to download all files.")
        os.makedirs(args.output_dir, exist_ok=True)
        downloaded = []
        for num, fid in id_map.items():
            try:
                fid_parsed = parse_google_drive_id(fid)
            except ValueError:
                print(f"Skipping invalid id for {num}: {fid}")
                continue
            filename = build_filename(num, args.prefix, args.suffix)
            destination = os.path.join(args.output_dir, filename)
            print(f"Downloading {filename} from {fid_parsed}...")
            try:
                download_file_from_google_drive(fid_parsed, destination)
                downloaded.append(destination)
            except Exception as exc:
                print(f"Failed {num}: {exc}")
        # combine into single CSV
        if downloaded:
            combined = os.path.join(args.output_dir, "all_signals_combined.csv")
            combine_csvs(downloaded, combined)
            print(f"Combined CSV saved to: {combined}")
        else:
            print("No files were downloaded.")
        return

    # Single file download
    file_id = None
    if args.file_id:
        file_id = parse_google_drive_id(args.file_id)
    else:
        # lookup in id_map
        file_id = id_map.get(args.number)
        if file_id is None:
            sys.exit(f"No file ID found for number '{args.number}' in {id_map_path}")
        try:
            file_id = parse_google_drive_id(file_id)
        except ValueError:
            sys.exit(f"Invalid file ID or link for number {args.number}: {file_id}")

    filename = build_filename(args.number, args.prefix, args.suffix)
    destination = os.path.join(args.output_dir, filename)

    print(f"Downloading {filename} from Google Drive file ID {file_id}...")
    try:
        download_file_from_google_drive(file_id, destination)
    except Exception as exc:
        sys.exit(f"Download failed: {exc}")

    print(f"Downloaded to: {destination}")


if __name__ == "__main__":
    main()
