import os
import glob

def main():
    # Directory where the merged file should go
    output_dir = os.path.join("..", "..", "kg")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "wikidata-cached.txt")

    # Collect all .txt files in the current directory
    txt_files = glob.glob("*.txt")

    # Make sure we don't re-read a previously generated merged file
    if "merged_kg.txt" in txt_files:
        txt_files.remove("merged_kg.txt")

    unique_lines = set()

    # Read all txt files and add their lines to the set
    for file in txt_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                clean = line.strip()
                if clean:
                    unique_lines.add(clean)

    # Write deduplicated, sorted list
    with open(output_file, "w", encoding="utf-8") as out:
        for line in sorted(unique_lines):
            out.write(line + "\n")

    print(f"Merged {len(txt_files)} files.")
    print(f"Deduplicated triples: {len(unique_lines)}")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    main()
