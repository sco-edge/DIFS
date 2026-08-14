#!/bin/bash
# Robust dump script with dynamic extensions
# Usage:
#   ./dump_codebase_v2.sh
#   ./dump_codebase_v2.sh py cpp h proto txt

OUTPUT_FILE="codebase_dump.txt"

> "$OUTPUT_FILE"

echo "=== DIRECTORY STRUCTURE ===" >> "$OUTPUT_FILE"
tree -a -I '__pycache__|*.pyc|node_modules|.git|.DS_Store' >> "$OUTPUT_FILE"
echo -e "\n\n=== FILE CONTENTS ===\n" >> "$OUTPUT_FILE"

#########################################
# Build extension filter safely
#########################################

FIND_EXT_ARGS=()

if [ "$#" -gt 0 ]; then
    FIND_EXT_ARGS+=("(")
    first=1
    for ext in "$@"; do
        clean_ext="${ext#.}"

        if [ $first -eq 0 ]; then
            FIND_EXT_ARGS+=("-o")
        fi

        FIND_EXT_ARGS+=("-name" "*.${clean_ext}")
        first=0
    done
    FIND_EXT_ARGS+=(")")
fi

#########################################
# Run find safely (NO eval)
#########################################

find . -type f \
    \( -not -path '*/\.*' \
       -a -not -path '*/__pycache__/*' \
       -a -not -path '*/node_modules/*' \) \
    "${FIND_EXT_ARGS[@]}" \
    -print0 | while IFS= read -r -d '' file; do

    if file "$file" | grep -qE "text|empty"; then
        echo "========BEGIN:$file========" >> "$OUTPUT_FILE"
        cat "$file" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "========END:$file========" >> "$OUTPUT_FILE"
        echo -e "\n" >> "$OUTPUT_FILE"
    fi

done

echo "Codebase dumped to $OUTPUT_FILE"
