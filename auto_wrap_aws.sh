#!/usr/bin/env bash
set -euo pipefail

# AUTO-WRAP AWS USAGES WITH #ifndef DISABLE_AWS
# - Creates backups: <file>.awswrap.bak
# - Only wraps single lines containing selected AWS patterns (conservative)
# - Does NOT attempt to parse C++; it is a line-based heuristic
# - Review changes with `git diff` or by inspecting the .awswrap.bak files

ROOT_DIR="${1:-.}"

# file extensions to process
EXTENSIONS="cc cpp cxx h hh hpp c"

# patterns to look for (basic; can be extended)
PATTERNS=(
  "s3_client"
  "Aws::S3::"
  "Aws::S3Client"
  "Aws::S3"
  "Aws::S3::Model::"
  "Aws::"
)

# join patterns into a grep-compatible regex (word-safe)
GREP_PATTERN="$(IFS="|"; echo "${PATTERNS[*]}")"

echo "Auto-wrapping AWS lines under ${ROOT_DIR}"
echo "Patterns: ${PATTERNS[*]}"
echo

# find files under root directory
find "${ROOT_DIR}" -type f \( \
$(for e in $EXTENSIONS; do printf -- "-iname '*.%s' -o " "$e"; done | sed 's/ -o $//') \
\) | while read -r file; do
  # skip backup files produced earlier
  if [[ "${file}" == *.awswrap.bak ]]; then
    continue
  fi

  # quick test: does the file contain any of the patterns?
  if ! grep -E --line-number -q "${GREP_PATTERN}" "$file"; then
    continue
  fi

  echo "Processing: $file"

  # create a safe backup
  cp --preserve=mode,timestamps "$file" "${file}.awswrap.bak"

  # Build a new temporary file
  tmp="$(mktemp)"
  awk -v pat="${GREP_PATTERN}" '
  BEGIN {
    IGNORECASE=0
    # We will track a small lookback of previous non-empty lines to avoid double wrapping
  }
  {
    line = $0
    # print_line flag - 0 means we will print later, 1 means we already printed
    printed = 0

    # detect whether current line is already inside a DISABLE_AWS guard by checking
    # nearby lines (naive: examine current and previous 3 lines for DISABLE_AWS macros)
    # For robust detection we should scan backward — we store prev lines in array prev_lines
    prev_lines[NR] = line
    if (line ~ pat) {
      # check simple guard patterns in the last 10 lines (including current)
      guarded = 0
      for (i = NR; i >= NR-10 && i in prev_lines; --i) {
        pl = prev_lines[i]
        if (pl ~ /#\s*ifndef\s+DISABLE_AWS/ || pl ~ /#\s*ifdef\s+DISABLE_AWS/ || pl ~ /#\s*if\s+defined\s*\(\s*DISABLE_AWS\s*\)/) {
          guarded = 1
          break
        }
        # stop scanning if we hit #endif (assume we left guard scope)
        if (pl ~ /#\s*endif/) {
          break
        }
      }
      # also check the next 6 lines (we read ahead via getline, then unget)
      # read ahead into array ahead_lines
      ahead_cnt = 0
      while (ahead_cnt < 6) {
        if ((getline ahead_line) <= 0) break
        ahead[++ahead_cnt] = ahead_line
      }
      # check ahead for #endif or guard start
      for (i = 1; i <= ahead_cnt; ++i) {
        if (ahead[i] ~ /#\s*endif/ || ahead[i] ~ /#\s*ifndef\s+DISABLE_AWS/ || ahead[i] ~ /#\s*ifdef\s+DISABLE_AWS/) {
          guarded = 1
          break
        }
      }
      # print the current context back (we consumed ahead lines)
      # first, if not guarded, insert the guard start
      if (!guarded) {
        print "#ifndef DISABLE_AWS" > "'"${tmp}"'"
        print line > "'"${tmp}"'"
        print "#endif  // DISABLE_AWS" > "'"${tmp}"'"
      } else {
        # guarded: just print original line
        print line > "'"${tmp}"'"
      }
      # print ahead lines unchanged
      for (i = 1; i <= ahead_cnt; ++i) {
        print ahead[i] > "'"${tmp}"'"
      }
      # clear ahead array
      for (i = 1; i <= ahead_cnt; ++i) delete ahead[i]
      next
    }
    # no pattern -> print original
    print line > "'"${tmp}"'"
  }
  END {
  }' "$file"

  # move tmp back to file
  mv "$tmp" "$file"
done

echo
echo "Done. Backup copies saved as <file>.awswrap.bak"
echo "Run 'git diff' to inspect changes, then run your build."
echo
echo "If build failures remain, search for 'Aws::' or 's3_client' usages and I'll help craft precise fallback stubs."
