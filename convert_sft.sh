#!/bin/bash
# Convert WTForacle JSONL to SFT plain text for l.c/moe.c
# Input format: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
# Output format: <user>question<end><assistant>answer<end>

INPUT="${1:-/Users/ataeff/Downloads/WTForacle/wtf2/wtforacle_identity_v3_final.jsonl}"
OUTPUT="${2:-personality_sft.txt}"

> "$OUTPUT"
count=0

while IFS= read -r line; do
    # Extract user content: between first "content": " and next "}
    user=$(echo "$line" | sed -n 's/.*"role": *"user"[^}]*"content": *"\([^"]*\)".*/\1/p')
    # Extract assistant content
    asst=$(echo "$line" | sed -n 's/.*"role": *"assistant"[^}]*"content": *"\([^"]*\)".*/\1/p')

    if [ -n "$user" ] && [ -n "$asst" ]; then
        # Unescape \n
        user=$(echo "$user" | sed 's/\\n/\n/g')
        asst=$(echo "$asst" | sed 's/\\n/\n/g')
        printf '<user>%s<end><assistant>%s<end>\n' "$user" "$asst" >> "$OUTPUT"
        count=$((count + 1))
    fi
done < "$INPUT"

bytes=$(wc -c < "$OUTPUT")
echo "[sft] converted $count pairs, $((bytes/1024))KB -> $OUTPUT"
