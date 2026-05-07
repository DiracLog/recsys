#!/bin/bash

# run ./track_new_branch.sh force to also commit and push changes to Git/LFS
FORCE_PARAM=$1

TARGET=$(readlink artifacts/runs/latest)

if [ -z "$TARGET" ]; then
  echo "Error: Symlink 'artifacts/runs/latest' not found."
  exit 1
fi

TARGET=$(basename "$TARGET")

# 2. Update .gitignore (Find !artifacts/runs but not !artifacts/runs/latest)
awk -v target="$TARGET" '
  /^!artifacts\/runs/ && !/^!artifacts\/runs\/latest/ {
      print "!artifacts/runs/" target
      next
  }
  { print }
' .gitignore > .gitignore.tmp && mv .gitignore.tmp .gitignore

echo "Updated .gitignore for: $TARGET"

# if force, remove old LFS files, add new ones, commit and push
if [ "$FORCE_PARAM" == "force" ]; then
    echo "--- Force parameter detected. Starting Git LFS and Push operations ---"

    # Track the new directory in LFS
    git lfs track "artifacts/runs/$TARGET/**"
    
    # Prune old local LFS objects to save space
    git lfs prune

    # Stage changes
    git add .gitignore .gitattributes "artifacts/runs/$TARGET"
    
    # Commit and Push
    git commit -m "Auto-update: whitelisting and tracking run $TARGET"
    git push origin main
    
    echo "Done: Files pushed to LFS and repository."
else
    echo "--- Skipping Git push ---"
    echo "Run './script.sh force' to commit and push changes."
fi