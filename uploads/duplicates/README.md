# Duplicate Videos Folder

This folder contains videos that have been identified as duplicates by the AI system.

## Structure
- Videos are moved here when they are detected as duplicates of existing videos
- The original (best quality/longest) video remains in the main uploads folder
- Duplicate filenames preserve the original upload filename for reference

## Duplicate Detection Process
1. Frame-by-frame comparison using perceptual hashing
2. Content similarity analysis
3. Duration and quality comparison
4. Automatic selection of the best version as the canonical video