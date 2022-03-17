[Back to home.](../README.md)

# Get Started: Running the Analysis

These python scripts use locally-referenced data files that encode the ML
models that are being applied, so they should be run in-place
from a clone of this repo.  They rely on several python
dependencies, [described here](developer.md).

## Scoring DEXA images using a directory

The scoring of DISH from either a directory of images or a text file that lists the
image files to be scored, one per line:

```shell
python scoreSpines.py -i ${INPUT_DIR} -o ${RESULT_FILE} [--aug_flip/--aug_one]
```

**Input/output arguments:**
- `-i INPUT_DIR`: a directory of source images.  The file names, minus appends, will be
  used as the identifiers in the output file.  Corrupted or non-image files and
  sub-directories will be skipped over.
- `-o RESULT_FILE`: a two-column, tab-delimited file with DEXA photo ID's in the left
  column and predicted DISH scores in the right.  Specifying "stdout" will result in
  results being written to standard out.
- `--details DETAILS`: an optional addition to the main calling paradigm (`-i, -o`) that
  will output a second file containing per-bridge scoring data.  For each image, 14 lines
  of five-column data will be output: image identity (as in `-o` output), bridge instance
  number, per-bridge DISH score, y-axis position (how far down the image, as a fraction of
  image height), x-axis position (how far to the right in the image, as a fraction of image
  width).  Specifying "stdout" will result in results being written to standard out.

**Augmentation option arguments:** 
- `--aug_flip`: will flip each image horizontally and repeat the analysis, and output the
  average of the flipped and non-flipped DISH scores.  If `--details` was invoked, those
  bridges will also be reported (28 total), with position always reported in terms of the
  original (non-flipped) image.
- `--aug_one`: downgrades scores of 1 by replacing with the ratio of confidence scores <1 vs >1.
  Improves probabilistic accuracy for borderline bridges.


## Evaluating performance versus pre-scored DEXA images

An easy way to get statistics quickly for performance versus a small set of 
your own annotations:

```shell
python scoreSpines.py -a ${ANNOT_FILE} [--aug_flip/--aug_one]
```
**Invoking argument:**
- `-a ANNOT_FILE`: an alternative calling method: input a text file with two tab-separated
  columns (left: file name path; right: annotated DISH score), and this script will
  apply the DISH scoring algorithm to those images and print a statistical analysis
  of performance versus the annotations to standard out.
- Augmentation arguements apply as before.
