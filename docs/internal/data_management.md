# Data Management

## Useful Commands

Ensure no duplicated data exists
`fdupes -r data/training data/validation data/testing data/raw_data data/raw_data_validation data/raw_data_testing`

Remove all metadata from JPEG files
`exiftool -all:all= -overwrite_original -ext jpeg .`

Convert the training directory into webp format and resize to max image size
`find training/ -type f -name '*.jpeg' -print0 | parallel -0 'mkdir -p "training_webp/$(dirname {})" && mogrify -format webp -resize "1048576@>" -path "training_webp/$(dirname {})" {}' \;`

Count files in all sub directories according to file extensions
`find . -type f -name '*.*' -not -name '.*' | sed -Ee 's,.*/.+\.([^/]+)$,\1,' | sort | uniq -ci | sort -n`

Generate simple table of contents for a markdown file
`cat training.md | grep -E "^### " | sed -E 's/(#+) (.+)/\1:\2:\2/g' | awk -F ":" '{ gsub(/#/,"  ",$1); gsub(/[ ]/,"-",$3); print $1 "- [" $2 "](#" tolower($3) ")" }'`

Install current venv as an ipykernel
`python -m ipykernel install --user --name birder`

Count code lines of the project
`cloc --fullpath --not-match-d='data/' --exclude-dir=.mypy_cache,.venv .`

Examine the annotations status
`cat data/annotations_status.csv | column -t -s, --table-noextreme 8`

Remove all json files in all tar files
`for f in *.tar; do tar -vf "$f" --wildcards --delete "*.json"; done`

Check for duplicated rows in a class file
`cat data/il-all_classes.txt | sort | uniq -cd`

List missing classes in training directory
`comm -23 <(sort data/eu-common_classes.txt) <(ls data/training | sort)`

## Useful Websites

<https://www.israbirding.com/checklist/>

<https://www.birds.org.il/he/species-families>

<https://en.wikipedia.org/wiki/Biogeographic_realm>

<https://digdeep1962.wordpress.com/2014/03/14/keep-calm-and-study-snipes-part-1/>

## Detection Annotations

`labelme --labels ../birder/data/detection_data/classes.txt --nodata --output ../birder/data/detection_data/training_annotations --flags unknown ../birder/data/detection_data/training`

`labelme --labels ../birder/data/detection_data/classes.txt --nodata --output ../birder/data/detection_data/validation_annotations --flags unknown ../birder/data/detection_data/validation`
