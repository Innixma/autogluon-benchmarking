#!/usr/bin/env bash
HERE=$(dirname "$0")
if [[ -x "$(command -v apt-get)" ]]; then
    sudo apt-get update
    sudo apt-get install -y wget unzip openjdk-8-jdk
fi

AUTOWEKA_ARCHIVE="autoweka-2.6.zip"
DOWNLOAD_DIR="$HERE/lib"
TARGET_DIR="$DOWNLOAD_DIR/autoweka"
if [[ ! -e "$TARGET_DIR" ]]; then
    wget http://www.cs.ubc.ca/labs/beta/Projects/autoweka/$AUTOWEKA_ARCHIVE -P $DOWNLOAD_DIR
    unzip $DOWNLOAD_DIR/$AUTOWEKA_ARCHIVE -d $TARGET_DIR
fi
