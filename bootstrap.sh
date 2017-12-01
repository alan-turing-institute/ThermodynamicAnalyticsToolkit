#!/bin/sh

set -x

# create ChangeLog from log of git repo
if test ! -e ChangeLog; then
        git log --pretty=format:'%h - %an: %s' > ChangeLog
else
        if git log --pretty=format:'%h - %an: %s' > ChangeLog.t \
           && diff ChangeLog.t ChangeLog >/dev/null 2>&1; then
                rm -f ChangeLog.t
        else
                mv -f ChangeLog.t ChangeLog
        fi
fi

mkdir -p build-aux
autoreconf -v -i -I build-aux -I m4

