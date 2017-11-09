#!/bin/sh

set -x

# create ChangeLog from log of git repo
if test ! -e ChangeLog; then
        git log --pretty=format:'%h - %an: %s' > ChangeLog
else
        if git log --pretty=format:'%h - %an: %s' > ChangeLog.t \
           && diff ChangeLog.t ChangeLog >/dev/null 2>&1; then
                mv -f ChangeLog.t ChangeLog
        else
                rm -f ChangeLog.t
        fi
fi

mkdir -p build-aux
autoreconf -v -i -I build-aux -I m4

