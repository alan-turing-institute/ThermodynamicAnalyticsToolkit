#!/bin/sh

set -x

# create ChangeLog from log of git repo
if test ! -e ChangeLog; then
        git log > ChangeLog
else
        if git log > ChangeLog.t \
           && diff ChangeLog.t ChangeLog >/dev/null 2>&1; then
                mv -f ChangeLog.t ChangeLog
        else
                rm -f ChangeLog.t
        fi
fi


autoreconf -v -i -I build-aux -I m4

