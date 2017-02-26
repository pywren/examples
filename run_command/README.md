# `run_command.py` 

This example simply uses `subprocess` to invoke a command on the 
remote lambda. For example, to get a list of `/usr/`, run

```
python run_command.py ls /usr/
```
which produces

```
bin
etc
games
include
lib
lib64
libexec
local
sbin
share
src
tmp
```
