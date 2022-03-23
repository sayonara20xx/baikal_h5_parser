all scripts from 'util' dir start in current active directoty
e.g. './util/create_h5_dataset.sh'
all relative paths are writed relying on highlighted circumstances

start your work with `./util/loop_h5_gen.sh` {number_of_files_you_want_get}
it will auto delete generated h5 files after data extracting with `create_dataset.py`

use `create_h5_dataset_save.sh` {number} to create only one h5 files with num at the end of filename
or write into loop script that script to generate numerous of h5 files

also, sh-script using docker with `sudo`, relying you add active user
in sudoers file, otherwise you will need to enter password periodically