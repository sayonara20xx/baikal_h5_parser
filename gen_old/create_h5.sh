#!/bin/sh

sudo docker run --name baikal_gen_auto -d -it git.jinr.ru:5005/baikal/generators:latest bash

echo "copying config files inside container:"
sudo docker cp ./util/current_conf.mac baikal_gen_auto:/generators/packages/framework/current_config.mac
sudo docker cp ./util/required_h5Writer_conf.conf baikal_gen_auto:/generators/packages/framework/configs/h5Writer.conf

echo "currently used following '.mac' file:"
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; cat current_config.mac'
echo "currently used following '.conf' h5Writer file:"
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework/configs; cat h5Writer.conf'

echo "start generating events..."
sudo docker exec -it baikal_gen_auto /bin/bash -c 'source generators_env.sh; cd packages/framework; python3 ./loop.py --source LiGen --n_events 200 --ligen_macro ./current_config.mac'
sudo docker cp baikal_gen_auto:/generators/packages/framework/h5_output/events.h5 ./h5_coll/auto_events_${1}.h5
sudo docker container stop baikal_gen_auto
sudo docker container rm baikal_gen_auto
