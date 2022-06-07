#!/bin/sh

sudo docker run --name baikal_gen_auto -d -it git.jinr.ru:5005/baikal/generators:1.1 bash

echo "copying config files inside container:"
sudo docker cp ./gen_1.1/ligen.mac baikal_gen_auto:/generators/packages/chains/ligen.mac
sudo docker cp ./gen_1.1/h5writer.cfg baikal_gen_auto:/generators/packages/chains/configs/h5writer.cfg

echo "currently used following '.mac' file:"
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/chains; cat ligen.mac'
echo "currently used following 'h5writer.cfg' file:"
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/chains/configs; cat h5writer.cfg'

echo "start generating events..."
sudo docker exec -it baikal_gen_auto /bin/bash -c 'source generators_env.sh; cd packages/chains; python3 simGVD.py --primary_config=configs/g4particlePrimary.cfg --g4particle_macro=ligen.mac --n_events=50;'
sudo docker cp baikal_gen_auto:/generators/packages/chains/h5_output/events.h5 ./h5_coll/auto_events_${1}.h5
sudo docker container stop baikal_gen_auto
sudo docker container rm baikal_gen_auto
