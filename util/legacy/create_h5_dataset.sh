#!/bin/sh

sudo docker run --name baikal_gen_auto -d -it git.jinr.ru:5005/baikal/generators:latest bash
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/control/verbose 1" > current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/tracking/verbose 0" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "#" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/LiGen/phys/verbose 0" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "#" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/LiGen/det/z_min 0 m" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/LiGen/det/z_max 1500 m" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/LiGen/det/rho 2000 m" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "#" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/gps/particle e-" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/gps/energy 1 GeV" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/gps/position 0 0 0 m" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/gps/pos/type Volume" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/gps/pos/shape Cylinder" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/gps/pos/centre 0 0 300 m" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/gps/pos/radius 2000 m" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/gps/pos/halfz 250 m" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/gps/ang/type iso" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "#" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/LiGen/output/store_photons 1" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/LiGen/output/store_histos 0" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/LiGen/output/photon_suppression_factor 1" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "#" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/run/initialize" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "#" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/run/printProgress 10" >> current_config.mac'
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; echo "/run/beamOn 200" >> current_config.mac'
echo "currently used following '.mac' file:"
sudo docker exec -it baikal_gen_auto /bin/bash -c 'cd packages/framework; cat current_config.mac'
echo "start generating events..."
sudo docker exec -it baikal_gen_auto /bin/bash -c 'source generators_env.sh; cd packages/framework; python3 ./loop.py --source LiGen --n_events 200 --ligen_macro ./current_config.mac'
sudo docker cp baikal_gen_auto:/generators/packages/framework/h5_output/events.h5 ./h5_coll/auto_events_${1}.h5
sudo docker container stop baikal_gen_auto
sudo docker container rm baikal_gen_auto
