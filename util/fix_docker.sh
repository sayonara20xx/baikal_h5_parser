#!/bin/sh

c_name=baikal_gen_auto
sudo docker container stop ${c_name}
sudo docker container rm ${c_name}
