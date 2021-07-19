

scenario=( "badWeather" "badWeather" "badWeather" "badWeather" "baseline" "baseline" "baseline" "baseline" "cameraJitter" "cameraJitter" "cameraJitter" "cameraJitter" "dynamicBackground" "dynamicBackground" "dynamicBackground" "dynamicBackground" "dynamicBackground" "dynamicBackground" "intermittentObjectMotion" "intermittentObjectMotion" "intermittentObjectMotion" "intermittentObjectMotion" "intermittentObjectMotion" "lowFramerate" "lowFramerate" "lowFramerate" "lowFramerate" "nightVideos" "nightVideos" "nightVideos" "nightVideos" "nightVideos" "nightVideos" "shadow" "shadow" "shadow" "shadow" "shadow" "shadow" "thermal" "thermal" "thermal" "thermal" "thermal" "turbulence" "turbulence" "turbulence" "turbulence" )

sequence=( "blizzard" "skating" "snowFall" "wetSnow" "highway" "office" "pedestrians" "PETS2006" "badminton" "boulevard" "sidewalk" "traffic" "boats" "canoe" "fall" "fountain01" "fountain02" "overpass" "abandonedBox" "parking" "sofa" "tramstop" "winterDriveway" "port_0_17fps" "tramCrossroad_1fps" "tunnelExit_0_35fps" "turnpike_0_5fps" "bridgeEntry" "busyBoulvard" "fluidHighway" "streetCornerAtNight" "tramStation" "winterStreet" "backdoor" "bungalows" "busStation" "copyMachine" "cubicle" "peopleInShade" "corridor" "diningRoom" "lakeSide" "library" "park" "turbulence0" "turbulence1" "turbulence2" "turbulence3" )



#ls -F badWeather/blizzard/ |grep -v / | wc -l

#${batchsizes[$i]}


for i in "${!scenario[@]}"
do
	echo "${scenario[$i]}/${sequence[$i]}: "
	ls -F "${scenario[$i]}/${sequence[$i]}/" |grep -v / | wc -l

done