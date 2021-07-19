cdnet_data = {
    "badWeather"                : ["blizzard","skating","snowFall","wetSnow"],
    "baseline"                  : ["highway","office","pedestrians","PETS2006"],
    "cameraJitter"              : ["badminton","boulevard","sidewalk","traffic"],
    "dynamicBackground"         : ["boats","canoe","fall","fountain01","fountain02","overpass"],
    "intermittentObjectMotion"  : ["abandonedBox","parking","sofa","streetLight","tramstop","winterDriveway"],
    "lowFramerate"              : ["port_0_17fps","tramCrossroad_1fps","tunnelExit_0_35fps","turnpike_0_5fps"],
    "nightVideos"               : ["bridgeEntry","busyBoulvard","fluidHighway","streetCornerAtNight","tramStation","winterStreet"],
    "PTZ"                       : ["continuousPan","intermittentPan","twoPositionPTZCam","zoomInZoomOut"],
    "shadow"                    : ["backdoor","bungalows","busStation","copyMachine","cubicle","peopleInShade"],
    "thermal"                   : ["corridor","diningRoom","lakeSide","library","park"],
    "turbulence"                : ["turbulence0","turbulence1","turbulence2","turbulence3"]
}

listScen = []
listSequ = []

for scenario in cdnet_data:
    for sequence in cdnet_data[scenario]:
        if scenario == "PTZ" or sequence == "streetLight":
            continue
            
        listScen.append(scenario)
        listSequ.append(sequence)

print(len(listScen))

print("( ", end="")
for i in range(len(listScen)):
    print(f"\"{listScen[i]}\"", end=" ")
print(")")

print("( ", end="")
for i in range(len(listSequ)):
    print(f"\"{listSequ[i]}\"", end=" ")
print(")")
