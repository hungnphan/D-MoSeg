from data_io.fg_data_generator import FgDataGenerator

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

if __name__ == '__main__':
    # Path to config file
    config_file = 'config/config.json'

    for scenario in cdnet_data:
        for sequence in cdnet_data[scenario]:
            if(scenario != "intermittentObjectMotion") or (sequence != "streetLight"):
                continue

            print(f"----- Generate foreground data for scenario `{scenario}` - sequence `{sequence}`` -----")

            # Init FgDataGenerator
            fg_generator = FgDataGenerator(config_file='config/config.json', 
                                           scenario_name=scenario, 
                                           sequence_name=sequence)
            # Execute the generator
            fg_generator.prepare_fg_training_data()
            # fg_generator.check_fg_training_data()
            fg_generator.export_fg_training_data()
