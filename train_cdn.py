from train.bg_trainer import BackgroundTrainer

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

# cdnet_data = {
#     "badWeather"                : ["skating", "blizzard"]
# }

if __name__ == "__main__":

    # Path to config file
    config_file = 'config/config.json'

    for scenario in cdnet_data:
        for sequence in cdnet_data[scenario]:
            
            print(f"----- Starting to train CDN with scenario {scenario} use squence {sequence} -----")

            # Init Background trainer
            bg_trainer = BackgroundTrainer(config_file=config_file, 
                                           scenario_name=scenario, 
                                           sequence_name=sequence)
            # Execute the trainer
            bg_trainer.train()

            # Demo background
            # bg_trainer.demo_background()

