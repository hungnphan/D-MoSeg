from data_io.fg_data_generator import FgDataGenerator
import torch

cdnet_data = {
    "badWeather"                : ["blizzard","skating","snowFall","wetSnow"],
    "baseline"                  : ["highway","office","pedestrians","PETS2006"]
}

if __name__ == '__main__':
    # Path to config file
    config_file = 'config/config.json'

    device = torch.device('cuda:0')

    for scenario in cdnet_data:
        for sequence in cdnet_data[scenario]:
            if(scenario == "turbulence") or (sequence == "streetLight"):
                continue

            print(f"----- Generate foreground data for scenario `{scenario}` - sequence `{sequence}`` -----")

            # Init FgDataGenerator
            fg_generator = FgDataGenerator(config_file='config/config.json', 
                                           scenario_name=scenario, 
                                           sequence_name=sequence,
                                           cuda_device=device)
            # Execute the generator
            fg_generator.prepare_fg_training_data()
            # fg_generator.check_fg_training_data()
            # fg_generator.export_fg_training_data()
