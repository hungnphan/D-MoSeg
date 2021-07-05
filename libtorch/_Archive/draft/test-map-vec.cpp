#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <stdint.h>

// Dictionary of dataset CDnet2014
// std::map <std::string, std::vector<std::string> > const cdnet_data;
// cdnet_data["badWeather"]               = std::vector<std::string>{"blizzard","skating","snowFall","wetSnow"};
// cdnet_data["baseline"]                 = std::vector<std::string>{"highway","office","pedestrians","PETS2006"};
// cdnet_data["cameraJitter"]             = std::vector<std::string>{"badminton","boulevard","sidewalk","traffic"};
// cdnet_data["dynamicBackground"]        = std::vector<std::string>{"boats","canoe","fall","fountain01","fountain02","overpass"};
// cdnet_data["intermittentObjectMotion"] = std::vector<std::string>{"abandonedBox","parking","sofa","tramstop","winterDriveway"};
// cdnet_data["lowFramerate"]             = std::vector<std::string>{"port_0_17fps","tramCrossroad_1fps","tunnelExit_0_35fps","turnpike_0_5fps"};
// cdnet_data["nightVideos"]              = std::vector<std::string>{"bridgeEntry","busyBoulvard","fluidHighway","streetCornerAtNight","tramStation","winterStreet"};
// cdnet_data["PTZ"]                      = std::vector<std::string>{"continuousPan","intermittentPan","twoPositionPTZCam","zoomInZoomOut"};
// cdnet_data["shadow"]                   = std::vector<std::string>{"backdoor","bungalows","busStation","copyMachine","cubicle","peopleInShade"};
// cdnet_data["thermal"]                  = std::vector<std::string>{"corridor","diningRoom","lakeSide","library","park"};
// cdnet_data["turbulence"]               = std::vector<std::string>{"turbulence0","turbulence1","turbulence2","turbulence3"};

std::map<std::string, std::vector<std::string>> const cdnet_data {
   { "badWeather",                  { "blizzard","skating","snowFall","wetSnow" } },
   { "baseline",                    { "highway","office","pedestrians","PETS2006" } },
   { "cameraJitter",                { "badminton","boulevard","sidewalk","traffic" } },
   { "dynamicBackground",           { "boats","canoe","fall","fountain01","fountain02","overpass" } },
   { "intermittentObjectMotion",    { "abandonedBox","parking","sofa","tramstop","winterDriveway" } },
   { "lowFramerate",                { "port_0_17fps","tramCrossroad_1fps","tunnelExit_0_35fps","turnpike_0_5fps" } },
   { "nightVideos",                 { "bridgeEntry","busyBoulvard","fluidHighway","streetCornerAtNight","tramStation","winterStreet" } },
   { "PTZ",                         { "continuousPan","intermittentPan","twoPositionPTZCam","zoomInZoomOut" } },
   { "shadow",                      { "backdoor","bungalows","busStation","copyMachine","cubicle","peopleInShade" } },
   { "thermal",                     { "corridor","diningRoom","lakeSide","library","park" } },
   { "turbulence",                  { "turbulence0","turbulence1","turbulence2","turbulence3" } }
};

std::pair<std::string,std::string> query_data(int proc_id){
    std::string scenario_name;
    std::string sequence_name;

    int accum_sum = 0;
    for(auto data_pair : cdnet_data){
        std::string data_scenario = data_pair.first;
        std::vector<std::string> data_sequences = data_pair.second;

        // accum_sum += data_sequences.size();

        if(proc_id < (accum_sum+data_sequences.size())){
            int data_idx = proc_id - accum_sum;

            scenario_name = data_scenario;
            sequence_name = data_sequences[data_idx];

            break;
        }
        else accum_sum += data_sequences.size();
    }

    return std::make_pair(scenario_name,sequence_name);
}

std::map<int,int> nproc_per_gpu_by_batch_size {
    { 2, 14 },
    { 4, 10 },
    { 8, 6 },
    { 12, 4 },
};

int main(){

    for(auto par : nproc_per_gpu_by_batch_size){
        std::cout << par.first << " " << par.second << std::endl;
    }


    // std::vector<std::string> seq, sce;

    // int cnt = -1;
    // for(auto word : cdnet_data){
    //     auto key = word.first;
    //     auto val = word.second;

    //     for(auto x : val){
    //         cnt++;
    //         // std::cout << cnt << "\t" << key << "\t" << x << "\n";

    //         sce.push_back(key);
    //         seq.push_back(x);
    //     }
    // }

    // for(int i=0;i<=51;i++){
    //     std::cout << i;
    //     std::pair<std::string,std::string> data = query_data(i);

    //     std::cout << "\t" << data.first << "\t" << data.second << "\n";

    //     if(data.first != sce[i] || data.second != seq[i]){
    //         std::cout << "\t" << "Wrong: " <<  sce[i] << "\t" << seq[i] << "\n";
    //     }

    // }

    return 0;
}