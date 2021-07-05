
#include <upcxx/upcxx.hpp>

int main(){
    upcxx::init();

    int rank_me     = upcxx::rank_me();
    int rank_n      = upcxx::rank_n();
    int rank_left   = (rank_me-1+rank_n) % rank_n;
    int rank_right  = (rank_me+1) % rank_n;

    upcxx::dist_object<upcxx::global_ptr<float>> win_mem_left= upcxx::new_array<float>(2710);
    upcxx::dist_object<upcxx::global_ptr<float>> win_mem_right = upcxx::new_array<float>(2710);

    int n_epoch = 1000000;

    for(int epoch=0 ; epoch < n_epoch ; epoch++){
        std::cout << "Epoch #" << epoch << " --- " << "Rank #" << rank_me << ": Start to fetch rank_right" << "\n";
        upcxx::global_ptr<float> win_mem_neighbor_left = win_mem_left.fetch(rank_right).wait();
        std::cout << "Epoch #" << epoch << " --- " << "Rank #" << rank_me << ": Finish fetch rank_right" << "\n";

        std::cout << "Epoch #" << epoch << " --- " << "\tRank #" << rank_me << ": Start to fetch rank_left" << "\n";
        upcxx::global_ptr<float> win_mem_neighbor_right = win_mem_right.fetch(rank_left).wait();
        std::cout << "Epoch #" << epoch << " --- " << "\tRank #" << rank_me << ": Finish fetch rank_left" << "\n";
    }

    upcxx::finalize();
    return 0;
}