#include <upcxx/upcxx.hpp>

int main(){

    // Initialize process group
    upcxx::init();

    // Intro process id:
    int rank_me = upcxx::rank_me();
    int rank_n = upcxx::rank_n();

    // std::cout << "rank_me = " << rank_me << " out of " << rank_n << std::endl;

    // Barrier process
    upcxx::barrier();

    std::vector<int> vec;

    // Initialize array with sequential values
    int n_item = 9;
    for(int i=0;i<n_item;i++) vec.push_back(rank_me*10 + i);

    // Barrier process
    upcxx::barrier();

    // Print to debug value in vec
    // for(int id=0;id<rank_n;id++){
    //     if(rank_me==id){
    //         std::cout << "Rank " << rank_me << ": ";
    //         for(int i=0;i<n_item;i++) std::cout << vec[i] << "\t";
    //         std::cout << std::endl;
    //     }
    //     upcxx::barrier();
    // }

    // Collective communication
    int* vec_arr = &vec[0];
    // int sum[n_item] = {0};
    upcxx::barrier();
    upcxx::reduce_all(vec_arr, vec_arr, n_item, upcxx::op_fast_add).wait();

    if(rank_me == 0) std::cout << "\n";
    upcxx::barrier();

    // Print to debug value in vec
    for(int id=0;id<rank_n;id++){
        if(rank_me==id){
            std::cout << "Rank " << rank_me << ": ";
            for(int i=0;i<n_item;i++) std::cout << vec[i] << "\t";
            std::cout << std::endl;
        }
        upcxx::barrier();
    }

    upcxx::barrier();
    int x = rank_me + 100;

    upcxx::barrier();

    // Print to debug value in vec
    // for(int id=0;id<rank_n;id++){
    //     if(rank_me==id){
    //         std::cout << "Rank " << rank_me << ": " << x << "\n";
    //     }
    //     upcxx::barrier();
    // }

    x = upcxx::reduce_all(x, upcxx::op_fast_add).wait();

    if(rank_me == 0) std::cout << "\n";
    upcxx::barrier();

    // Print to debug value in vec
    for(int id=0;id<rank_n;id++){
        if(rank_me==id){
            std::cout << "Rank " << rank_me << ": " << x << "\n";
        }
        upcxx::barrier();
    }


    // Finalize process group
    upcxx::finalize();




    return 0;
}