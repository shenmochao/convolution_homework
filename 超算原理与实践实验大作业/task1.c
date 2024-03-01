# include <stdio.h>
# include <stdlib.h>
# include <mpi.h>
# include <omp.h>
# include <time.h>

# define H 256
# define K 3
# define S 3
# define W ((H-K)/S)

int input[3][H][H];
int ker[3][K][K];
int rec[W][W];
int output[W][W];

void show_parameter(){
    printf("input size 3*N*N, N = %d\n", H);
    printf("stride = %d\n", S);
}

void input_init(){
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < 3; i++)
        for(int j = 0; j < H; j++)
            for(int k = 0; k < H; k++)
                input[i][j][k] = rand() % 100;
}

void ker_init(){
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < 3; i++)
        for(int j = 0; j < K; j++)
            for(int k = 0; k < K; k++)
                ker[i][j][k] = rand() % 100;
}

int main(int argc,char* argv[]){
    MPI_Init(&argc,&argv);
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
    if(rank == 0){
        // 初始化
        show_parameter();
        input_init();
        ker_init();
    }

    // 更新input和ker的数据
    MPI_Bcast(&input, 3*H*H, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ker, 3*K*K, MPI_INT, 0, MPI_COMM_WORLD);
    // 划分行
    int col = H;
    int rows_per_proc = H / num_procs;
    int start_row = rank * rows_per_proc;
    int end_row = start_row + rows_per_proc - 1;
    // 计算
    double start_time = MPI_Wtime();
#pragma omp parallel for collapse(3)
    for (int i = 0; i < 3; i++)
        for(int j = start_row; j < end_row - S; j+=S)
            for(int k = 0; k < col - S + 1; k+=S)
                for(int ii = 0; ii < 3; ii++)
                    for(int jj = 0; jj < K; jj++)
                        for(int kk = 0; kk < K; kk++)
                            rec[j/S][k/S] += input[i][j + jj][k + kk] * ker[ii][jj][kk];

    double end_time = MPI_Wtime();
    printf("Process %d finished in %f seconds.\n", rank, end_time - start_time);
    // 收集output
    MPI_Reduce((void *)&rec, (void *)&output, W * W, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0){
        printf("\nfinish\n");
    }
    MPI_Finalize();
}