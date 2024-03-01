# include <stdio.h>
# include <stdlib.h>
# include <mpi.h>
# include <omp.h>
# include <time.h>

# define H 512
# define K 3
# define S 1
# define W ((H-K)/S + 1)

// 定义输入图片、卷积核和输出特征
int input[3][H][H];
int ker[3][K][K];
int output[W][W];

// 定义展开后的矩阵X和Y
int X[K * K * 3][W * W];
int Y[1][K * K * 3];

// 定义输出特征的展开形式Z
int Z[W * W];

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

void other_init(){
    for (int i = 0; i < K * K * 3; i++)
        for(int j = 0; j < W * W; j++){
            X[i][j] = 0;
            Z[j] = 0;
        }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < K; j++)
            for (int k = 0; k < K; k++)
                Y[0][i*9+j*3+k] = ker[i][j][k];
}

int main(int argc,char* argv[]){
    MPI_Init(&argc,&argv);
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
    if(rank == 0){
        show_parameter();
        input_init();
        ker_init();
        other_init();
    }
    MPI_Bcast(&input, 3*H*H, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ker, 3*K*K, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&X, 3*K*K*W*W, MPI_INT, 0, MPI_COMM_WORLD);

    int cols_per_proc = W*W / num_procs;
    int start_col = rank * cols_per_proc;
    int end_col = start_col + cols_per_proc - 1;
    double start_time = MPI_Wtime();

#pragma omp parallel for collapse(1)
    // 将input转换为X矩阵，X的大小为K*K*3行，W*W列。i:0-W*W
    for(int i = start_col; i < end_col; i++){ // 对于给定范围的一列
        int inp_row = (i / W) * S;  // 逆向求回其在输入矩阵的位置
        int inp_col = (i % W) * S;
        int row = 0;  // 在一列中的行位置
        for(int kk = 0; kk < 3; kk++)  // 走过input中，卷积核的层数
            for(int ii = inp_row; ii < K; ii++) // 走过input中，卷积核的行数
                for(int jj = inp_col; jj < K; jj++){  // 走过input中，卷积核对应的列数
                        X[row][i] = input[kk][ii][jj];
                        row ++;
                    }
    }

    MPI_Bcast(&X, 3*K*K*W*W, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Y, 1*3*K*K, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Z, W*W, MPI_INT, 0, MPI_COMM_WORLD);

#pragma omp parallel for collapse(2)
    for(int i = start_col; i < end_col; i++)
        for(int j = 0; j < 3*K*K; j++){
            Z[i] += X[j][i] * Y[0][j];  // 计算Z矩阵的值
        }

    // 将Z还原为输出特征output
#pragma omp parallel for collapse(1)
    for (int j = start_col; j <= end_col; j++) { // 遍历当前处理单元负责的每一个元素
        output[(j / W)][(j % W)] = Z[j]; // 将Z矩阵中的元素赋给output矩阵
    }

    double end_time = MPI_Wtime();
    printf("Process %d finished in %f seconds.\n", rank, end_time - start_time);
    MPI_Reduce((void *)&X, (void *)&output, W*W, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        printf("finish\n");
    }
    MPI_Finalize();
}