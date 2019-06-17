/*****************************
Copyright:Yes
Author:Ytong
Date:2019.4.8
Desscription:本次实验的主函数（包含其他函数的声明和定义），所有的尝试都在这里运行
*****************************/

#include "stdafx.cuh"
#include <omp.h>   

/*************************************************
Function:gPU_gauss_1
Description:使用GPU计算二维高斯函数的情形1
Calls:NULL
Table Accessed:被访问的表（此项仅对于牵扯到数据库操作的程序）
Table Updated:被修改的表（此项仅对于牵扯到数据库操作的程序）
Input:输入参数double类型s，用于作为高斯函数中的参数s
Output:void
Return:None
Others:对应的是CUDA_syn_Dimen_1
*************************************************/
__global__ void gPU_gauss_1(double s)
{
	double arr = 0;
	int row = threadIdx.x / int(6 * s + 1);
	int col = threadIdx.x % int(6 * s + 1);
	//printf("blockId:%d %d blockDim:%d %d threadId:%d %d\n",blockIdx.x,blockIdx.y,blockDim.x,blockDim.y,threadIdx.x,threadIdx.y);
	//printf("%d %d\n", row, col);
	arr = exp(-(pow(double(row) - 3 * s, 2) + pow(3 * s - double(col), 2))/ 2 / s / s)/ s / pow(2 * Pi, 0.5);
	printf("%5.4f ", arr);
}

/*************************************************
Function:gPU_gauss_2
Description:使用GPU计算二维高斯函数的情形2
Calls:NULL
Table Accessed:被访问的表（此项仅对于牵扯到数据库操作的程序）
Table Updated:被修改的表（此项仅对于牵扯到数据库操作的程序）
Input:输入参数double类型s，用于作为高斯函数中的参数s
Output:void
Return:None
Others:对应的是CUDA_syn_Dimen_2
*************************************************/
__global__ void gPU_gauss_2(double s)
{
	double arr = 0;
	int row = threadIdx.x;
	int col = threadIdx.y;
	//printf("blockId:%d %d blockDim:%d %d threadId:%d %d\n",blockIdx.x,blockIdx.y,blockDim.x,blockDim.y,threadIdx.x,threadIdx.y);
	//printf("%d %d\n", row, col);
	//arr = exp(-(pow(double(row) - 3 * s, 2) + pow(3 * s - double(col), 2)) / 2 / s / s) / s / pow(2 * Pi, 0.5);
	//printf("%5.4f ", arr);
}

/*************************************************
Function:gPU_gauss_3
Description:使用GPU计算二维高斯函数的情形1
Calls:NULL
Table Accessed:被访问的表（此项仅对于牵扯到数据库操作的程序）
Table Updated:被修改的表（此项仅对于牵扯到数据库操作的程序）
Input:输入参数double类型s，用于作为高斯函数中的参数s
Output:void
Return:None
Others:对应的是CUDA_syn_Dimen1_2
*************************************************/
__global__ void gPU_gauss_3(double s)
{
	double arr = 0;
	int row = threadIdx.x;
	int col = 0;
	//printf("blockId:%d %d blockDim:%d %d threadId:%d %d\n",blockIdx.x,blockIdx.y,blockDim.x,blockDim.y,threadIdx.x,threadIdx.y);
	//printf("%d %d\n", row, col);
		for (; col<int(6 * s + 1); col++)
		{
			arr = exp(-(pow(double(row) - 3 * s, 2) + pow(3 * s - double(col), 2)) / 2 / s / s) / s / pow(2 * Pi, 0.5);
			printf("%5.4f ", arr);
		}
}

/*************************************************
Function:cPU_gauss
Description:使用CPU计算二维高斯函数的情形1
Calls:NULL
Table Accessed:被访问的表（此项仅对于牵扯到数据库操作的程序）
Table Updated:被修改的表（此项仅对于牵扯到数据库操作的程序）
Input:输入参数double类型s，用于作为高斯函数中的参数s
Output:void
Return:None
Others:2层循环，串行计算（2s+1)^2次
*************************************************/
void cPU_gauss(double s)
{
	int size_x = 6 * int(s) + 1;
	int size_y = size_x;
	double arr[100][100] = {};
	//#pragma omp parallel    //决定是否使用OpenMP并行绪
	for (int i = 0; i < size_x; i++)
	{
		for (int j = 0; j < size_y; j++)
		{
			arr[i][j] = exp(-(pow(double(i) - 3 * s, 2) + pow(3 * s - double(j), 2)) / 2 / s / s) / s / pow(2 * Pi, 0.5);
			printf("%5.4f ", arr[i][j]);
		}
		printf("\n");
	}
}


/*************************************************
Function:CUDA_syn_Dimen_1
Description:定义kernel的执行配置，grid和block的规格，即1x1x1，和169x1x1（以s = 2为例）每个线程计算13次
Calls:gPU_gauss_1
Table Accessed:被访问的表（此项仅对于牵扯到数据库操作的程序）
Table Updated:被修改的表（此项仅对于牵扯到数据库操作的程序）
Input:输入参数double s，用于传递给gPU_gauss作为参数
Output:对输出参数的说明。
Return:函数返回值的说明
Others:其它说明
*************************************************/
void CUDA_syn_Dimen_1(double s)
{
	dim3 grid_size(1, 1, 1);
	dim3 block_size(pow(6*s+1,2), 1, 1);
	gPU_gauss_1<<<grid_size, block_size>>>(s);
	cudaDeviceSynchronize();
}

/*************************************************
Function:CUDA_syn_Dimen_2
Description:定义kernel的执行配置，grid和block的规格，即1x1x1，和13x13x1（以s = 2为例）每个线程计算1次
Calls:gPU_gauss_2
Table Accessed:被访问的表（此项仅对于牵扯到数据库操作的程序）
Table Updated:被修改的表（此项仅对于牵扯到数据库操作的程序）
Input:输入参数double s，用于传递给gPU_gauss作为参数
Output:对输出参数的说明。
Return:函数返回值的说明
Others:其它说明
*************************************************/
void CUDA_syn_Dimen_2(double s)
{
	dim3 grid_size(1, 1, 1);
	dim3 block_size(6 * s + 1, 6 * s + 1, 1);
	gPU_gauss_2<<<grid_size, block_size >>>(s);
	cudaDeviceSynchronize();
}


/*************************************************
Function:CUDA_syn_Dimen1_2
Description:定义kernel的执行配置，grid和block的规格，即1x1x1，和13x1x1（以s = 2为例）,每个线程计算13次
Calls:gPU_gauss_3
Table Accessed:被访问的表（此项仅对于牵扯到数据库操作的程序）
Table Updated:被修改的表（此项仅对于牵扯到数据库操作的程序）
Input:输入参数double s，用于传递给gPU_gauss作为参数
Output:对输出参数的说明。
Return:函数返回值的说明
Others:其它说明
*************************************************/
void CUDA_syn_Dimen1_2(double s)
{
	dim3 grid_size(1, 1, 1);
	dim3 block_size(6 * s + 1, 1, 1);
	gPU_gauss_3 << <grid_size, block_size >> >(s);
	cudaDeviceSynchronize();
}

/*************************************************
Function:main
Description:主函数，用于读入s，调用不同的函数来计算高斯函数，计时以评估性能
Calls:	QueryPerformanceCounter（&num）
		QueryPerformanceCounter(&num);
		cPU_gauss(s);
		CUDA_syn_Dimen_1(s);
		CUDA_syn_Dimen_2(s);
		CUDA_syn_Dimen1_2(s);
Table Accessed:被访问的表（此项仅对于牵扯到数据库操作的程序）
Table Updated:被修改的表（此项仅对于牵扯到数据库操作的程序）
Input:	none
Output:对输出参数的说明。
Return:函数返回值的说明
Others:其它说明
*************************************************/
int main()
{
	printf("Input the Number s:");
	double s = 0.0;
	scanf("%lf", &s);

	LARGE_INTEGER  num;
	long long int start, end, freq;
	QueryPerformanceFrequency(&num);
	freq = num.QuadPart;
	QueryPerformanceCounter(&num);
	start = num.QuadPart;
	//cPU_gauss(s);
	//CUDA_syn_Dimen_1(s);
	//CUDA_syn_Dimen_2(s);
	CUDA_syn_Dimen1_2(s);
	QueryPerformanceCounter(&num);
	end = num.QuadPart;
	printf("多核函数部分，运行耗时：%dms\n", (end - start) * 1000 / freq);

	return 0;

}