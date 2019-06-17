/*****************************
Copyright:Yes
Author:Ytong
Date:2019.4.8
Desscription:����ʵ������������������������������Ͷ��壩�����еĳ��Զ�����������
*****************************/

#include "stdafx.cuh"
#include <omp.h>   

/*************************************************
Function:gPU_gauss_1
Description:ʹ��GPU�����ά��˹����������1
Calls:NULL
Table Accessed:�����ʵı����������ǣ�������ݿ�����ĳ���
Table Updated:���޸ĵı����������ǣ�������ݿ�����ĳ���
Input:�������double����s��������Ϊ��˹�����еĲ���s
Output:void
Return:None
Others:��Ӧ����CUDA_syn_Dimen_1
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
Description:ʹ��GPU�����ά��˹����������2
Calls:NULL
Table Accessed:�����ʵı����������ǣ�������ݿ�����ĳ���
Table Updated:���޸ĵı����������ǣ�������ݿ�����ĳ���
Input:�������double����s��������Ϊ��˹�����еĲ���s
Output:void
Return:None
Others:��Ӧ����CUDA_syn_Dimen_2
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
Description:ʹ��GPU�����ά��˹����������1
Calls:NULL
Table Accessed:�����ʵı����������ǣ�������ݿ�����ĳ���
Table Updated:���޸ĵı����������ǣ�������ݿ�����ĳ���
Input:�������double����s��������Ϊ��˹�����еĲ���s
Output:void
Return:None
Others:��Ӧ����CUDA_syn_Dimen1_2
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
Description:ʹ��CPU�����ά��˹����������1
Calls:NULL
Table Accessed:�����ʵı����������ǣ�������ݿ�����ĳ���
Table Updated:���޸ĵı����������ǣ�������ݿ�����ĳ���
Input:�������double����s��������Ϊ��˹�����еĲ���s
Output:void
Return:None
Others:2��ѭ�������м��㣨2s+1)^2��
*************************************************/
void cPU_gauss(double s)
{
	int size_x = 6 * int(s) + 1;
	int size_y = size_x;
	double arr[100][100] = {};
	//#pragma omp parallel    //�����Ƿ�ʹ��OpenMP������
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
Description:����kernel��ִ�����ã�grid��block�Ĺ�񣬼�1x1x1����169x1x1����s = 2Ϊ����ÿ���̼߳���13��
Calls:gPU_gauss_1
Table Accessed:�����ʵı����������ǣ�������ݿ�����ĳ���
Table Updated:���޸ĵı����������ǣ�������ݿ�����ĳ���
Input:�������double s�����ڴ��ݸ�gPU_gauss��Ϊ����
Output:�����������˵����
Return:��������ֵ��˵��
Others:����˵��
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
Description:����kernel��ִ�����ã�grid��block�Ĺ�񣬼�1x1x1����13x13x1����s = 2Ϊ����ÿ���̼߳���1��
Calls:gPU_gauss_2
Table Accessed:�����ʵı����������ǣ�������ݿ�����ĳ���
Table Updated:���޸ĵı����������ǣ�������ݿ�����ĳ���
Input:�������double s�����ڴ��ݸ�gPU_gauss��Ϊ����
Output:�����������˵����
Return:��������ֵ��˵��
Others:����˵��
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
Description:����kernel��ִ�����ã�grid��block�Ĺ�񣬼�1x1x1����13x1x1����s = 2Ϊ����,ÿ���̼߳���13��
Calls:gPU_gauss_3
Table Accessed:�����ʵı����������ǣ�������ݿ�����ĳ���
Table Updated:���޸ĵı����������ǣ�������ݿ�����ĳ���
Input:�������double s�����ڴ��ݸ�gPU_gauss��Ϊ����
Output:�����������˵����
Return:��������ֵ��˵��
Others:����˵��
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
Description:�����������ڶ���s�����ò�ͬ�ĺ����������˹��������ʱ����������
Calls:	QueryPerformanceCounter��&num��
		QueryPerformanceCounter(&num);
		cPU_gauss(s);
		CUDA_syn_Dimen_1(s);
		CUDA_syn_Dimen_2(s);
		CUDA_syn_Dimen1_2(s);
Table Accessed:�����ʵı����������ǣ�������ݿ�����ĳ���
Table Updated:���޸ĵı����������ǣ�������ݿ�����ĳ���
Input:	none
Output:�����������˵����
Return:��������ֵ��˵��
Others:����˵��
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
	printf("��˺������֣����к�ʱ��%dms\n", (end - start) * 1000 / freq);

	return 0;

}