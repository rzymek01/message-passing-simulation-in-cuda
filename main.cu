#include <cstdio>
#include <iostream>

// from CUDA book
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
//

#define imin(a,b) (a<b?a:b)

struct NodeData {
	float v_h;
	float G_0;
	float G_max;
	float v_d;
	float v_r;
};


__global__ void dot(const int N, const int *dev_V, NodeData *dev_Vdata, const int Elen, const int *dev_E,
		int *dev_M, const int t_c, const int t_p) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	dev_M[tid] = 2;

	// synchronize threads in this block
	//__syncthreads();


}



int main(void) {

	// read parameters
	int N,				// no. of vertices (nodes)
		Elen,			// no. of edges
		Vsrc;			// source vertex
	int t_c = 3,		// communication time [s]
		t_p = 30,		// processing time [s] e.g. short movie
		t_s = 330;		// max. simulation time [s]
	float v_h = 0,		// context potential [-1; 1]
	    G_0 = 1,		// initial conductivity [0, G_max)
	    G_max = 100,	// max. conductivity (G_0, +inf)
	    v_r = 1,		// registration potential (0, +inf)
	    v_d = 1;		// decision-making potential (0, +inf)

	int *V, *dev_V,
		*E, *dev_E,
		*M, *dev_M;
	NodeData *Vdata, *dev_Vdata;

	//
	std::cin >> N;

	// N+1 - space for one extra element at the end (for easiest iteration through graph)
	V = (int*) malloc((N+1) * sizeof(int));
	HANDLE_ERROR(cudaMalloc((void**) &dev_V, (N+1) * sizeof(int)));

	Vdata = (NodeData*) malloc(N * sizeof(NodeData));
	HANDLE_ERROR(cudaMalloc((void**) &dev_Vdata, N * sizeof(NodeData)));

	for (int i = 0; i < N; ++i) {
		std::cin >> v_h >> G_0 >> G_max >> v_d;
		Vdata[i].v_h = v_h;
		Vdata[i].G_0 = G_0;
		Vdata[i].G_max = G_max;
		Vdata[i].v_d = v_d;
		Vdata[i].v_r = v_r;
	}

	//
	std::cin >> Elen;

	E = (int*) malloc(Elen * sizeof(int));
	HANDLE_ERROR(cudaMalloc((void**) &dev_E, Elen * sizeof(int)));

	M = (int*) calloc(Elen, sizeof(int));	// zero-initialized
	HANDLE_ERROR(cudaMalloc((void**) &dev_M, Elen * sizeof(int)));

	V[0] = 0;
	{
		int Elen_i, start, end, e_i;
		for (int i = 0; i < N; ++i) {
			std::cin >> Elen_i;
			start = V[i];
			end = start + Elen_i;
			V[i + 1] = end;

			for (int j = start; j < end; ++j) {
				std::cin >> e_i;
				E[j] = e_i;
			}
		}
	}

	// get source vector and create M(0)
	std::cin >> Vsrc;

	{
		int start = V[Vsrc];
		int end = V[Vsrc + 1];

		for (int i = start; i < end; ++i) {
			M[i] = 1;
		}
	}

	//
	std::cin >> t_c >> t_p >> t_s;

	// debug
	for (int i = 0; i < N; ++i) {
		std::cout << V[i] << " ";
	}
	std::cout << std::endl;

	for (int i = 0; i < Elen; ++i) {
		std::cout << E[i] << " ";
	}
	std::cout << std::endl;

	for (int i = 0; i < Elen; ++i) {
		std::cout << M[i] << " ";
	}
	std::cout << std::endl;

	//
	const int threadsPerBlock = 1024;
	int blocksPerGrid = (N + threadsPerBlock-1) / threadsPerBlock;	// floor

	// copy the data to the GPU
	HANDLE_ERROR(cudaMemcpy(dev_V, V, (N+1) * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_E, E, Elen * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_M, M, Elen * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_Vdata, Vdata, N * sizeof(NodeData), cudaMemcpyHostToDevice));

	dot<<<blocksPerGrid, threadsPerBlock>>>(N, dev_V, dev_Vdata, Elen, dev_E, dev_M, t_c, t_p);

	// copy the data from the GPU to the CPU
	HANDLE_ERROR(cudaMemcpy(M, dev_M, Elen * sizeof(int), cudaMemcpyDeviceToHost));

	// debug
	for (int i = 0; i < Elen; ++i) {
		std::cout << M[i] << " ";
	}
	std::cout << std::endl;

	// free memory on the GPU side
	cudaFree(dev_V);
	cudaFree(dev_E);
	cudaFree(dev_M);
	cudaFree(dev_Vdata);

	// free memory on the CPU side
	free(V);
	free(E);
	free(M);
	free(Vdata);
}
