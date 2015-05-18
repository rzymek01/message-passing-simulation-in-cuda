#include <cstdio>
#include <iostream>
#include <iomanip>

#define _DEBUG

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

template <typename T>
inline void printArray(const T *array, const int size) {
	for (int i = 0; i < size; ++i) {
		std::cout << array[i] << "\t";
	}
	std::cout << std::endl;
}

template <typename T>
inline void arrayMap(T *array, const int size, void (*callback)(T&) ) {
	for (int i = 0; i < size; ++i) {
		callback(array[i]);
	}
}

struct NodeData {
	float v_h;
	float G_0;
	float G_max;
	float v_d;
	float v_r;
	int   last_t;
	bool  send;
};

inline void printG(NodeData &data) {
	std::cout << std::fixed << std::setprecision(2) << data.G_0 << "\t";
}


__global__ void recv(const int N, const int *V, NodeData *Vdata, const int Elen, const int *E,
		int *M, const int t_c, const int t_p) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	NodeData *data = &Vdata[tid];

	if (data->send || tid >= N) {
		return;
	}

	int lastTime = 0, msgCount = 0;
	int start = V[tid];
	int end = V[tid + 1], msg;

	// reading messages
	for (int i = start; i < end; ++i) {
		msg = M[i];

		if (msg <= 0 || msg <= data->last_t) {
			continue;
		}

		if (lastTime < msg) {
			lastTime = msg;
		}
		++msgCount;
	}

	data->last_t = lastTime;

	// processing messages
	data->G_0 = data->G_max - (data->G_max - data->G_0) * exp(-0.01 * msgCount * t_p);

}

__global__ void send(const int N, const int *V, NodeData *Vdata, const int Elen, const int *E,
		int *M, const int t_c, const int t_p) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	NodeData *data = &Vdata[tid];

	if (data->send || tid >= N) {
		return;
	}

	int start = V[tid];
	int end = V[tid + 1];
	int lastTime, start2, end2, v;

	// sending messages
	if (data->G_0 * (data->v_h + data->v_r) >= data->v_d) {
		data->send = true;
		lastTime = data->last_t + t_p + t_c;

		for (int i = start; i < end; ++i) {
			v = E[i];
			start2 = V[v];
			end2 = V[v + 1];
			for (int j = start2; j < end2; ++j) {
				if (E[j] == tid) {
					M[j] = lastTime;
					break;
				}
			}
		}
	}
}


int main(void) {

	// read parameters
	int N,				// no. of vertices (nodes)
		Elen,			// no. of edges
		Vsrc;			// source vertex
	int t_c = 3,		// communication time [s]
		t_p = 30,		// processing time [s] e.g. short movie
		t_s = 330;		// max. simulation time [s]
	float v_h = 0,		// reflection potential [-1; 1]
	    G_0 = 1,		// initial conductivity [0, G_max)
	    G_max = 100,	// max. conductivity (G_0, +inf)
	    v_r = 1,		// registration potential (0, +inf)
	    v_d = 1;		// decision-making potential (0, +inf)

	int *V, *dev_V,
		*E, *dev_E,
		*M, *dev_M;
	NodeData *Vdata, *dev_Vdata;

#ifdef _DEBUG
	std::cout << "Program started." << std::endl;
#endif

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
		Vdata[i].last_t = 0;
		Vdata[i].send = false;
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
	Vdata[Vsrc].send = true;

	{
		int start = V[Vsrc];
		int end = V[Vsrc + 1];
		int start2, end2, v;

		for (int i = start; i < end; ++i) {
			v = E[i];
			start2 = V[v];
			end2 = V[v + 1];
			for (int j = start2; j < end2; ++j) {
				if (E[j] == Vsrc) {
					M[j] = 1;
					break;
				}
			}
		}
	}

	//
	std::cin >> t_c >> t_p >> t_s;

#ifdef _DEBUG
	std::cout << "V:\t";
	printArray(V, N);

	std::cout << "E:\t";
	printArray(E, Elen);

	std::cout << "M_0:\t";
	printArray(M, Elen);
#endif

	//
	const int threadsPerBlock = 1024;
	int blocksPerGrid = (N + threadsPerBlock-1) / threadsPerBlock;	// floor

	// copy the data to the GPU
	HANDLE_ERROR(cudaMemcpy(dev_V, V, (N+1) * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_E, E, Elen * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_M, M, Elen * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_Vdata, Vdata, N * sizeof(NodeData), cudaMemcpyHostToDevice));

	for (int t = 1; t <= t_s; t += t_c + t_p) {
		recv<<<blocksPerGrid, threadsPerBlock>>>(N, dev_V, dev_Vdata, Elen, dev_E, dev_M, t_c, t_p);
		send<<<blocksPerGrid, threadsPerBlock>>>(N, dev_V, dev_Vdata, Elen, dev_E, dev_M, t_c, t_p);

#ifdef _DEBUG
		HANDLE_ERROR(cudaMemcpy(M, dev_M, Elen * sizeof(int), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(Vdata, dev_Vdata, N * sizeof(NodeData), cudaMemcpyDeviceToHost));

		std::cout << "M_" << (t / (t_c + t_p) + 1) << ":\t";
		printArray(M, Elen);

		std::cout << "G_" << (t / (t_c + t_p) + 1) << ":\t";
		arrayMap(Vdata, N, printG);
		std::cout << std::endl;
#endif
	}

	// copy the data from the GPU to the CPU
	HANDLE_ERROR(cudaMemcpy(M, dev_M, Elen * sizeof(int), cudaMemcpyDeviceToHost));

//#ifdef _DEBUG
//	std::cout << "M: ";
//	printArray(M, Elen);
//#endif

	// generate output
	// 1. how many recipients
	// 2. max. distance from source (range)
	// 3. a graph at the end of the simulation in dot format
	//@todo: ^^

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
