/**
 * Usage:
 * ./msg-spr-sim [<no_of_threads_per_block>] [<device_id>] <”<input_file>” >”<output_file>” 2>”<file_with_exec_time>”
 * ./msg-spr-sim [...] | dot -Tpng -o graph.png
 *
 * Input file format:
 * 		| %d			// #V
 * x #V	| %f %f %f %f 	// v_h_i, G_0_i, G_max_i, v_d_i
 * 		| %d			// #E
 * x #V	| %d [%d, ...]	// #edges of v_i [v_j, ...]
 * 		| %d			// source v_i
 * 		| %d %d %d 		// t_c, t_p, t_s
 *
 * Output file format: dot
 * Stderr output: single double number - execution time
 */
#include <cstdio>
#include <iostream>
#include <iomanip>

//#define _DEBUG

// from CUDA book
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << "in " << file << " at " << line << std::endl;
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
	int   new_last_t;
	bool  send;
	bool  new_send;
};

inline void printG(NodeData &data) {
	std::cout << std::fixed << std::setprecision(2) << data.G_0 << "\t";
}

__global__ void recv(const int N, const int *V, NodeData *Vdata, const int Elen,
		const int *X, const int *M, const int t_p) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= Elen) {
		return;
	}

	int msg = M[tid];
	NodeData *data = &Vdata[X[tid]];

	data->send = data->new_send;

	if (msg <= 0 || msg <= data->last_t) {
		return;
	}

	// determine new last time
	atomicMax(&(data->new_last_t), msg);

	// change G_0
	float G_0 = data->G_0, G_max = data->G_max;
	atomicExch(&(data->G_0), G_max - (G_max - G_0) * exp(-0.01 * t_p));
}

__global__ void send(const int N, const int *V, NodeData *Vdata, const int Elen,
		const int *E, int *M, const int t_c, const int t_p) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= Elen) {
		return;
	}

	NodeData *data = &Vdata[E[tid]];

	if (data->send) {
		return;
	}

	data->last_t = data->new_last_t;

	// sending messages
	if (data->G_0 * (data->v_h + data->v_r) >= data->v_d) {
		data->new_send = true;
		M[tid] = data->new_last_t + t_p + t_c;
	}
}


int main(int argc, char* argv[]) {

	// obtain available devices
	int deviceCount;
	HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

	if (deviceCount <= 0) {
		std::cerr << "No CUDA devices has been found!" << std::endl;
		exit(1);
	}

	cudaDeviceProp *prop = new cudaDeviceProp[deviceCount];

	for (int i = 0; i < deviceCount; ++i) {
		HANDLE_ERROR(cudaGetDeviceProperties(&(prop[i]), i));
#ifdef _DEBUG
		std::cout << "/*name: " << prop[i].name << "; max_threads_per_block: " << prop[i].maxThreadsPerBlock << "*/\n";
#endif
	}

	// read parameters
	int threadsPerBlock = -1,
		deviceId = -1;

	if (argc >= 2) {
		threadsPerBlock = atoi(argv[1]);
	}
	if (argc >= 3) {
		deviceId = atoi(argv[2]);
	}

	if (deviceId == -1) {
		deviceId = 0;
	}
	if (threadsPerBlock == -1) {
		threadsPerBlock = prop[deviceId].maxThreadsPerBlock;
	}

	if (deviceId < 0 || deviceId >= deviceCount) {
		std::cerr << "Device id out of range!" << std::endl;
		exit(3);
	}
	if (threadsPerBlock < 2 || threadsPerBlock > prop[deviceId].maxThreadsPerBlock) {
		std::cerr << "Number of threads per block is too small or too big!" << std::endl;
		exit(2);
	}

	// select device
	cudaSetDevice(deviceId);

	std::cout << "/*selected device: " << prop[deviceId].name << "; threads_per_block: "
			<< threadsPerBlock << "*/\n";

	// read input data
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
		*M, *dev_M, *dev_X;
	NodeData *Vdata, *dev_Vdata;

#ifdef _DEBUG
	std::cout << "/*\nProgram started." << std::endl;
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
		Vdata[i].new_last_t = 0;
		Vdata[i].send = false;
		Vdata[i].new_send = false;
	}

	//
	std::cin >> Elen;

	E = (int*) malloc(Elen * sizeof(int));
	HANDLE_ERROR(cudaMalloc((void**) &dev_E, Elen * sizeof(int)));

	M = (int*) malloc(Elen * sizeof(int));
	HANDLE_ERROR(cudaMalloc((void**) &dev_M, Elen * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**) &dev_X, Elen * sizeof(int)));

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
				M[j] = i; // M is used here as X (X is used only in kernel)
			}
		}
	}

//	printArray(M, Elen);

	HANDLE_ERROR(cudaMemcpy(dev_X, M, Elen * sizeof(int), cudaMemcpyHostToDevice));

	// zero-filling M
	memset((void*)M, 0, Elen * sizeof(int));

//	printArray(M, Elen);

	// get source vector and create M(0)
	std::cin >> Vsrc;
	Vdata[Vsrc].send = true;
	Vdata[Vsrc].new_send = true;

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
	const int blocksPerGrid = (Elen + threadsPerBlock-1) / threadsPerBlock;	// ceil(x/y)

	// capture the start time
	cudaEvent_t	startEvent, stopEvent;
	HANDLE_ERROR(cudaEventCreate(&startEvent));
	HANDLE_ERROR(cudaEventCreate(&stopEvent));
	HANDLE_ERROR(cudaEventRecord(startEvent, 0));

	// copy the data to the GPU
	HANDLE_ERROR(cudaMemcpy(dev_V, V, (N+1) * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_E, E, Elen * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_M, M, Elen * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_Vdata, Vdata, N * sizeof(NodeData), cudaMemcpyHostToDevice));

	for (int t = 1, i = 1; t <= t_s; t += t_c + t_p, ++i) {
		recv<<<blocksPerGrid, threadsPerBlock>>>(N, dev_V, dev_Vdata, Elen, dev_X, dev_M, t_p);
		send<<<blocksPerGrid, threadsPerBlock>>>(N, dev_V, dev_Vdata, Elen, dev_E, dev_M, t_c, t_p);
	}

	// copy the data from the GPU to the CPU
	HANDLE_ERROR(cudaMemcpy(M, dev_M, Elen * sizeof(int), cudaMemcpyDeviceToHost));

	// capture the end time
	HANDLE_ERROR(cudaEventRecord(stopEvent, 0));
	cudaEventSynchronize(stopEvent);

	float compTime;
	HANDLE_ERROR(cudaEventElapsedTime(&compTime, startEvent, stopEvent));

	HANDLE_ERROR(cudaEventDestroy(startEvent));
	HANDLE_ERROR(cudaEventDestroy(stopEvent));

#ifdef _DEBUG
	std::cout << "M: ";
	printArray(M, Elen);
	std::cout << "*/\n";
#endif

	// write computing time (to cerr for simplicity)

	std::cerr << compTime << std::endl;

	// generate output
	// 1. how many recipients
	// 2. max. distance from source (range)
	// -> 3. a graph at the end of the simulation in dot format

	std::cout << "digraph G {\n";
	std::cout << "\tnode [fontsize=12]\n";
	std::cout << "\tedge [fontcolor=\"0.5 0.5 0.5\",fontsize=8]\n";
	std::cout << "\t" << Vsrc << " [label=\"src\"]\n\n";

	int start, end;
	for (int i = 0; i < N; ++i) {
		start = V[i];
		end = V[i + 1];
		for (int j = start; j < end; ++j) {
			std::cout << "\t" << E[j] << " -> " << i;
			if (M[j] > 0) {
				std::cout << " [label=" << M[j] << "]\n";
			} else {
				std::cout << " [style=dotted]\n";
			}
		}
	}


	std::cout << "}" << std::endl;

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
