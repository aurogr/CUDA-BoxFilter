//****************************************************************************
// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//****************************************************************************

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

const int TILE_WIDTH = 32;
const int TILE_HEIGHT = 32;
// Variable global que debe ser tratadas como si fueran constantes si no es _CANNY_EDGE
int FILTERSIZE = 9;

// Defines a utilizar en caso de realizar esa funcionalidad (con ifdef y ifndef)
//#define _CONSTANT_MEMORY 

#ifdef _CONSTANT_MEMORY
    #define MAX_CONSTANT_FILTER_SIZE FILTERSIZE
    __constant__ float d_filter_costant[CONSTANT_FILTER_SIZE * CONSTANT_FILTER_SIZE];
#endif // _CONSTANT_MEMORY


//#define _SHARED_MEMORY
//#define _CANNY_EDGE

__host__ __device__ void clamp(int &value, int min, int max) {
    if (value < min) value = min;
    else if (value > max) value = max;
}

__host__ __device__ void clamp(float &value, float min, float max) {
    if (value < min) value = min;
    else if (value > max) value = max;
}

__global__
void convolution(const unsigned char* const inputChannel,
    unsigned char* const outputChannel,
    int numRows, int numCols,
    const float* const filter, const int filterWidth)
{
	// TODO: DONE
    
	size_t absolute_image_position_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t absolute_image_position_y = blockIdx.y * blockDim.y + threadIdx.y;

    // NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la imagen
    // numRows = height, numCols = width
     if ( absolute_image_position_x >= numCols ||
          absolute_image_position_y >= numRows ) 
     {
         return;
     }

	 int hafFilterWidth = filterWidth / 2; // offset that i need to apply to the globalPos inside the convolution loop to get the correct neighbor
	 float convResult = 0.f;

     for (int krow = 0; krow < filterWidth; ++krow) {
		 for (int kcol = 0; kcol < filterWidth; ++kcol) { // loop over the filter elements

             int neighborRow = absolute_image_position_y + krow - hafFilterWidth;
			 int neighborCol = absolute_image_position_x + kcol - hafFilterWidth;

             // NOTA: Que un thread tenga una posición correcta en 2D no quiere decir que al aplicar el filtro
             // los valores de sus vecinos sean correctos, ya que pueden salirse de la imagen.
			 // clamp neighborRow to be within the image boundaries by repeting the edge values
			 clamp(neighborRow, 0, numRows - 1);
			 clamp(neighborCol, 0, numCols - 1);

#ifdef _CONSTANT_MEMORY
			 convResult += d_filter_costant[krow * filterWidth + kcol] * inputChannel[neighborRow * numCols + neighborCol];
#elif _SHARED_MEMORY

#else // Global memory (default case)
			 convResult += filter[krow * filterWidth + kcol] * inputChannel[neighborRow * numCols + neighborCol];
#endif
         }

	 }

	 int idx = absolute_image_position_y * numCols + absolute_image_position_x;
     clamp(convResult, 0.f, 255.f);
	 outputChannel[idx] = convResult;
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
    int numRows,
    int numCols,
    unsigned char* const redChannel,
    unsigned char* const greenChannel,
    unsigned char* const blueChannel)
{
    // TODO: DONE
	size_t absolute_image_position_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t absolute_image_position_y = blockIdx.y * blockDim.y + threadIdx.y;


    // NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la imagen
     if ( absolute_image_position_x >= numCols ||
          absolute_image_position_y >= numRows )
     {
         return;
     }

	 size_t idx = absolute_image_position_y * numCols + absolute_image_position_x;

	 redChannel[idx] = inputImageRGBA[idx].x;
	 greenChannel[idx] = inputImageRGBA[idx].y;
	 blueChannel[idx] = inputImageRGBA[idx].z;
}

//This kernel takes in three color channels and recombines them
//into one image. The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
    const unsigned char* const greenChannel,
    const unsigned char* const blueChannel,
    uchar4* const outputImageRGBA,
    int numRows,
    int numCols)
{
    const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    //make sure we don't try and access memory outside the image
    //by having any threads mapped there return early
    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return;

    unsigned char red = redChannel[thread_1D_pos];
    unsigned char green = greenChannel[thread_1D_pos];
    unsigned char blue = blueChannel[thread_1D_pos];

    //Alpha should be 255 for no transparency
    uchar4 outputPixel = make_uchar4(red, green, blue, 255);

    outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char* d_red, * d_green, * d_blue;

void allocateMemoryGPU(const size_t numRowsImage, const size_t numColsImage)
{
    //allocate memory for the three different channels
    checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));
}

void allocateFilterAndCopyToGPU(const float* h_filter, const size_t filterWidth, float** d_filter)
{
	// TODO: DONE
#ifdef _CONSTANT_MEMORY
	cudaMemcpyToSymbol(d_filter_costant, h_filter, sizeof(float) * filterWidth * filterWidth);
	*d_filter = nullptr; // pointer is not used in this case since we will access the filter directly from constant memory

#elif _SHARED_MEMORY

#else // Global memory (default case)
    checkCudaErrors(cudaMalloc(d_filter, sizeof(float) * filterWidth * filterWidth));
	checkCudaErrors(cudaMemcpy(*d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
#endif
}

//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanupGPU() {
    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_green));
    checkCudaErrors(cudaFree(d_blue));
}


void create_filter(float** h_filter, int* filterWidth, int id_filter) {

    const int KernelWidth = FILTERSIZE; //OJO CON EL TAMAÑO DEL FILTRO//
    *filterWidth = KernelWidth;

    //create and fill the filter we will convolve with
    *h_filter = new float[KernelWidth * KernelWidth];

    switch (id_filter)
    {

    case 0: //Filtro gaussiano: blur
    {
        const float KernelSigma = 2.;

        float filterSum = 0.f; //for normalization

        for (int r = -KernelWidth / 2; r <= KernelWidth / 2; ++r) {
            for (int c = -KernelWidth / 2; c <= KernelWidth / 2; ++c) {
                float filterValue = expf(-(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
                (*h_filter)[(r + KernelWidth / 2) * KernelWidth + c + KernelWidth / 2] = filterValue;
                filterSum += filterValue;
            }
        }

        float normalizationFactor = 1.f / filterSum;

        for (int r = -KernelWidth / 2; r <= KernelWidth / 2; ++r) {
            for (int c = -KernelWidth / 2; c <= KernelWidth / 2; ++c) {
                (*h_filter)[(r + KernelWidth / 2) * KernelWidth + c + KernelWidth / 2] *= normalizationFactor;
            }
        }
    }
    break;

    case 1: // Filtro Laplaciano 5x5 
    {
        (*h_filter)[0] = 0;   (*h_filter)[1] = 0;    (*h_filter)[2] = -1.;  (*h_filter)[3] = 0;    (*h_filter)[4] = 0;
        (*h_filter)[5] = 0;  (*h_filter)[6] = -1.;  (*h_filter)[7] = -2.;  (*h_filter)[8] = -1.;  (*h_filter)[9] = 0;
        (*h_filter)[10] = -1.; (*h_filter)[11] = -2.; (*h_filter)[12] = 17.; (*h_filter)[13] = -2.; (*h_filter)[14] = -1.;
        (*h_filter)[15] = 0; (*h_filter)[16] = -1.; (*h_filter)[17] = -2.; (*h_filter)[18] = -1.; (*h_filter)[19] = 0;
        (*h_filter)[20] = 0;  (*h_filter)[21] = 0;   (*h_filter)[22] = -1.; (*h_filter)[23] = 0;   (*h_filter)[24] = 0;
    }

    case 2: // Filtro sobel horizontal 3x3
    {
        (*h_filter)[0] = -1; (*h_filter)[1] = 0; (*h_filter)[2] = 1;
        (*h_filter)[3] = -2; (*h_filter)[4] = 0; (*h_filter)[5] = 2;
        (*h_filter)[6] = -1; (*h_filter)[7] = 0; (*h_filter)[8] = 1;
	}

    case 3: // Filtro sobel vertical 3x3
    {
        (*h_filter)[0] = -1; (*h_filter)[1] = -2; (*h_filter)[2] = -1;
        (*h_filter)[3] = 0; (*h_filter)[4] = 0; (*h_filter)[5] = 0;
        (*h_filter)[6] = 1; (*h_filter)[7] = 2; (*h_filter)[8] = 1;
	}
    break;

    //TODO: crear los filtros segun necesidad. filter debe contener el filtro al finalizar esta función
    //NOTA: cuidado al establecer el tamaño del filtro a utilizar 

    default:
        printf("Filtro no definido\n");
        exit(1);
    }
}


void box_filter(uchar4* const d_inputImageRGBA,
    uchar4* const d_outputImageRGBA,
    const size_t numRows, const size_t numCols,
    unsigned char* d_redFiltered,
    unsigned char* d_greenFiltered,
    unsigned char* d_blueFiltered,
    int id_filter)
{

    float* h_filter;
    float* d_filter;
    int filterWidth;

    //Crea d_red, d_green y d_blue en GPU. Son variables globales con una vez basta
    allocateMemoryGPU(numRows, numCols);

    // Crear el filtro en CPU y subirlo a GPU 
    create_filter(&h_filter, &filterWidth, id_filter);
    allocateFilterAndCopyToGPU(h_filter, filterWidth, &d_filter);


    //En _CANNY_EDGE el metodo tendra que ejcutar todos los pasos llamando a diferentes kernels y creando los filtros en CPU (create_filter) y subiendolos a GPU correspondientes (allocateFilterAndCopyToGPU)

    //En el caso de Box Filter (un único filtro) el metodo realiza la convolucion siguiendo los siguientes pasos 

    //TODO: DONE Calcular tamaños de bloque
    const dim3 blockSize(TILE_WIDTH, 
                        TILE_HEIGHT, 
                        1);
    const dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x,
                        (numRows + blockSize.y - 1) / blockSize.y,
                        1);

    //TODO: Lanzar kernel para separar imagenes RGBA en diferentes colores
    separateChannels << <gridSize, blockSize >> > (d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
     
    //TODO: DONE Ejecutar kernels para convoluciones teniendo uno por canal
    create_filter(&h_filter, &filterWidth, 0);
    allocateFilterAndCopyToGPU(h_filter, filterWidth, &d_filter);

    convolution << <gridSize, blockSize >> > (d_red, d_redFiltered, numRows, numCols, d_filter, filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    convolution << <gridSize, blockSize >> > (d_green, d_greenFiltered, numRows, numCols, d_filter, filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    convolution << <gridSize, blockSize >> > (d_blue, d_blueFiltered, numRows, numCols, d_filter, filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Recombining the results. 
    recombineChannels << <gridSize, blockSize >> > (d_redFiltered, d_greenFiltered, d_blueFiltered, d_outputImageRGBA, numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}

__global__ void rgba_to_greyscale(const uchar4* const rgbaImage,
    unsigned char* const greyImage,
    int numRows, int numCols)
{
    size_t absolute_image_position_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t absolute_image_position_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (absolute_image_position_x >= numCols ||
        absolute_image_position_y >= numRows)
    {
        return;
    }

    size_t idx = absolute_image_position_y * numCols + absolute_image_position_x;
    uchar4 rgbaPixel = rgbaImage[idx];
    // Convert to greyscale using the luminosity method
    greyImage[idx] = static_cast<unsigned char>(0.299f * rgbaPixel.x + 0.587f * rgbaPixel.y + 0.114f * rgbaPixel.z);
}

void canny_edge_detector_filter(uchar4* const d_inputImageRGBA,
    uchar4* const d_outputImageRGBA,
    const size_t numRows, const size_t numCols,
    unsigned char* d_redFiltered, unsigned char*d_greenFiltered)
{

    float* h_filter;
    float* d_filter;
    int filterWidth;

    // Crea d_red
    checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
    checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRows * numCols));

    // Calcular tamaños de bloque
    const dim3 blockSize(TILE_WIDTH, TILE_HEIGHT, 1);
    const dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x,
        (numRows + blockSize.y - 1) / blockSize.y,
        1);

	// 1. Change to greyscale
	rgba_to_greyscale << <gridSize, blockSize >> > (d_inputImageRGBA, d_red, numRows, numCols);

    // 2. Blur filter 
    create_filter(&h_filter, &filterWidth, 0);
    allocateFilterAndCopyToGPU(h_filter, filterWidth, &d_filter);

	// Now we only need to convolute one of the channels since they are all the same after converting to greyscale
    convolution << <gridSize, blockSize >> > (d_red, d_redFiltered, numRows, numCols, d_filter, filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// 3. Sobel filters
    create_filter(&h_filter, &filterWidth, 2);
    allocateFilterAndCopyToGPU(h_filter, filterWidth, &d_filter);
	convolution << <gridSize, blockSize >> > (d_redFiltered, d_greenFiltered, numRows, numCols, d_filter, filterWidth);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	create_filter(&h_filter, &filterWidth, 3);
	allocateFilterAndCopyToGPU(h_filter, filterWidth, &d_filter);
	convolution << <gridSize, blockSize >> > (d_greenFiltered, d_redFiltered, numRows, numCols, d_filter, filterWidth);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// 4. Non-maximum suppression, double thresholding and edge tracking by hysteresis

    // Recombining the results. 
    recombineChannels << <gridSize, blockSize >> > (d_redFiltered, d_redFiltered, d_redFiltered, d_outputImageRGBA, numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}



