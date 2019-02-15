//#include "loadImage.h"
#include "commandLine.h"

#define CREATE_INFER_BUILDER nvinfer1::createInferBuilder
#define CREATE_INFER_RUNTIME nvinfer1::createInferRuntime

// constructor
yoloNet::imageNet(): {	
	//mWidth          = 0;
	//mHeight         = 0;
	//mInputSize      = 0;
	//mMaxBatchSize   = 0;
	//mInputCPU       = NULL;
	//mInputCUDA      = NULL;
	//mEnableDebug    = false;
	//mEnableProfiler = false;
	//mEnableFP16     = false;
	//mOverride16     = false;

	/// imageNet
	//mOutput = 0;

}

// destructor
yoloNet::~imageNet() {
}

// LoadNetwork
bool yoloNet::LoadNetwork(const char* model_path, const char* input_blob,
	                      const std::vector<std::string>& output_blobs)

	std::ifstream cache(model_path); //*****
	printf(LOG_GIE "loading network profile from cache... %s\n", cache_path);
	gieModelStream << cache.rdbuf(); //*****
	cache.close();

	// test for half FP16 support
	nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(gLogger); //*****
	
	if( builder != NULL ) {
		mEnableFP16 = !mOverride16 && builder->platformHasFastFp16();
		printf(LOG_GIE "platform %s FP16 support.\n", mEnableFP16 ? "has" : "does not have");
		builder->destroy();	
	}
	/////////////////////////////////////////////
	printf(LOG_GIE "%s loaded\n", model_path);
	nvinfer1::IRuntime* infer = CREATE_INFER_RUNTIME(gLogger); //*****

	/////////////////////////////////////////////
	gieModelStream.seekg(0, std::ios::end);
	const int modelSize = gieModelStream.tellg();
	gieModelStream.seekg(0, std::ios::beg);

	void* modelMem = malloc(modelSize);

	if( !modelMem )
	{
		printf(LOG_GIE "failed to allocate %i bytes to deserialize model\n", modelSize);
		return 0;
	}

	gieModelStream.read((char*)modelMem, modelSize);
	nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(modelMem, modelSize, NULL);
	free(modelMem);
	/////////////////////////////////////////////

bool yoloNet::init(const char* model_path)
{
	tensorNet::LoadNetwork(model_path) 
	printf(LOG_GIE "%s loaded\n", model_path);
	printf("%s initialized.\n", model_path);
	return true;
}

// Create
yoloNet* imageNet::Create(const char* model_path)
{
	yoloNet* net = new imageNet();
	
	if( !net )
		return NULL;
	
	if( !net->init(model_path) )
	{
		printf("imageNet -- failed to initialize.\n");
		return NULL;
	}
	
	return net;
}

// Classify

int imageNet::Classify( float* rgba, float* net_out ) {
/*
	// downsample and convert to band-sequential BGR
	if( CUDA_FAILED(cudaPreImageNetMean((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
								 make_float3(104.0069879317889f, 116.66876761696767f, 122.6789143406786f),
								 GetStream())) )
	{
		printf(LOG_TRT "imageNet::PreProcess() -- cudaPreImageNetMean() failed\n");
		return false;
	}
	// process with TRT
    //imageNet::Process()
	void* bindBuffers[] = { mInputCUDA, mOutputs[0].CUDA };	
	cudaStream_t stream = GetStream();

	if( !stream ) {
		if( !mContext->execute(1, bindBuffers) ) {
			printf(LOG_TRT "imageNet::Process() -- failed to execute TensorRT network\n");
			return false;
		}
	}

	else {		
		// queue the inference processing kernels
		const bool result = mContext->enqueue(1, bindBuffers, stream, NULL);
		CUDA(cudaStreamSynchronize(stream));

		if( !result ) {
			printf(LOG_TRT "imageNet::Process() -- failed to enqueue TensorRT network\n");
			return false;
		}	
	}
	PROFILER_REPORT();
	//imageNet::Process() Done
	
	// determine the maximum class
	int classIndex = -1;
	float classMax = -1.0f;
	
	//mOutputClasses
	//const float value = mOutputs[0].CPU[n];
	//*net_out = classMax;
*/
}

// main entry point
int main( int argc, char** argv ) {
	const char* imgFilename = argv[1];
	const char* model_path = '/home/nolan/Desktop/YOLO/licence_plate/v1/export/onnx/out.engin';

	// create Net
	/*
	commandLine cmdLine();
	const char* input    = cmdLine.GetString("input_blob");
	const char* output   = cmdLine.GetString("output_blob");
	*/
	imageNet* net = yoloNet::Create(model_path);

	if( !net )
	{
		printf("imagenet-console:   failed to initialize imageNet\n");
		return 0;
	}
	/*
	net->EnableProfiler(); //???

	// load image from file on disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;
		
	if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}
	
	float net_out[] = [0] * 10
	// classify image
	while( loop ) {
		void* imgRGBA = NULL;
		// get imgRGBA from rostopic

		//net->Classify((float*)imgRGBA, &net_out);
		net->Classify(imgCUDA, imgWidth, imgHeight, &net_out);

		std::cout << net_out << endl;
		// net_out_msgs = imgRGBA.header
		// net_out_data = net_out
		// rospublish(net_out_msgs)	
	}

	CUDA(cudaFreeHost(imgCPU));
	delete net;
	return 0;
	*/
}