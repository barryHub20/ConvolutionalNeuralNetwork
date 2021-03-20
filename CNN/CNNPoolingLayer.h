#pragma once
#include "CNNConvLayer.h"

struct PoolNeuron
{
	double input;
	double inputActivated;
	int xIndex;
	int yIndex;

	PoolNeuron() {
		input = inputActivated = 0.0;
		xIndex = yIndex = -1;
	}
};

class CNNPoolingLayer : public Layer
{
public:
	PoolNeuron** input;
	CNNConvLayer* convLayer;
	int inputWidth, inputHeight;
	bool pooling;	// is pooling activated?

public:
	// functions
	CNNPoolingLayer();
	~CNNPoolingLayer();

	// init
	void init(CNNConvLayer* convLayer, bool pooling);

	// perform pooling
	void performPooling();
};