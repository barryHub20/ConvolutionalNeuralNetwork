#pragma once
#include "CNNFilter.h"
#include "Layer.h"

class CNNPoolingLayer : public Layer
{
public:
	PoolNeuron** input;
	int inputWidth, inputHeight;
	int convLayerWidth, convLayerHeight;
	double** convInput;
	double** convInputActivated;
	bool pooling;	// is pooling activated?

public:
	// functions
	CNNPoolingLayer();
	~CNNPoolingLayer();

	// init
	void init(int convLayerWidth, int convLayerHeight, double** convInput, double** convInputActivated, bool pooling);

	// perform pooling
	void performPooling();

	// get
	int get1DSize();
};