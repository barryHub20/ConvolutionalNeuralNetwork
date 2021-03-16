#pragma once
#include "MLPNeuron.h"
#include "Utilities.h"

/************************************************************************************************************
Training:
init(...);
loop:
	loadImage(...);
	train();
saveToTextFile();
*************************************************************************************************************/
class MLP
{
public:
	// data
	int imageIndex, correctIndex;

	// layers
	vector< vector<MLPNeuron> > layers;
	vector<double> costLayer;

	// flags
	bool usingFCLayer;

public:
	// functions
	MLP();
	~MLP();

	// init
	void init(int inputTotalPixels, const vector<int>& hiddenLayersSize, int totalOutputClasses, int layerOffset, double customDivider);

	// load dataset
	void loadImage(const vector<char>& contents, const vector<char>& labels, int imageIndex);
	void loadFCLayer(const vector<double>& FCLayer, int imageIndex, int correctImageIndex);

	// train
	void train(bool showCost, int iteration, int epoch);
	void forwardPass();
	void lossFunction(bool showCost, int iteration, int epoch);
	void backwardPass();
	void weightsUpdate();

	// save results
	void saveToTextFile();

	// test
	void test(const vector<char>& contents, const vector<char>& labels, bool onlyShowAccuracyAtEnd);
};