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
	vector<double> dropoutRates;
	vector< vector<bool> > dropoutInputs;

	// flags
	bool usingFCLayer;

public:
	// functions
	MLP();
	~MLP();

	// init
	void init(int inputTotalPixels, const vector<int>& hiddenLayersSize, const vector<double>& dropoutRates, int totalOutputClasses);

	// load dataset
	void loadImage(const vector<char>& contents, const vector<char>& labels, int imageIndex);
	void loadFCLayer(const vector<double>& FCLayer, int imageIndex, int correctImageIndex);
	void setCorrectIndex(int imageIndex, int correctIndex);

	// train
	bool train(bool showCost, int iteration, int epoch, ofstream& outputStream);
	void forwardPass();
	bool lossFunction(bool showCost, int iteration, int epoch, ofstream& outputStream);
	void backwardPass();
	void weightsUpdate();

	// save results
	string logFileName();

	// test
	void test(const vector<char>& contents, const vector<char>& labels, bool onlyShowAccuracyAtEnd, ofstream& outputStream);
};