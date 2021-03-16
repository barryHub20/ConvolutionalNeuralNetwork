#pragma once
#include "CNNConvLayer.h"
#include "CNNFilter.h"
#define CNN_FULLY_CONNECTED_SIZE 50000

/************************************************************************************************************
Training:
init(...);
loop:
	loadImage(...);
	train();
saveToTextFile();
*************************************************************************************************************/
class CNN
{
public:
	// data
	int imageIndex, correctIndex;
	int FCLayerSize;

	// layers
	vector< vector<CNNFilter*> > filterList;
	vector< vector<CNNConvLayer*> > convLayerList;

	// MLP
	MLP myMlp;

	// misc
	vector<double> FCLayerVector;

public:
	// functions
	CNN();
	~CNN();

	// init: totalLayers includes pooling layers
	void init();
	void addNewLayer(int layersPerIndex, int filterSize);

	// load image
	void loadImage(const vector<char>& contents, const vector<char>& labels, int imageIndex);

	// train
	void train(bool showCost, int iteration, int epoch);
	void forwardPass(bool showCost, int iteration, int epoch, bool testMode);
	void backwardPass();
	void weightsUpdate();

	// save results
	void saveToTextFile();

	// test
	void test(const vector<char>& contents, const vector<char>& labels, bool onlyShowAccuracyAtEnd);
};

