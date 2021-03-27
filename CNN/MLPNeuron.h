#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include "Neuron.h"
using namespace std;

/************************************************************************************************************
MLP neurons
*************************************************************************************************************/
class MLPNeuron : public Neuron
{
public:
	// indexing
	int layer, index;

	// for prev. layer
	vector<double> weights, weightsGradient;
	double bias, biasGradient;

	// value
	double z, localGradient;	// local gradient is eq1 * eq2

public:
	// functions
	MLPNeuron();
	~MLPNeuron();

	// init
	void initRandomize(int layer, int index, int totalWeights);
	void initAsPixel(int layer, int index, double pixel);
	void reset();

	// forward pass
	void forwardPass(const vector<MLPNeuron>& prevLayer);
	void forwardPassDropout();

	// backward pass
	void backwardPassOutputLayer(const vector<MLPNeuron>& prevLayer, double cost_m);
	void backwardPass(const vector<MLPNeuron>& prevLayer, const vector<MLPNeuron>& nextLayer);
	void backwardPassDropout();

	// weights update
	void weightsUpdate();
};