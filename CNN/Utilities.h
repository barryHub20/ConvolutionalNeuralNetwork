#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iterator>
#include <bitset>
#include <iomanip>
using namespace std;
const double learningRate = 0.001;

// logistics functions
double ReLU_Function(double x);
double ReLU_Derivative(double reluVal);

// file io
void readMnistFile(string fileName, vector<char>& contents);
void printImage(int imageIdx, vector<char>& contents, vector<char>& labels);

// read image to MLP
void LoadMnistImageToNeurons(const vector<char>& contents, int imageIdx, vector<double>& neurons);
void LoadMnistImageToNeurons(const vector<char>& contents, int imageIdx, double** neurons);
int readLabelFromMnist(const vector<char>& labels, int imageIdx);