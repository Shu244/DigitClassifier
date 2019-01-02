/*
 * Author: Shuhao Lai
 * Date: 12/30/2018
 * Tester.cpp
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include "DigitClassifier.h"

using std::ifstream;
using std::string;
using std::stringstream;
using std::vector;
using std::cout;
using std::endl;
using std::pair;
using std::make_pair;

typedef vector<pair<int, vector<double>>> labeledImages;
typedef vector<vector<double>> twoDArray;

std::ostream& operator<<(std::ostream & o, const vector<double> & vec)
{
	for (double a : vec)
		o << a << " ";
	return o;
}

void testHadamard(DigitClassifier & obj)
{
	vector<double> a
	{ 1, 2, 3, 4 };
	vector<double> b
	{ 5, 6, 7, 8 };
	vector<double> result
	{ 5, 12, 21, 32 };
	assert(obj.hadamard(a, b) == result);
}

//Note that regular vectors are row vectors not column vectors, which is the standard idea.
void testMultiplyMatrices(DigitClassifier & obj)
{
	twoDArray a
	{
	{ 1, 2, 3 },
	{ 4, 5, 6, },
	{ 7, 8, 9 } };
	twoDArray b
	{
	{ 5 },
	{ 6 },
	{ 7 } };
	twoDArray result
	{
	{ 38 },
	{ 92 },
	{ 146 } };
	assert(result == obj.multiplyMatrices(a, b));
}

//Transpose always returns a 2D array.
void testTranspose(DigitClassifier & obj)
{
	twoDArray a
	{
	{ 1, 2, 3 },
	{ 4, 5, 6, },
	{ 7, 8, 9 } };
	twoDArray b
	{
	{ 5 },
	{ 6 },
	{ 7 } };
	twoDArray c
	{
	{ 5, 6, 7 } };
	twoDArray result1
	{
	{ 1, 4, 7 },
	{ 2, 5, 8 },
	{ 3, 6, 9 } };
	twoDArray result2
	{
	{ 5, 6, 7 } };
	twoDArray result3
	{
	{ 5 },
	{ 6 },
	{ 7 } };
	assert(obj.transpose(a) == result1);
	assert(obj.transpose(b) == result2);
	assert(obj.transpose(c) == result3);

}

void testExtractDoubles(DigitClassifier & obj)
{
	string line = "1 2 3 5.0 5.4 0 0 0 0 0 0 1.2";
	vector<double> result
	{ 1, 2, 3, 5.0, 5.4, 0, 0, 0, 0, 0, 0, 1.2 };
	assert(result == obj.extractDoubles(line));
}

//Must manually check files to see if they are the same.
void testToStringAndReadIn(DigitClassifier & obj)
{
	//Test one
	obj.toString("test1.txt");
	DigitClassifier test("test1.txt");
	test.toString("test2.txt");

	//Using a manually written readIn file.
	DigitClassifier test2("readInTest.txt");
	test2.toString("ReadInTestOutPut.txt");
}

//Must manually check output for correctness.
void testFeedForwardOnce()
{
	DigitClassifier test("ExampleNeuralNetwork1.txt");
	vector<double> inputs
	{ 1, 2, 3 };
	vector<double> result = test.feedForwardOnce(inputs, 1);
	test.toString("FeedForwardOnceTest.txt");
	cout << result << endl;
	vector<double> result2 = test.feedForwardOnce(test.activations(result), 2);
	cout << result2 << endl;
}

//Must manually check output for correctness.
void testFillSystemRandomly()
{
	vector<int> conditions
	{ 4, 6, 2 };
	DigitClassifier test(conditions);
	test.toString("FillSystemRandomlyTest.txt");

}

//Must manually check output for correctness.
void testLastLayerError()
{
	/*
	 * Expected error values:
	 * 7.58256041e-10, -5.7498556e-19, 2.78946809e-10
	 */
	DigitClassifier test("ExampleNeuralNetwork1.txt");
	vector<double> inputs
	{ 1, 2, 3 };
	vector<double> result = test.feedForwardOnce(inputs, 1);
	vector<double> result2 = test.feedForwardOnce(test.activations(result), 2);
	vector<int> y
	{ 0, 1, 0 };
	vector<double> error = test.lastLayerError(result2, y);
	cout << error;
}

//Must manually check output for correctness.
void testBackpropagate()
{
	/*
	 * Expected errors are:
	 * 1.46947457e-20, 1.87328856e-23
	 * 7.58256041e-10, -5.7498556e-19, 2.78946809e-10
	 */
	DigitClassifier test("ExampleNeuralNetwork1.txt");
	twoDArray zVals;
	vector<double> inputs{ 1, 2, 3 };
	vector<double> result = test.feedForwardOnce(inputs, 1);
	vector<double> result2 = test.feedForwardOnce(test.activations(result), 2);
	zVals.push_back(result);
	zVals.push_back(result2);
	vector<int> y{ 0, 1, 0 };
	vector<double> preError = test.lastLayerError(result2, y);
	twoDArray totalError;
	totalError.push_back(preError);
	test.backpropagate(0, preError, zVals, totalError);
	cout << totalError[1] << endl;
	cout << totalError[0] << endl;
}

//Must manually check output for correctness.
void testShuffleImages(DigitClassifier & obj)
{
	labeledImages a;
	for(int i = 0; i < 15; i++)
	{
		vector<double> b = {1.0+i, 2.0+i, 3.0+i, 4.0+i};
		pair<int, vector<double>> c{i, b};
		a.push_back(c);
	}
	obj.shuffleImages(a);
	for(pair<int, vector<double>> img : a)
		cout << img.second << endl;
}

//Must manually check output for correctness.
void testSGD(DigitClassifier & obj)
{
	labeledImages a;
	for(int i = 0; i < 15; i++)
	{
		vector<double> b = {1.0+i, 2.0+i, 3.0+i, 4.0+i};
		pair<int, vector<double>> c{i, b};
		a.push_back(c);
	}
	//With 15 images and batch size 4, the last batch contains three images
	//while the others contain 4.
	obj.SGD(a, 2, 4, 5);
}
//Must manually check output for correctness.
void testUpdateSystem()
{
	DigitClassifier test("ExampleNeuralNetwork1.txt");
	labeledImages a;
	for(int i = 0; i < 15; i++)
	{
		vector<double> b = {1.0+i, 2.0+i, 3.0+i};
		pair<int, vector<double>> c{i, b};
		a.push_back(c);
	}
	test.updateSystem(a, 2);
}

//Must manually check output for correctness.

void testShuffleImagesImporved()
{
	vector<int> conditions{1,2,3};
	DigitClassifier test(conditions);
	labeledImages imgs= test.getImages("mnist_train_very_short.csv");
	test.shuffleImagesImproved(imgs);
	for(int i = 0; i < 10; i++)
	{
		for(double a : imgs[i].second)
			cout << a << " ";
		cout << endl;

	}

}

void testActivations(DigitClassifier & obj)
{
	/*
	 * expected activation values are:
	 * 0.731059, 0.880797, 0.952574, 0.982014
	 */
	vector<double> zVals{1, 2, 3, 4};
	cout << obj.activations(zVals);
}


/*int main()
{
	std::vector<int> conditions =
	{ 28 * 28, 15, 10 };
	DigitClassifier obj("train3.txt");
	//getImages works and has been tested separately.
	//testHadamard(obj); //Passed
	//testMultiplyMatrices(obj); //Passed
	//testTranspose(obj); //Passed
	//testExtractDoubles(obj); //Passed
	//testToStringAndReadIn(obj); //Passed
	//testFeedForwardOnce(); //Passed
	//testFillSystemRandomly(); //Passed
	//testLastLayerError(); //Passed
	//testBackpropagate(); //Passed
	//testShuffleImages(obj); //Passed
	//testSGD(obj); //Passed, though the updateSystem function was not tested yet.
	//testUpdateSystem();
	//testShuffleImagesImporved(); //Passed.
	//testActivations(obj);
	obj.updateSystem(obj.getImages("mnist_train_very_short.csv"), 3);

	cout << "tests passed" << endl;
}*/
