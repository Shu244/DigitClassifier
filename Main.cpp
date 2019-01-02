/*
 * dataset obtained from https://pjreddie.com/projects/mnist-in-csv/
 * Author: Shuhao Lai
 * Date: 12/27/2018
 * Main.cpp
 */
#include <iostream>
#include <ctime>
#include "DigitClassifier.h"

int main()
{
	clock_t begin = clock();

	std::vector<int> conditions = {784, 30, 10};
	DigitClassifier test(conditions);
	std::cout << "Training the neural network." << std::endl;
	test.train("mnist_train.csv", 30, 20, 3);
	std::cout << "Training complete." << std::endl;
	test.toString("Trained1.txt");

	std::clock_t trained = clock();

	test.evaluate("mnist_test.csv");
	std::cout << "Evaluation complete." << std::endl;

	std::clock_t done = clock();

	double timeToTrain = (double(trained-begin) / CLOCKS_PER_SEC)/60;
	double timeToFinish = (double(done-begin) / CLOCKS_PER_SEC)/60;

	std::cout << "Time to train: " << timeToTrain << " minutes" << std::endl;
	std::cout << "Time to finish everything: " << timeToFinish << " minutes" << std::endl;
}


