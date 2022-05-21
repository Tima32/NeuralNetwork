#include <random>
#include <iostream>
#include "Net.hpp"

using namespace std;

double Net::Neyron::Sigmoid(double d)
{
	return 1 / (1 + exp(-d * 0.9));
}
double Net::Neyron::SigmoidDX(double d)
{
	return Sigmoid(d) * (double(1.0) - Sigmoid(d));
}

double Net::Neyron::Threshold(double d)
{
	return d >= 0.7;
}
double Net::Neyron::ThresholdDX(double d)
{
	return 0.01;
}

double Net::Neyron::Tanh(double d)
{
	return tanh(d);
}
double Net::Neyron::TanhDX(double d)
{
	return 1.0 - d * d;
}


Net::Net(const std::vector<size_t>& size)
{
	net.resize(size.size());

	for (size_t l = 0; l < size.size(); l++)
	{
		net[l].resize(size[l]);
		if ( l < size.size() - 1)//игнорируем выходной слой
			for (size_t n = 0; n < size[l]; n++)
				net[l][n].weights.resize(size[l + 1]);
	}

	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<size_t> dist(0, 1000);

	for (size_t l = 0; l < net.size(); l++)
		for (size_t n = 0; n < net[l].size(); n++)
			for (size_t nw = 0; nw < net[l][n].weights.size(); nw++)
			{
				net[l][n].weights[nw] = dist(gen) / 1000.f;
			}
}

const std::vector<Net::Neyron>& Net::predict()
{
	for (size_t l = 1; l < net.size(); l++)
	{
		auto& level = net[l];
		for (size_t n = 0; n < level.size(); n++)
		{
			double sum{ 0 };
			auto& previous_level = net[l - 1];
			for (size_t nw = 0; nw < previous_level.size(); nw++)
			{
				sum += (previous_level[nw].output * previous_level[nw].weights[n]);
			}
			auto& neuron = level[n];
			neuron.output = neuron.active(sum);
		}
	}

	return net[net.size() - 1];
}
void Net::train(double learning_rate, size_t max_epochs, double max_mse, const std::vector<std::vector<double>>& in, const std::vector<std::vector<double>>& out)
{
	double mse = 999, old_mse = 999;
	size_t i = 0;
	while (mse > max_mse && i < max_epochs)
	{
		mse = 0;
		for (size_t c = 0; c < in.size(); c++)
		{
			for (size_t n = 0; n < in[c].size(); n++)
			{
				net[0][n].output = in[c][n];
			}

			predict();
			train(learning_rate, out[c]);

			for (size_t i = 0; i < net[net.size() - 1].size(); i++)
				mse += pow(net[net.size() - 1][i].output - out[c][i], 2);
		}
		
		if (i % 100000 == 0)
		{
			cout << "mse: " << mse << endl;
			if (old_mse - mse < max_mse / 10)
			{
				cout << "RE" << endl;
				random_device rd;
				mt19937 gen(rd());
				uniform_int_distribution<size_t> dist(0, 1000);

				for (size_t l = 0; l < net.size(); l++)
					for (size_t n = 0; n < net[l].size(); n++)
						for (size_t nw = 0; nw < net[l][n].weights.size(); nw++)
						{
							net[l][n].weights[nw] = dist(gen) / 1000.f;
						}
			}
			old_mse = mse;
		}
		i++;
	}
	cout << "mse: " << mse << endl;
}
void Net::train(double learning_rate, const std::vector<double>& out)
{
	auto old_net = net;

	vector<vector<double>> previous_level_delta;
	previous_level_delta.resize(net.size());
	for (size_t l = 0; l < previous_level_delta.size(); l++)
		previous_level_delta[l].resize(net[l].size(), 999);

	for (size_t l = net.size() - 1; l != 0; l--)
	{
		for (size_t n = 0; n < net[l].size(); n++)
		{
			double error{ 0 };

			if (l == net.size() - 1)
				error = (net[l][n].output - out[n]) * net[l][n].output * (1 - net[l][n].output);
			else
			{
				for (size_t wn = 0; wn < net[l][n].weights.size(); wn++)
				{
					error += ((previous_level_delta[l + 1][wn] * old_net[l][n].weights[wn]) * old_net[l][n].output * (1 - old_net[l][n].output));
				}
			}

			double weights_delta = error;
			previous_level_delta[l][n] = weights_delta;

			for (size_t wn = 0; wn < net[l - 1].size(); wn++)
			{
				double weight = net[l - 1][wn].weights[n] - net[l - 1][wn].output * weights_delta * learning_rate;
				net[l - 1][wn].weights[n] = weight;
			}
		}
	}
}