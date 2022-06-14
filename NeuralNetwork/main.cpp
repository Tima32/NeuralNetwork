#include <iostream>

#include "Net.hpp"

using namespace std;

double out(double d)
{
	return d;
}

int main()
{
	Net nn({ 4, 4, 3 });

	//nn.net[2][0].active = Net::Neyron::Threshold;


	vector<vector<double>> t_case{
		{1, 0, 0, 0},
		//{1, 0, 0, 1},
		{1, 0, 1, 0},
		{1, 0, 1, 1},
		{1, 1, 0, 0},
		{1, 1, 0, 1},
		{1, 1, 1, 0},
		//{1, 1, 1, 1},
	};
	vector<vector<double>> t_result{
		{1, 1, 1},
		//{1, 1, 0},
		{1, 0, 1},
		{1, 0, 0},
		{0, 1, 1},
		{0, 1, 0},
		{0, 0, 1},
		//{0, 0, 0},
	};

	cout << fixed;
	cout.precision(6);

	nn.train(1, 1000000, 0.001, t_case, t_result);

	for (size_t c = 0; c < t_case.size(); c++)
	{
		cout << "{ ";
		for (size_t n = 0; n < t_case[c].size(); n++)
		{
			nn.net[0][n].output = t_case[c][n];
			cout << out(t_case[c][n]) << " ";
		}
		cout << "}";
		
		auto& r = nn.predict();
		cout << " -> { ";
		for (auto n : r)
			cout << out(n.output) << " ";
		cout << "}\n";
	}

	cout << "Test 1" << endl;
	{
		vector<double> in = {1, 1, 1, 1};
		cout << "{ ";
		for (size_t n = 0; n < in.size(); n++)
		{
			nn.net[0][n].output = in[n];
			cout << out(in[n]) << " ";
		}
		cout << "}";
		
		auto& r = nn.predict();
		cout << " -> { ";
		for (auto n : r)
			cout << out(n.output) << " ";
		cout << "}\n";
	}

	cout << "Test 2" << endl;
	{
		vector<double> in = {1, 0, 0, 1};
		cout << "{ ";
		for (size_t n = 0; n < in.size(); n++)
		{
			nn.net[0][n].output = in[n];
			cout << out(in[n]) << " ";
		}
		cout << "}";
		
		auto& r = nn.predict();
		cout << " -> { ";
		for (auto n : r)
			cout << out(n.output) << " ";
		cout << "}\n";
	}
	return 0;
}
