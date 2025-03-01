#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
using namespace std;

enum CellState { EMPTY = 0, TREE = 1, BURNING = 2, BURNT = 3 };

void initializeGrid(vector<vector<int>> &grid, int N, double p) 
{
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            double r = (double)rand() / RAND_MAX;
            grid[i][j] = (r < p) ? TREE : EMPTY;
        }
    }
    
    for (int j = 0; j < N; j++) 
    {
        if (grid[0][j] == TREE)
            grid[0][j] = BURNING;
    }
}

bool simulateStep(const vector<vector<int>> &current, vector<vector<int>> &next, int N) 
{
    bool fireStill = false;
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            int state = current[i][j];
            if (state == BURNING) 
            {
                next[i][j] = BURNT;
            } 
            else if (state == TREE) 
            {
                bool neighborBurning = false;
                if (i > 0 && current[i - 1][j] == BURNING) neighborBurning = true;
                if (i < N - 1 && current[i + 1][j] == BURNING) neighborBurning = true;
                if (j > 0 && current[i][j - 1] == BURNING) neighborBurning = true;
                if (j < N - 1 && current[i][j + 1] == BURNING) neighborBurning = true;
                if (neighborBurning) 
                {
                    next[i][j] = BURNING;
                    fireStill = true;
                } 
                else 
                {
                    next[i][j] = TREE;
                }
            } 
            else 
            {
                next[i][j] = state;
            }
        }
    }
    return fireStill;
}

int main() 
{
    int N = 100;          
    double p = 0.6;       
    srand(time(NULL));

    vector<vector<int>> grid(N, vector<int>(N, EMPTY));
    vector<vector<int>> nextGrid(N, vector<int>(N, EMPTY));

    initializeGrid(grid, N, p);

    auto start = chrono::high_resolution_clock::now();
    int steps = 0;
    bool fireActive = true;
    while (fireActive) 
    {
        fireActive = simulateStep(grid, nextGrid, N);
        grid.swap(nextGrid);
        steps++;
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    bool reachedBottom = false;
    for (int j = 0; j < N; j++) 
    {
        if (grid[N - 1][j] == BURNT) 
        {
            reachedBottom = true;
            break;
        }
    }

    cout << "Simulation completed in " << steps << " steps." << endl;
    cout << "Fire reached bottom: " << (reachedBottom ? "Yes" : "No") << endl;
    cout << "Time taken: " << elapsed.count() << " seconds." << endl;
    return 0;
}
