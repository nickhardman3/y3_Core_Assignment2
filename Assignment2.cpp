#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;

enum CellState { EMPTY = 0, TREE = 1, BURNING = 2, BURNT = 3 };

void initializeGrid(vector<vector<int>> &localGrid, int N, double p, int rank, int size, int localStart, int localEnd) {
    int localRows = localGrid.size();
    int cols = N;
    for (int i = 0; i < localRows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            double r = (double)rand() / RAND_MAX;
            localGrid[i][j] = (r < p) ? TREE : EMPTY;
        }
    }
    if (localStart == 0) 
    {
        for (int j = 0; j < cols; j++) 
        {
            if (localGrid[0][j] == TREE)
                localGrid[0][j] = BURNING;
        }
    }
}

bool simulateStep(vector<vector<int>> &localGrid, int N, int rank, int size) 
{
    int localRows = localGrid.size();
    int cols = N;
    vector<int> topGhost(cols, 0), bottomGhost(cols, 0);
    MPI_Status status;
    if (rank > 0) 
    {
        MPI_Sendrecv(localGrid[0].data(), cols, MPI_INT, rank - 1, 0,
                     topGhost.data(), cols, MPI_INT, rank - 1, 1,
                     MPI_COMM_WORLD, &status);
    }
    if (rank < size - 1) 
    {
        MPI_Sendrecv(localGrid[localRows - 1].data(), cols, MPI_INT, rank + 1, 1,
                     bottomGhost.data(), cols, MPI_INT, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
    }
    vector<vector<int>> newGrid = localGrid;
    bool localFire = false;
    for (int i = 0; i < localRows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            if (localGrid[i][j] == BURNING) 
            {
                newGrid[i][j] = BURNT;
            }
            else if (localGrid[i][j] == TREE) 
            {
                bool neighborBurning = false;
                if (i == 0) 
                {
                    if (rank > 0 && topGhost[j] == BURNING)
                        neighborBurning = true;
                } 
                else 
                {
                    if (localGrid[i - 1][j] == BURNING)
                        neighborBurning = true;
                }
                if (!neighborBurning) 
                {
                    if (i == localRows - 1) {
                        if (rank < size - 1 && bottomGhost[j] == BURNING)
                            neighborBurning = true;
                    } 
                    else 
                    {
                        if (localGrid[i + 1][j] == BURNING)
                            neighborBurning = true;
                    }
                }
                if (!neighborBurning && j > 0) 
                {
                    if (localGrid[i][j - 1] == BURNING)
                        neighborBurning = true;
                }
                if (!neighborBurning && j < cols - 1) 
                {
                    if (localGrid[i][j + 1] == BURNING)
                        neighborBurning = true;
                }
                if (neighborBurning) 
                {
                    newGrid[i][j] = BURNING;
                    localFire = true;
                }
            }
        }
    }
    localGrid = newGrid;
    return localFire;
}

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int N = 100;
    double p = 0.6;
    if (argc >= 3) 
    {
        N = atoi(argv[1]);
        p = atof(argv[2]);
    }
    int rowsPerProc = N / size;
    int remainder = N % size;
    int localStart, localEnd;
    if (rank < remainder) 
    {
        localStart = rank * (rowsPerProc + 1);
        localEnd = localStart + rowsPerProc + 1;
    } 
    else 
    {
        localStart = rank * rowsPerProc + remainder;
        localEnd = localStart + rowsPerProc;
    }
    int localRows = localEnd - localStart;
    vector<vector<int>> localGrid(localRows, vector<int>(N, EMPTY));
    srand(time(NULL) + rank);
    initializeGrid(localGrid, N, p, rank, size, localStart, localEnd);
    double startTime = MPI_Wtime();
    int steps = 0;
    bool globalFire = true;
    while (globalFire) 
    {
        bool localFire = simulateStep(localGrid, N, rank, size);
        int localFireInt = localFire ? 1 : 0;
        int globalFireInt = 0;
        MPI_Allreduce(&localFireInt, &globalFireInt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        globalFire = (globalFireInt > 0);
        steps++;
    }
    double endTime = MPI_Wtime();
    bool reachedBottom = false;
    if (rank == size - 1) 
    {
        for (int j = 0; j < N; j++) 
        {
            if (localGrid[localRows - 1][j] == BURNING) 
            {
                reachedBottom = true;
                break;
            }
        }
    }
    MPI_Bcast(&reachedBottom, 1, MPI_CXX_BOOL, size - 1, MPI_COMM_WORLD);
    if (rank == 0) 
    {
        cout << "Simulation completed in " << steps << " steps." << endl;
        cout << "Fire reached bottom: " << (reachedBottom ? "Yes" : "No") << endl;
        cout << "Time taken: " << (endTime - startTime) << " seconds." << endl;
    }
    MPI_Finalize();
    return 0;
}
