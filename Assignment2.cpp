#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
using namespace std;

enum CellState { EMPTY = 0, TREE = 1, BURNING = 2, BURNT = 3 };

vector<int> readGridFromFile(const string &filename, int &N) 
{
    ifstream inFile(filename);
    if (!inFile.is_open()) 
    {
        cerr << "Error: Cannot open " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    vector<int> tokens;
    int token;
    while (inFile >> token) 
    {
        tokens.push_back(token);
    }
    inFile.close();
    int tokenCount = tokens.size();
    int sqrtCount = static_cast<int>(sqrt(tokenCount));
    if (sqrtCount * sqrtCount == tokenCount) 
    {
        N = sqrtCount;
        return tokens;
    } 
    else 
    {
        N = tokens[0];
        vector<int> grid(tokens.begin() + 1, tokens.end());
        if ((int)grid.size() != N * N) 
        {
            cerr << "Error: Token count does not match grid size." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        return grid;
    }
}

vector<int> extractLocalGrid(const vector<int>& globalGrid, int N, int localStart, int localRows) 
{
    vector<int> localData(localRows * N);
    for (int i = 0; i < localRows; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            localData[i * N + j] = globalGrid[(localStart + i) * N + j];
        }
    }
    return localData;
}

vector<vector<int>> convert1Dto2D(const vector<int>& flatGrid, int rows, int cols) 
{
    vector<vector<int>> grid(rows, vector<int>(cols));
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            grid[i][j] = flatGrid[i * cols + j];
        }
    }
    return grid;
}

vector<int> flattenGrid(const vector<vector<int>>& grid) 
{
    int rows = grid.size();
    int cols = (rows > 0) ? grid[0].size() : 0;
    vector<int> flat(rows * cols);
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            flat[i * cols + j] = grid[i][j];
        }
    }
    return flat;
}

vector<vector<int>> generateLocalGrid(int localRows, int N, double p) 
{
    vector<vector<int>> grid(localRows, vector<int>(N, 0));
    for (int i = 0; i < localRows; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            double r = (double)rand() / RAND_MAX;
            grid[i][j] = (r < p) ? TREE : EMPTY;
        }
    }
    return grid;
}

void igniteTopRow(vector<vector<int>> &localGrid, int localStart) 
{
    if (localStart == 0 && !localGrid.empty()) 
    {
        int cols = localGrid[0].size();
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
    vector<int> topGhost(N, EMPTY), bottomGhost(N, EMPTY);
    MPI_Status status;

    if (rank > 0) 
    {
        MPI_Sendrecv(localGrid[0].data(), N, MPI_INT, rank - 1, 0,
                     topGhost.data(), N, MPI_INT, rank - 1, 1,
                     MPI_COMM_WORLD, &status);
    }
    if (rank < size - 1) 
    {
        MPI_Sendrecv(localGrid[localRows - 1].data(), N, MPI_INT, rank + 1, 1,
                     bottomGhost.data(), N, MPI_INT, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
    }

    vector<vector<int>> newGrid = localGrid;
    bool localFire = false;
    for (int i = 0; i < localRows; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            int state = localGrid[i][j];
            if (state == BURNING) 
            {
                newGrid[i][j] = BURNT;
            } else if (state == TREE) 
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
                    if (i == localRows - 1) 
                    {
                        if (rank < size - 1 && bottomGhost[j] == BURNING)
                            neighborBurning = true;
                    } else {
                        if (localGrid[i + 1][j] == BURNING)
                            neighborBurning = true;
                    }
                }
                if (!neighborBurning && j > 0) 
                {
                    if (localGrid[i][j - 1] == BURNING)
                        neighborBurning = true;
                }
                if (!neighborBurning && j < N - 1) 
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
    int M = 1;               
    string mode = "normal";   
    string initFilename = "";

    if (argc >= 5) 
    {
        N = atoi(argv[1]);
        p = atof(argv[2]);
        M = atoi(argv[3]);
        mode = argv[4];
    }
    if (argc >= 6) 
    {
        initFilename = argv[5];
        M = 1;  
    }

    bool hasInputFile = !initFilename.empty();
    bool isConvMode   = (mode == "conv");

    double totalSteps  = 0.0;
    double totalTime   = 0.0;
    int totalReached   = 0;
    vector<int> finalGlobalData;

    int runCount = 0;
    double oldAvgSteps = 0.0;
    double tolerance   = 1e-5;
    bool converged     = false;
    int maxRuns        = 50000; 

    while (true) {
        runCount++;

        if (hasInputFile && runCount > 1) 
        {
            break;
        }
        if (!hasInputFile && !isConvMode && runCount > M) 
        {
            break;
        }
        if (!hasInputFile && isConvMode && runCount > maxRuns) 
        {
            if (rank == 0) 
            {
                cerr << "[Convergence Mode] Warning: reached maxRuns without converging.\n";
            }
            break;
        }

        int rowsPerProc = N / size;
        int remainder = N % size;
        int localStart, localEnd;
        if (rank < remainder) 
        {
            localStart = rank * (rowsPerProc + 1);
            localEnd   = localStart + rowsPerProc + 1;
        } 
        else 
        {
            localStart = rank * rowsPerProc + remainder;
            localEnd   = localStart + rowsPerProc;
        }
        int localRows = localEnd - localStart;

        vector<vector<int>> localGrid;
        if (hasInputFile) 
        {
            vector<int> globalGridFlat;
            if (rank == 0) 
            {
                globalGridFlat = readGridFromFile(initFilename, N);
            }
            MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank != 0) 
            {
                globalGridFlat.resize(N * N);
            }
            MPI_Bcast(globalGridFlat.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
            vector<int> localData = extractLocalGrid(globalGridFlat, N, localStart, localRows);
            localGrid = convert1Dto2D(localData, localRows, N);
        } 
        else 
        {
            srand((unsigned)time(NULL) + runCount*10000 + rank*777);
            localGrid = generateLocalGrid(localRows, N, p);
        }
        igniteTopRow(localGrid, localStart);

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
        double elapsed = endTime - startTime;

        int localSize = localRows * N;
        vector<int> localData = flattenGrid(localGrid);

        vector<int> recvCounts(size), displs(size);
        if (rank == 0) 
        {
            for (int r = 0; r < size; r++) 
            {
                int rLocalStart, rLocalEnd, rLocalRows;
                if (r < remainder) 
                {
                    rLocalStart = r * (rowsPerProc + 1);
                    rLocalEnd   = rLocalStart + rowsPerProc + 1;
                } 
                else 
                {
                    rLocalStart = r * rowsPerProc + remainder;
                    rLocalEnd   = rLocalStart + rowsPerProc;
                }
                rLocalRows = rLocalEnd - rLocalStart;
                recvCounts[r] = rLocalRows * N;
            }
            displs[0] = 0;
            for (int r = 1; r < size; r++) 
            {
                displs[r] = displs[r - 1] + recvCounts[r - 1];
            }
        }
        vector<int> globalData;
        if (rank == 0) 
        {
            globalData.resize(N * N);
        }
        MPI_Gatherv(localData.data(), localSize, MPI_INT,
                    globalData.data(), recvCounts.data(), displs.data(), MPI_INT,
                    0, MPI_COMM_WORLD);

        bool reachedBottom = false;
        if (rank == 0) 
        {
            for (int j = 0; j < N; j++) 
            {
                int cell = globalData[(N - 1)*N + j];
                if (cell == BURNING || cell == BURNT) 
                {
                    reachedBottom = true;
                    break;
                }
            }
        }
        MPI_Bcast(&reachedBottom, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

        if (rank == 0) 
        {
            finalGlobalData = globalData; 
            totalTime   += elapsed;
            totalSteps  += steps;
            totalReached += (reachedBottom ? 1 : 0);

            if (!isConvMode && !hasInputFile) 
            {
                cout << "Run " << runCount << " of " << M << ":\n";
                cout << "Steps: " << steps << "\n";
                cout << "Time:  " << elapsed << " seconds.\n";
                cout << "Fire reached bottom: " << (reachedBottom ? "Yes" : "No") << "\n\n";
            }
        }

        if (isConvMode && !hasInputFile) 
        {
            bool stopNow = false;
            if (rank == 0) 
            {
                if (runCount >= 3) 
                {
                    double newAvgSteps = totalSteps / runCount;
                    double diff = fabs(newAvgSteps - oldAvgSteps);
                    if (diff < 1e-5) 
                    {
                        stopNow = true;
                        converged = true;
                    }
                    oldAvgSteps = newAvgSteps;
                }
            }
            MPI_Bcast(&stopNow, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
            if (stopNow) 
            {
                break;
            }
        }
    } 

    if (rank == 0) 
    {
        int runsDone = (hasInputFile ? 1 : (isConvMode ? (converged ? 0 : 0) : M));
        if (hasInputFile) 
        {
            runsDone = 1;
        } 
        else 
        {
            runsDone = runCount;
        }

        double avgSteps = (runsDone > 0) ? (totalSteps / runsDone) : 0.0;
        double avgTime  = (runsDone > 0) ? (totalTime / runsDone) : 0.0;
        double reachPercent = (runsDone > 0) ? (double)totalReached * 100.0 / runsDone : 0.0;

        if (hasInputFile) 
        {
            cout << "\n[Single-Run: File Provided]\n";
        } 
        else if (isConvMode) 
        {
            cout << "\n[Convergence Mode]\n";
            cout << "Runs performed until average steps converged: " << runsDone << endl;
        } 
        else 
        {
            cout << "\n[Normal Mode]\n";
            cout << "Total runs performed: " << runsDone << endl;
        }

        cout << "Average steps: " << avgSteps << endl;
        cout << "Average time:  " << avgTime << " seconds.\n";
        cout << "Fire reached bottom in " << reachPercent << "% of runs.\n";

        ofstream outFile("finalforest");
        if (outFile.is_open()) 
        {
            for (int i = 0; i < N; i++) 
            {
                for (int j = 0; j < N; j++) 
                {
                    outFile << finalGlobalData[i * N + j] << " ";
                }
                outFile << "\n";
            }
            outFile.close();
        } 
        else 
        {
            cerr << "Error: Unable to write finalforest.\n";
        }
    }

    MPI_Finalize();
    return 0;
}
