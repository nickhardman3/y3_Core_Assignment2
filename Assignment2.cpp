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

enum CellState { EMPTY = 0, TREE = 1, BURNING = 2, BURNT = 3 }; //defines the state for each cell in the grid

vector<int> read_grid_from_file(const string &filename, int &N) //if a file is provided, it is read and the grid size is determined
{
    ifstream inFile(filename);
    if (!inFile.is_open())
    {
        cerr << "Error: Cannot open " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); //MPI is aborted if the file cant open
    }
    vector<int> tokens;
    int token; //reads all integer tokens from the file
    while (inFile >> token)
    {
        tokens.push_back(token);
    }
    inFile.close();
    int token_count = tokens.size();
    int sqrt_count = static_cast<int>(sqrt(token_count));
    if (sqrt_count * sqrt_count == token_count) //tokens treated as a complete grid if total tokens form a square
    {
        N = sqrt_count;
        return tokens;
    }
    else
    {
        N = tokens[0]; //otherwise, the first token is taken as the grid size and the rest is data
        vector<int> grid(tokens.begin() + 1, tokens.end());
        if ((int)grid.size() != N * N)
        {
            cerr << "Error: Token count does not match grid size." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        return grid;
    }
}

vector<int> extract_local_grid(const vector<int>& global_grid, int N, int local_start, int local_rows) //a local grid from the global grid is extracted for each process
{
    vector<int> local_data(local_rows * N);
    for (int i = 0; i < local_rows; i++)
    {
        for (int j = 0; j < N; j++)
        {
            local_data[i * N + j] = global_grid[(local_start + i) * N + j]; //row of data from global grid is copied into the local grid
        }
    }
    return local_data;
}

vector<vector<int>> convert_1D_to_2D(const vector<int>& flat_grid, int rows, int cols) //1d vector converted into a 2d grid
{
    vector<vector<int>> grid(rows, vector<int>(cols));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            grid[i][j] = flat_grid[i * cols + j];
        }
    }
    return grid;
}

vector<int> flatten_grid(const vector<vector<int>>& grid) //flattens the grid into a 1d vector to gather results
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

vector<vector<int>> generate_local_grid(int local_rows, int N, double p) //generates a local grid with probability p that a tree (1) is produced
{
    vector<vector<int>> grid(local_rows, vector<int>(N, 0));
    for (int i = 0; i < local_rows; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double r = (double)rand() / RAND_MAX;
            grid[i][j] = (r < p) ? TREE : EMPTY;
        }
    }
    return grid;
}

void ignite_top_row(vector<vector<int>> &local_grid, int local_start) //top row of trees are set on fire (1 --> 2)
{
    if (local_start == 0 && !local_grid.empty())
    {
        int cols = local_grid[0].size();
        for (int j = 0; j < cols; j++)
        {
            if (local_grid[0][j] == TREE)
            {
                local_grid[0][j] = BURNING;
            }
        }
    }
}

bool simulate_step(vector<vector<int>> &local_grid, int N, int rank, int size) //simulates one step of the fire spread
{
    int local_rows = local_grid.size();
    vector<int> top_ghost(N, 0), bottom_ghost(N, 0); //ghost rows for the top and bottom neighbours
    MPI_Status status;
    if (rank > 0)
    {
        MPI_Sendrecv(local_grid[0].data(), N, MPI_INT, rank - 1, 0,
                     top_ghost.data(), N, MPI_INT, rank - 1, 1,
                     MPI_COMM_WORLD, &status); //exchanges the top row with the process above
    }
    if (rank < size - 1)
    {
        MPI_Sendrecv(local_grid[local_rows - 1].data(), N, MPI_INT, rank + 1, 1,
                     bottom_ghost.data(), N, MPI_INT, rank + 1, 0,
                     MPI_COMM_WORLD, &status); //exchanges the bottom row with the process below
    }
    vector<vector<int>> new_grid = local_grid; //copies the current grid to new_grid for updating states
    bool local_fire = false;
    for (int i = 0; i < local_rows; i++) //processes each cell in the local grid
    {
        for (int j = 0; j < N; j++)
        {
            int state = local_grid[i][j];
            if (state == BURNING)
            {
                new_grid[i][j] = BURNT; //burning trees become burnt (2 --> 3)
            }
            else if (state == TREE)
            {
                bool neighbour_burning = false;
                if (i == 0) //uses ghost row if on the top boundary of the local grid
                {
                    if (rank > 0 && top_ghost[j] == BURNING)
                    {
                        neighbour_burning = true;
                    }
                }
                else 
                {
                    if (local_grid[i - 1][j] == BURNING)
                    {
                        neighbour_burning = true;
                    }
                }
                if (!neighbour_burning) //uses ghost row if on the bottom boundary of the local grid
                {
                    if (i == local_rows - 1)
                    {
                        if (rank < size - 1 && bottom_ghost[j] == BURNING)
                        {
                            neighbour_burning = true;
                        }
                    }
                    else
                    {
                        if (local_grid[i + 1][j] == BURNING)
                        {
                            neighbour_burning = true;
                        }
                    }
                }
                if (!neighbour_burning && j > 0) //checks to see if the left neighbour is burning
                {
                    if (local_grid[i][j - 1] == BURNING)
                    {
                        neighbour_burning = true;
                    }
                }
                if (!neighbour_burning && j < N - 1) //same but for right neighbour
                {
                    if (local_grid[i][j + 1] == BURNING)
                    {
                        neighbour_burning = true;
                    }
                }
                if (neighbour_burning) //if a neighbour is burning, burn tree
                {
                    new_grid[i][j] = BURNING;
                    local_fire = true;
                }
            }
        }
    }
    local_grid = new_grid; //updates the grid with the new values
    return local_fire;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int N = 100; //default grid size
    double p = 0.6; //default probability
    int M = 1; //default number of runs
    string mode = "normal"; //mode: "normal" or "conv"
    string initFilename = "";
    string print_steps = "no"; //option to print the list of step counts at the end ("yes" or "no")
    //usage (no input file): mpirun -np <P> ./A2 N p M mode print_steps
    //usage (with input file): mpirun -np <P> ./A2 N p M mode initFilename print_steps
    if (argc >= 6 && argc < 7)
    {
        N = atoi(argv[1]);
        p = atof(argv[2]);
        M = atoi(argv[3]);
        mode = argv[4];
        print_steps = argv[5];
    }
    else if (argc >= 7)
    {
        N = atoi(argv[1]);
        p = atof(argv[2]);
        M = atoi(argv[3]);
        mode = argv[4];
        initFilename = argv[5];
        print_steps = argv[6];
        M = 1; //only one run if input file
    }
    bool has_input_file = !initFilename.empty();
    bool is_conv_mode = (mode == "conv");
    double total_steps = 0.0;
    double total_time = 0.0;
    int total_reached = 0;
    vector<int> final_global_data;
    int run_count = 0;
    double old_avg_steps = 0.0;
    double tolerance = 0.0005;
    bool converged = false;
    int max_runs = 50000;
    vector<int> forest_run_steps; //record steps for each forest run
    while (true) //main simulation loop for multiple runs/convergence
    {
        run_count++;
        if (has_input_file && run_count > 1) //break conditions
        {
            break;
        }
        if (!has_input_file && !is_conv_mode && run_count > M)
        {
            break;
        }
        if (!has_input_file && is_conv_mode && run_count > max_runs)
        {
            if (rank == 0)
            {
                cerr << "[Convergence Mode] Warning: reached maximum amount of runs without converging.\n";
            }
            break;
        }
        int rows_per_proc = N / size; //determine the local grid boundaries for each process 
        int remainder = N % size;
        int local_start, local_end;
        if (rank < remainder)
        {
            local_start = rank * (rows_per_proc + 1);
            local_end = local_start + rows_per_proc + 1;
        }
        else
        {
            local_start = rank * rows_per_proc + remainder;
            local_end = local_start + rows_per_proc;
        }
        int local_rows = local_end - local_start;
        vector<vector<int>> local_grid;
        if (has_input_file) //initialize the local grid: either read from file or generate randomly
        {
            vector<int> global_grid_flat;
            if (rank == 0)
            {
                global_grid_flat = read_grid_from_file(initFilename, N);
            }
            MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank != 0)
            {
                global_grid_flat.resize(N * N);
            }
            MPI_Bcast(global_grid_flat.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
            vector<int> local_data = extract_local_grid(global_grid_flat, N, local_start, local_rows);
            local_grid = convert_1D_to_2D(local_data, local_rows, N);
        }
        else
        {
            srand((unsigned)time(NULL) + run_count * 10000 + rank * 777); //seed the random number generator uniquely for each process and run
            local_grid = generate_local_grid(local_rows, N, p);
        }
        ignite_top_row(local_grid, local_start); //ignites top row
        double start_time = MPI_Wtime(); //starts timing
        int steps = 0;
        bool global_fire = true;
        while (global_fire) //run until no more trees burning (no 2s)
        {
            bool local_fire = simulate_step(local_grid, N, rank, size);
            int local_fire_int = local_fire ? 1 : 0;
            int global_fire_int = 0;
            MPI_Allreduce(&local_fire_int, &global_fire_int, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); //combines fire status across all processes to check if the simulation should continue
            global_fire = (global_fire_int > 0);
            steps++;
        }
        double end_time = MPI_Wtime();
        double elapsed = end_time - start_time;
        int local_size = local_rows * N; //prepares the local grid data for gathering into the global grid
        vector<int> local_data = flatten_grid(local_grid);
        vector<int> recv_counts(size), displs(size);
        if (rank == 0)
        {
            for (int r = 0; r < size; r++) //calculates the receive counts and displacements for MPI_Gatherv
            {
                int r_local_start, r_local_end, r_local_rows;
                if (r < remainder)
                {
                    r_local_start = r * (rows_per_proc + 1);
                    r_local_end = r_local_start + rows_per_proc + 1;
                }
                else
                {
                    r_local_start = r * rows_per_proc + remainder;
                    r_local_end = r_local_start + rows_per_proc;
                }
                r_local_rows = r_local_end - r_local_start;
                recv_counts[r] = r_local_rows * N;
            }
            displs[0] = 0;
            for (int r = 1; r < size; r++)
            {
                displs[r] = displs[r - 1] + recv_counts[r - 1];
            }
        }
        vector<int> global_data; //gathers all of the local grids into the global grid on process 0
        if (rank == 0)
        {
            global_data.resize(N * N);
        }
        MPI_Gatherv(local_data.data(), local_size, MPI_INT, global_data.data(), recv_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
        bool reached_bottom = false; //checks if the fire has reached the bottom row of the global grid
        if (rank == 0)
        {
            for (int j = 0; j < N; j++)
            {
                int cell = global_data[(N - 1) * N + j];
                if (cell == BURNING || cell == BURNT)
                {
                    reached_bottom = true;
                    break;
                }
            }
        }
        MPI_Bcast(&reached_bottom, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        if (rank == 0) //process 0 collects the results and updates the performance metrics
        {
            final_global_data = global_data;
            total_time += elapsed;
            total_steps += steps;
            total_reached += (reached_bottom ? 1 : 0);
            forest_run_steps.push_back(steps); //record this run's step count
        }
        if (is_conv_mode && !has_input_file && rank == 0 && run_count >= 10) //checks the convergence criteria in convergence mode
        {
            double new_avg_steps = total_steps / run_count;
            double diff = fabs(new_avg_steps - old_avg_steps);
            old_avg_steps = new_avg_steps;
            if (diff < 0.0005)
            {
                converged = true;
            }
        }
        if (is_conv_mode && !has_input_file)
        {
            bool stop_now = false;
            if (rank == 0)
            {
                stop_now = converged;
            }
            MPI_Bcast(&stop_now, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
            if (stop_now)
            {
                break;
            }
        }
    }
    if (rank == 0) //final output and file writing 
    {
        int runs_done = (has_input_file ? 1 : run_count -1);
        if (print_steps == "yes")
        {
            for (size_t i = 0; i < forest_run_steps.size(); i++) //prints only the list of step count
                cout << forest_run_steps[i] << "\n";
        }
        else
        {
            double avg_steps = (runs_done > 0) ? (total_steps / runs_done) : 0.0; //calculates the average steps, average time, and the percentage of runs reaching the bottom
            double avg_time = (runs_done > 0) ? (total_time / runs_done) : 0.0;
            double reach_percent = (runs_done > 0) ? (double)total_reached * 100.0 / runs_done : 0.0;
            if (reach_percent > 100)
            {
                (reach_percent) = 100;
            }
            if (has_input_file)
            {
                cout << "\n[Single-Run: File Provided]\n";
            }
            else if (is_conv_mode)
            {
                cout << "\n[Convergence Mode]\n";
                cout << "Runs performed until convergence: " << runs_done << endl;
            }
            else
            {
                cout << "\n[Normal Mode]\n";
                cout << "Total runs performed: " << runs_done << endl;
            }
            cout << "Average steps: " << avg_steps << endl;
            cout << "Average time:  " << avg_time << " seconds." << endl;
            cout << "Fire reached bottom in " << reach_percent << "% of runs.\n" << endl;
        }
        ofstream outFile("finalforest"); //writes the final grid
        if (outFile.is_open())
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    outFile << final_global_data[i * N + j] << " ";
                }
                outFile << "\n";
            }
            outFile.close();
        }
        else
        {
            cerr << "Error: Unable to write finalforest." << endl;
        }
        ofstream listFile("forest_run_steps.txt"); //writes the list of forest run step counts
        if (listFile.is_open())
        {
            for (size_t i = 0; i < forest_run_steps.size(); i++)
                listFile << forest_run_steps[i] << "\n";
            listFile.close();
        }
    }
    MPI_Finalize();
    return 0;
}
